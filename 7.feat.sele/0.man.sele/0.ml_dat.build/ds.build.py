# rebuild dataset with selected important features
# Probably could be used to shuffle subset of features in the model.
# Zilin Song, 28SEP2019
#

import numpy as np
import subprocess as subp
import sys as sys

def get_bond_labels(file_dir):
    '''get labels of bonds from file_dir,
    all bond list and selected feature list.
    '''
    lines = open(file_dir, 'r').readlines()
    all_bond_list = []
    for line in lines:
        all_bond_list.append(line.split()[0].replace('\n', ''))
    return all_bond_list

def get_feature_indices(all_features, selected_features):
    '''get the indices for selected features,
    from the all feature list.
    '''
    indice_list = []
    
    for feature in selected_features:
        i = 0

        for label in all_features:
            
            if label == feature:
                indice_list.append(i)
            
            i+=1
    
    return indice_list

def retain_features(all_lists, feature_indices):
    '''generate new feature list with feature indices
    '''

    X_selected_feature = []

    for all_list in all_lists:
        temp_list = []

        for feature_indice in feature_indices:
            temp_list.append(all_list[feature_indice])

        X_selected_feature.append(temp_list)
    
    return X_selected_feature

def get_ds_filename(ds_id, which_ds):
    return 'ds_{0}.X_{1}'.format(str(ds_id), which_ds)

def main(basis):

    input_dir = '../../../6.data.prepare/3.dataset.conclude/{0}.path_dat'.format(basis)
    output_dir = '{0}.ml_dat'.format(basis)

    subp.call('mkdir -p {0}'.format(output_dir), shell=True)
    subp.call('cp {0}/pathid.assigned.log {1}/.'.format(input_dir, output_dir), shell=True)
    subp.call('cp ./features.man.dat {0}/.'.format(output_dir), shell=True)

    all_labels = get_bond_labels('{0}/{1}.bond_labels.dat'.format(input_dir, basis))
    feature_labels = get_bond_labels('./features.man.dat')

    X_indices = get_feature_indices(all_labels, feature_labels)
    
    ds_names = ['test', 'train']

    for i in range(18):
        current_outputdir = '{0}/ds_{1}'.format(output_dir, str(i))
        subp.call('mkdir -p {0}'.format(current_outputdir), shell=True)

        for ds_name in ds_names:
            # y
            subp.call('cp {0}/ds_{1}.y_*.npy {2}/.'.format(input_dir, str(i), current_outputdir), shell=True)

            # group_labels for logo cv
            subp.call('cp {0}/ds_{1}.group_labels.npy {2}/.'.format(input_dir, str(i), current_outputdir), shell=True)
            
            # X
            X_ds = np.load('{0}/{1}.npy'.format(input_dir, get_ds_filename(i, ds_name))).tolist()
            refined_X_ds = retain_features(X_ds, X_indices)
            np.save('{0}/{1}.npy'.format(current_outputdir, get_ds_filename(i, ds_name)), refined_X_ds)

    ### full set for training.
    # y & group labels
    subp.call('cp {0}/ds_all.y.npy {1}/ds_all.y_train.npy'.format(input_dir, output_dir), shell=True)
    subp.call('cp {0}/ds_all.group_labels.npy {1}/.'.format(input_dir, output_dir), shell=True)

    X_ds = np.load('{0}/ds_all.X.npy'.format(input_dir, )).tolist()
    refined_X_ds = retain_features(X_ds, X_indices)
    np.save('{0}/ds_all.X_train.npy'.format(output_dir, ), refined_X_ds)

if __name__ == "__main__":

    for basis in ['sccdftb', '631G', '631++Gdp', '631+Gd', 'dftd631++Gdp', '6311++Gdp', 'dftd6311++Gdp']: 
        main(basis)
