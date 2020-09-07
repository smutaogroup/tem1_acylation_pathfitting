# Shuffle features of the test dataset.
# Zilin Song 10NOV2019
# 

import subprocess as subp
import numpy as numpy

def get_ds(basis, ds_id):
    '''get interested dataset
    '''
    indir = '../0.ml_dat.build/{0}.ml_dat/ds_{1}/ds_{1}.X_train.npy'.format(basis, ds_id)
    return numpy.load(indir).tolist()

def shuffle_feature(ds, shff_feat_ids, ):
    '''shuffle feature from feature_id in ds.
    shff_feat_ids must be a list.
    '''

    shff_ds = []

    for d in ds:
        shff_d = []

        for i in range(len(d)):
            
            if i in shff_feat_ids:
                shff_d.append(0)

            else:
                shff_d.append(d[i])

        shff_ds.append(shff_d)
    
    return shff_ds

def main(basis, ds_id, ):
    # list of lists of features to shuffle.
    features_to_shuffle = [
        [0, 2, ],
        [9, 10, ],
        [3, 4, 5, ],
        [6, 7, 8, ],
        [1, ],
        [14, ],
        [11, ],
        [12, ],
        [13, ],
    ]
    output_dir = './{0}.shff_dat/ds_{1}'.format(basis, ds_id, )
    subp.call('mkdir -p {0}'.format(output_dir), shell=True)

    full_ds = get_ds(basis, ds_id, )

    for sf in range(len(features_to_shuffle)):
        shff_ds = shuffle_feature(full_ds, features_to_shuffle[sf], )
        numpy.save('{0}/ds_{1}.shff_{2}.X_train.npy'.format(output_dir, ds_id, sf, ), shff_ds)

if __name__ == '__main__':

    for i in range(18):
        for basis in ['sccdftb', '631G', '631++Gdp', '631+Gd', 'dftd631++Gdp', '6311++Gdp', 'dftd6311++Gdp']: 
            
            main(basis, i, )
