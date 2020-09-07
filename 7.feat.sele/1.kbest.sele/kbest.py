# recursive feature elimination
# Zilin Song, 19SEP2019
#

import sys
import subprocess as subp
import numpy as numpy
from sklearn.feature_selection import SelectKBest, mutual_info_regression

def build_data_strings(ds_dir, ):
    '''return 
            X_train_dir, y_train_dir, X_test_dir, y_test_dir, group_ids_dir
    '''

    X_dir = '{0}/ds_all.X.npy'.format(ds_dir, )
    y_dir = '{0}/ds_all.y.npy'.format(ds_dir, )

    return X_dir, y_dir,

def get_data_set(ds_dir,):
    '''return numpy.arraies:
            X_train, y_train, X_test, y_test, group_ids
    '''
    X_dir, y_dir = build_data_strings(ds_dir)

    X = numpy.load(X_dir)
    y = numpy.load(y_dir)

    return X, y,

def get_feature_labels(ds_dir, basis):
    '''get the label of each feature from ds_dir/bond_labels.dat
    '''
    fi = open('{0}/{1}.bond_labels.dat'.format(ds_dir, basis), 'r')
    label_list = []

    for line in fi.readlines():
        label_list.append(line.replace('\n', ''))

    return label_list

def main(basis, ):

    ds_dir = '../../6.data.prepare/3.dataset.conclude/{0}.path_dat'.format(basis)

    ## should be loaded to 1 set.
    X, y = get_data_set(ds_dir,)
    
    X_selected = SelectKBest(score_func=mutual_info_regression, k=15).fit(X, y).get_support()

    bon_labels = get_feature_labels(ds_dir, basis, )
    fo = open('{0}.kbest.dat'.format(basis, ), 'w')

    for i in range(len(X_selected)):
        if X_selected[i] == True:
            fo.write(str(bon_labels[i]) + '\n')
    
if __name__ == "__main__":
    
    basis_list =['sccdftb', '631G', '631++Gdp', ]
    for basis in basis_list:
        main(basis, )
