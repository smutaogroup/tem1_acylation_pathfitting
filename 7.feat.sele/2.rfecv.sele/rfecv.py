# recursive feature elimination
# Zilin Song, 19SEP2019
#

import sys
import subprocess as subp
import numpy as numpy
from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVR

def build_data_strings(ds_dir, ):
    '''return 
            X_dir, y_dir, group_ids_dir
    '''

    X_dir = '{0}/ds_all.X.npy'.format(ds_dir, )
    y_dir = '{0}/ds_all.y.npy'.format(ds_dir, )
    group_ids_dir = '{0}/ds_all.group_labels.npy'.format(ds_dir, )

    return X_dir, y_dir, group_ids_dir
    
def get_data_set(ds_dir, ):
    '''return numpy.arraies:
            X, y, group_ids
    '''
    X_dir, y_dir, group_ids_dir = build_data_strings(ds_dir, )

    X = numpy.load(X_dir)
    y = numpy.load(y_dir)
    group_ids = numpy.load(group_ids_dir)

    return X, y, group_ids

def get_cv_subsets(X, y, group_ids):
    '''return ds_splits - [
        (testset, validationset), => for cross validation
        (testset, validationset), => for cross validation
        (testset, validationset), => for cross validation
        ...
    ]

    LOGO algorithm.
    '''
    logo = LeaveOneGroupOut()

    ds_splits = []                  # (testset, validationset) => for cross validation
    for train_ids, valid_ids in logo.split(X, y, groups=group_ids):
        ds_splits.append(
            (train_ids, valid_ids)
        )

    return ds_splits

def get_feature_labels(ds_dir, basis):
    '''get the label of each feature from ds_dir/bond_labels.dat
    '''
    fi = open('{0}/{1}.bond_labels.dat'.format(ds_dir, basis), 'r')
    label_list = []

    for line in fi.readlines():
        label_list.append(line.replace('\n', ''))

    return label_list

def eliminate_feature(ds_X, ds_y, cv_spec, estimator):
    '''perform a logo cross validated recursive feature elimination 
    with linear kernel SVR.

    return support_ and ranking_
    '''
    
    selector = RFECV(estimator=estimator, 
                     step=1,
                     min_features_to_select=1,
                     cv=cv_spec, 
                     scoring='r2')
    selector = selector.fit(ds_X, ds_y)

    return selector.support_, selector.ranking_

def rank_feature(ranks, labels):
    '''sort the features with labels.
    '''
    if len(ranks) != len(labels):
        print('different list length: ranks, labels')
        exit()
    else:
        r_max = 0
        
        for i in ranks:
            if r_max <= i:
                r_max = i
        rank_feature_list = []
        for j in range(0, len(ranks)):
            rank_feature_list.append(
                (
                    ranks[j],
                    labels[j],
                )
            )
        
        sorted_list = []
        for k in range(1, r_max+1):
            for fe in rank_feature_list:
                if fe[0] == k:
                    sorted_list.append(fe)
        
        return sorted_list

def main(basis, ):

    estimator = SVR(kernel='linear', epsilon=0.001)

    ds_dir = '../../6.data.prepare/3.dataset.conclude/{0}.path_dat'.format(basis)

    X, y, group_ids = get_data_set(ds_dir)

    logo_cv = get_cv_subsets(X, y, group_ids)
    
    supp, rank = eliminate_feature(X, y, logo_cv, estimator)

    sorted_label = rank_feature(rank, get_feature_labels(ds_dir, basis))

    fo = open('{0}.rfecv.dat'.format(basis), 'w')

    for rank, name in sorted_label:
        fo.write(str(name) + '\t' + str(rank) + '\n')
    
if __name__ == "__main__":
    
    basis_list=['sccdftb', '631G', '631++Gdp', '631+Gd', 'dftd631++Gdp', '6311++Gdp', 'dftd6311++Gdp']
    main( basis_list[ int( sys.argv[1] ) ] )
