# This script retrains all ML models from the tuned hyparameters, and derives intrinsic / dynamic energy contributions. 
# Note that some of the comments may not correspond to the publication.
# 	shff = shuffled ds, meaning that certain feature was omitted by setting them to zero. 
# 	Note that setting to zero is mandatory, remove from the feature vector will yield unidentical ML models.
# Zilin Song, 30OCT2019
#

import subprocess as subp
import numpy as numpy
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def is_number(s):
    '''test if string is a number
    '''
    try:
        float(s)
        return True

    except ValueError:
        return False

def get_hyperparameters(param_dir, ):
    '''get tuned hyperparameter for model named model_name
    '''
    lines = open(param_dir, 'r').readlines()
    
    words = lines[0].split('*')
    hyperparam_list = []

    for i in range(0, len(words)-1):            # last element in words should be '\n'
        
        if is_number(words[i]):                 # float
            hyperparam_list.append(float(words[i]))

        else:
            hyperparam_list.append(words[i])
    
    return(hyperparam_list)

def get_ml_model(basis, model_name, ds_id, ):
    '''get machine learning model with hyperparams.
       note that ds_id could be a number => for intrinsic measurements
       or the letter 'a' => for dynamic measurements
    '''    
    model = None
    param_dir = '../1.params.extract/{0}.tuned/{1}/tuned.ds_{2}.log'.format(basis, model_name, ds_id, )
    
    hyperparameters = get_hyperparameters(param_dir)

    if model_name == 'svr' and len(hyperparameters) == 4:       # SVR should have 4 params
        model = SVR(C=hyperparameters[0], 
                    epsilon=hyperparameters[1],
                    gamma=hyperparameters[2], 
                    kernel=hyperparameters[3], 
                    )

    elif model_name == 'gpr' and len(hyperparameters) == 1:     # GPR should have 2 params
        model = GaussianProcessRegressor(alpha=hyperparameters[0],)

    elif model_name == 'krr' and len(hyperparameters) == 3:     # KRR should have 3 params
        model = KernelRidge(alpha=hyperparameters[0],
                            gamma=hyperparameters[1], 
                            kernel=hyperparameters[2], )

    else: 
        print("Error: bad model name or number of hyperparameters.\n")
        print("Check if no. hyperparameters are consistent w/ specifiedmodel: {0}.\n".format(model_name))
        exit()

    return ( model if model != None else print('Error: Model not sat.\n'))

def get_ds(basis, ds_id, which_ds):
    '''get dataset as a numpy array
    ds_id:          dataset id, int
    which_ds(str):  specify name of dataset, name should be one of the following
        X/y_test
        X/y_train
        group_labels
    '''
    ds_prefix = '../../7.feat.sele/0.man.sele/0.ml_dat.build/'
    ds_dir = '{0}/{1}.ml_dat/ds_{2}/ds_{2}.{3}.npy'.format(
        ds_prefix, basis, str(ds_id), which_ds, 
    ) if ds_id != 'all' else '{0}/{1}.ml_dat/ds_{2}.{3}.npy'.format(
        ds_prefix, basis, str(ds_id), which_ds, 
    )
    return numpy.load(ds_dir).tolist()

def get_shff_ds(basis, ds_id, which_feature_group, ):
    '''get shuffled dataset as a numpy array
    ds_id:          dataset id, int
    '''
    ds_prefix = '../../7.feat.sele/0.man.sele/1.shff_dat.build'
    ds_dir = '{0}/{1}.shff_dat/ds_{2}/ds_{2}.shff_{3}.X_train.npy'.format(
        ds_prefix, basis, str(ds_id), which_feature_group, 
    ) 
    return numpy.load(ds_dir).tolist()

def train_model(basis, model_name, X_train, y_train, ds_id, ):
    '''do fitting on dataset (ds_id) with model 'model_name',
    return the trained model
    '''
    # get model & fit & predict
    model = get_ml_model(basis, model_name, ds_id, ).fit(X_train, y_train)
    
    return model

def calc_path_gradient(model, X_test, which_features, ):
    ''' Essentially, calculate the dynamic contributions. 
    Calculate the perturbated slope of single/multi var.
    return a list of 48 slopes for single/multi feature, the r/p state is removed.

    NOTE: 
    Perturbation slopes (PS) must be WEIGHTED to tell the ACTUAL importance.
    PS is nothing but a numerical importance:

        If we have a h-bond distance which is rarely changd thoughout the reaction
        and a proton transfer distance which is largely changed,
        To create a similiar energy profile, 

        The H-bond distance does not have to fluctuate much, while the proton-trans
        distance will have to suffer huge numerical change,

        Therefore in this case, the H-bond distance is going to be way much more
        important as a small numerical change of H-bond could largely impact the PES.

        So we have to weight the PS to tell the actual importance.
    '''
    e_ratio = 0.01

    importance_list = []
    for i in range(1, len(X_test)-1):
        X_pt = X_test[i]

        # build correlation list
        c_e_list = []  # a list of tuple: (feature_id, correlation_factor, pert_value)
        for f in which_features:
            pert_f = e_ratio * (X_test[i+1][f] - X_test[i-1][f])
            s = calc_single_perturbed_slope(model, X_pt, X_test[i+1], X_test[i-1], f)
            c_e_list.append(
                    (f, (1), pert_f) # (1 if s>0 else 0 if s==0 else -1), pert_f)
                )

        importance = calc_multiple_weighted_slope(
            model, X_pt, which_features, c_e_list
        )

        importance = round(importance, 4) 
        importance_list.append(importance)

    return importance_list
    
def calc_multiple_weighted_slope(model, X_pt, which_features, c_e_list):
    '''calculate the E(r) => scaled pertubation
    calculated the perturbed slope for 1 single feature at point X.
    which_feature: list. the indices of the interested features.
    corr_list: list of tuples, [(feature_id, corr_coef, pert), ...]
    '''
    pos_X_pt = []
    neg_X_pt = []
    m = len(which_features)
    for i in range(len(X_pt)):
        
        if i in which_features:

            # firstly get correlation coefficients
            corr_coef = 0
            pert_factor = 0

            for fid, cc, pf in c_e_list:
            
                if fid == i:
                    corr_coef = cc
                    pert_factor = pf

            if not (corr_coef in [-1, 0, 1]):
                print('invalid which_features list and corr_list: {0}\n'.format(corr_coef))
                exit()
    
            pos_X_pt.append(X_pt[i] + corr_coef * pert_factor)
            neg_X_pt.append(X_pt[i] - corr_coef * pert_factor)
        else:
            pos_X_pt.append(X_pt[i])
            neg_X_pt.append(X_pt[i])

    pos_y_pt = float(model.predict([pos_X_pt]))
    neg_y_pt = float(model.predict([neg_X_pt]))

    e = get_euclidean_distance(pos_X_pt, neg_X_pt, which_features, )
    weight = (e)
    slope = abs((pos_y_pt - neg_y_pt) / (e)) if e != 0 else 0
    return (slope * weight)
    
def get_euclidean_distance(pt_x1, pt_x2, which_features):
    '''calculate the distance b/w 2 pts in the variable space.
    '''
    x1_m = [pt_x1[i] for i in which_features]
    x2_m = [pt_x2[i] for i in which_features]
    squared_sum = 0
    for i in range(len(x1_m)):
        squared_sum += ((x1_m[i] - x2_m[i]) ** 2)
    return squared_sum**0.5

def calc_single_perturbed_slope(model, X_pt, X_pt_plus, X_pt_minus, which_feature):
    '''get the local correlation matrix.
    calculated the perturbed slope for 1 single feature at point X.
    which_feature: int. the indice of the interested feature.
    used only to get the correlation list.
    '''
    pos_X_pt = []
    neg_X_pt = []
    x_diff = None
    for i in range(len(X_pt)):
        if i == which_feature:
            pos_X_pt.append(X_pt_plus[i])
            neg_X_pt.append(X_pt_minus[i])
            x_diff = X_pt_plus[i] - X_pt_minus[i]
        else:
            pos_X_pt.append(X_pt[i])
            neg_X_pt.append(X_pt[i])
    pos_y_pt = float(model.predict([pos_X_pt]))
    neg_y_pt = float(model.predict([neg_X_pt]))

    slope = (pos_y_pt - neg_y_pt) / (x_diff) if x_diff != 0 else 0
    return slope

def calc_shff_rmse(basis, model_name, ds_id, which_feature_group, ):
    '''calculates the intrinsic contributions.
    calculate the RMSE difference between models -
    trained with full set or shuffled set.
    '''
    X_train = get_ds(basis, ds_id, 'X_train', )
    X_shff = get_shff_ds(basis, ds_id, which_feature_group, )
    y_train = get_ds(basis, ds_id, 'y_train', )

    model_full = train_model(basis, model_name, X_train, y_train, ds_id, )
    model_shff = train_model(basis, model_name, X_shff, y_train, ds_id, )

    X_test = get_ds(basis, ds_id, 'X_test', )
    y_test = get_ds(basis, ds_id, 'y_test', )

    y_full_pred = model_full.predict(X_test).tolist()
    y_shff_pred = model_shff.predict(X_test).tolist()
    
    rmse = mean_squared_error(y_full_pred, y_shff_pred, ) ** 0.5
    rmse_to_calculation = mean_squared_error(y_test, y_shff_pred, ) ** 0.5
    
    return round(rmse, 4), round(rmse_to_calculation, 4)

def do_fitting(basis, model_name, pathcount=18):
    '''do fitting for all ds.
    '''
    # outdir
    outdir = './{0}.ml_test/{1}'.format(basis, model_name, )
    subp.call('mkdir -p {0}'.format(outdir), shell=True)

    # full set data
    X_train = get_ds(basis, 'all', 'X_train', )
    y_train = get_ds(basis, 'all', 'y_train', )
    model = train_model(basis, model_name, X_train, y_train, 'all', )
    
    # gradient slope calc => dynamica importance
    features_list = [
        [0, 2, ],       # ser70 proton to water;
        [9, 10, ],      # water proton to glu166;
        [3, 4, 5, ],    # lys73 proton to ser70/130;
        [6, 7, 8, ],    # ser130 proton to pnm;
        [1, ],          # ser70 pnm bond formation;
        [14, ],         # pnm C-N bond cleavage;
        [11, ],         # wat290-asn170 H bonding;
        [12, ],         # asn170-glu166 H bonding;
        [13, ],         # ser235-pnmO12 H bonding;
    ]
    perturbation_list = []
    
    for ds_id in range(pathcount):
        print('\tdyna: ds_id: {0}'.format(ds_id))
        X_pert = X_train[ds_id*50:(ds_id+1)*50]

        for fl in features_list:
            pert = calc_path_gradient(model, X_pert, fl, )
            perturbation_list.append(
                pert, 
            )

        numpy.save(
            '{0}/ds_{1}.pert.npy'.format(outdir, ds_id), perturbation_list, 
            )
    return
    for ds_id in range(pathcount):
        ds_stat=open('{0}/ds_{1}.stat'.format(outdir, ds_id), 'w')
        print('\tstat: ds_id: {0}'.format(ds_id))
        rmse_list = []

        for fl in range(len(features_list)):
            rmse = calc_shff_rmse(basis, model_name, ds_id, fl, )
            rmse_list.append((rmse))
        
        for r in range(len(rmse_list)):
            ds_stat.write('shff_rmse_{0}: {1} %contri_shff_rmse_{0}: {2} shff_rmse_{0}: {3} %contri_shff_rmse_{0}: {4}\n'.format(
                    r, rmse_list[r][0], round(rmse_list[r][0]/sum([i[0] for i in rmse_list])*100, 2), 
                       rmse_list[r][1], round(rmse_list[r][1]/sum([i[1] for i in rmse_list])*100, 2),
                )
            )
            print('\t\tshff_rmse_{0}: {1} %contri_shff_rmse_{0}: {2} shff_rmse_{0}: {3} %contri_shff_rmse_{0}: {4}'.format(
                    r, rmse_list[r][0], round(rmse_list[r][0]/sum([i[0] for i in rmse_list])*100, 2),
                       rmse_list[r][1], round(rmse_list[r][1]/sum([i[1] for i in rmse_list])*100, 2),
						                       )
            )
    
def build_output_dir(basis, ds_id, model_name, ):
    '''return the string for output directory.
    Also will build the directory in file system.
    '''
    outdir = './{0}.ml_test/ds_{1}/{2}'.format(basis, ds_id, model_name, )
    subp.call('mkdir -p {0}'.format(outdir), shell=True)
    return outdir

def main():
    basis_list =  ['sccdftb', '631G', '631++Gdp', ] # '631+Gd', 'dftd631++Gdp', '6311++Gdp', 'dftd6311++Gdp'] 
    model_names = ['gpr', 'svr', 'krr'] 

    for basis in ['631++Gdp']:
        
        for model in model_names: 
            print('Processing... basis:{0}  model_name:{1}'.format(
                basis, model,)
            )
            do_fitting(basis, model)

if __name__ == '__main__':
    main()
