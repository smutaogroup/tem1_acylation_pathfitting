# This script does prediction w/ all ML models on all testing sets
# models are trained with 17 of the pathways and attempts to predicts the rest.
# outputs are predictions results, rmse, and r2 
# NOTE that: 
#       get_ml_model(basis, model_name, ds_id) takes advantage of Python's dynamic typing prop.
#       You can not do it this way in static typing codes: c++, Java, C#, etc.
#       Magical python. => But I prefer C#.
# Zilin Song, 30OCT2019
#

import subprocess as subp
import numpy as numpy
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def is_number(s):
    '''test if string is a number
    '''
    try:
        float(s)
        return True

    except ValueError:
        return False

def get_hyperparameters(basis, ds_id, model_name, ):
    '''get tuned hyperparameter for model named model_name
    '''
    param_dir = '../1.params.extract/{0}.tuned/{1}.param'.format(basis, model_name)
    lines = open(param_dir, 'r').readlines()
    
    words = lines[ds_id].split('*')
    hyperparam_list = []

    for i in range(0, len(words)-1):            # last element in words should be '\n'
        
        if is_number(words[i]):                 # float
            hyperparam_list.append(float(words[i]))

        else:
            hyperparam_list.append(words[i])
    
    return(hyperparam_list)

def get_ml_model(basis, ds_id, model_name):
    '''get machine learning model with hyperparam sat.
    '''    
    model = None
    hyperparameters = get_hyperparameters(basis, ds_id, model_name,)

    if model_name == 'svr' and len(hyperparameters) == 4:       # SVR should have 4 params
        model = SVR(C=hyperparameters[0], 
                    epsilon=hyperparameters[1],
                    gamma=hyperparameters[2], 
                    kernel=hyperparameters[3], 
                    )

    elif model_name == 'gpr' and len(hyperparameters) == 1:     # GPR should have 1 params
        model = GaussianProcessRegressor(alpha=hyperparameters[0], )

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
    )
    return numpy.load(ds_dir)

def train_model(basis, ds_id, model_name, ):
    '''do fitting on dataset (ds_id) with model 'model_name',
    return the trained model
    '''
    # load data
    X_train = get_ds(basis, ds_id, 'X_train', )
    y_train = get_ds(basis, ds_id, 'y_train', )

    # get model & fit & predict
    model = get_ml_model(basis, ds_id, model_name, ).fit(X_train, y_train)
    
    return model

def do_fitting(basis, ds_id, model_name, ):
    '''do fitting and output results:
    y_pred;
    rmse;
    single_perturbation;
    shff_prediction;
    '''
    # output dir
    outdir = build_output_dir(basis, ds_id, model_name, )
    stat_out = open('{0}/stat'.format(outdir), 'w')
    stat_out.write('ds_{0}:\n'.format(ds_id, ))

    # no. replica
    repcount=48

    # ml model
    model = train_model(basis, ds_id, model_name, )

    # prediction
    y_test = get_ds(basis, ds_id, 'y_test', )
    X_test = get_ds(basis, ds_id, 'X_test', )
    y_pred = model.predict(X_test)       ## to be output
    numpy.save('{0}/y_pred.npy'.format(outdir), y_pred, )
    
    # metrics
    pred_rmse = round(mean_squared_error(y_test, y_pred, ) ** 0.5, 6)
    r2 = round(r2_score(y_test, y_pred, ), 6)
    stat_out.write('pred_rmse: {0}\n'.format(pred_rmse, ))
    stat_out.write('r2_score: {0}\n'.format(r2, ))

def build_output_dir(basis, ds_id, model_name, ):
    '''return the string for output directory.
    Also will build the directory in file system.
    '''
    outdir = './{0}.ml_test/ds_{1}/{2}'.format(basis, ds_id, model_name, )
    subp.call('mkdir -p {0}'.format(outdir), shell=True)
    return outdir

def get_feature_count(basis):
    '''get number of features in dataset.
    '''
    prefix = '../../7.feat.sele/0.man.sele/0.ml_dat.build'
    fdir = '{0}/{1}.ml_dat/features.man.dat'.format(prefix, basis, )
    return len(open(fdir, 'r').readlines())

def main():
    basis_list = ['sccdftb', '631G', '631++Gdp', '631+Gd', 'dftd631++Gdp', '6311++Gdp', 'dftd6311++Gdp'] 
    ds_ids = [i for i in range(0, 18)]
    model_names =['svr', 'gpr', 'krr', ] # 'knr' methods are generally not appliable => too many zeros in 

    for basis in basis_list:
                    
        for ds_id in ds_ids:
            
            for model_name in model_names:
                print('Processing... basis:{0}  model_name:{1}  dataset_id:{2}'.format(
                    basis, model_name, ds_id)
                )
                do_fitting(basis, ds_id, model_name)

if __name__ == "__main__":
    main()
