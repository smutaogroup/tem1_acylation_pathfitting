import subprocess as subp
import sys
import numpy as numpy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, 
                                              Matern, 
                                              RationalQuadratic,
                                              ExpSineSquared, 
                                              ConstantKernel, 
                                              )
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV

X_train_dir = sys.argv[1]  #"../../7.feature.filter/sccdftb.ml_dat/"
y_train_dir = sys.argv[2]
group_ids_dir = sys.argv[3]
log_dir = sys.argv[4]

# ============== load data set

X_train = numpy.load('{0}'.format(X_train_dir, ))
y_train = numpy.load('{0}'.format(y_train_dir, ))
group_ids = numpy.load('{0}'.format(group_ids_dir, ))

# logout dir must be mkdir before using this script. in submit.py
logout = open('{0}'.format(log_dir, ), 'w')    

# ============== leave one group out / LOGO cross validation

logo = LeaveOneGroupOut()
ds_splits = []
for train_ids, valid_ids in logo.split(X_train, y_train, groups=group_ids):
    ds_splits.append(
        (train_ids, valid_ids)
    )

# ============== def hyper parameters range

alpha_list = [(10**x) for x in range(-5, 0)]

# kernel: 1*rbf(1) is used
param_to_tune=[
    {   
        'alpha':alpha_list,
    },
]

# ============== grid search CV

gscv = GridSearchCV(
    GaussianProcessRegressor(), param_to_tune, cv=ds_splits, scoring='neg_mean_squared_error'
)

gscv.fit(X_train, y_train)

# ============== write results

logout.write("Best parameters set found on development set:\n\n")
for i in (gscv.best_params_):
    logout.write('{0}:{1}$'.format(str(i), str(gscv.best_params_[i])))
logout.write("\n\nGrid scores on development set:\n\n")

means = gscv.cv_results_['mean_test_score']
stds = gscv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
    logout.write("{0:.3f} (+/-{1:.3f}) for {2}\n".format(mean, std * 2, str(params))) 
