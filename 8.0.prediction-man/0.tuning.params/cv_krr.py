import subprocess as subp
import sys
import numpy as numpy
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV

ds_id = sys.argv[1]
ds_indir = sys.argv[2]  #"../../7.feature.filter/sccdftb.ml_dat/"
log_dir = sys.argv[3]

# ============== load data set

X_train = numpy.load('{0}/ds_{1}/ds_{1}.X_train.npy'.format(ds_indir, ds_id))
y_train = numpy.load('{0}/ds_{1}/ds_{1}.y_train.npy'.format(ds_indir, ds_id))
group_ids = numpy.load('{0}/ds_{1}/ds_{1}.group_labels.npy'.format(ds_indir, ds_id))

subp.call('mkdir -p {0}'.format(log_dir), shell=True)
logout = open('{0}/ds_{1}.tune.log'.format(log_dir, ds_id), 'w')

# ============== leave one group out / LOGO cross validation

logo = LeaveOneGroupOut()
ds_splits = []
for train_ids, valid_ids in logo.split(X_train, y_train, groups=group_ids):
    ds_splits.append(
        (train_ids, valid_ids)
    )

# ============== def hyper parameters range

alpha_list = [0.01, 0.1 ]
gamma_list = [10 ** x for x in range(-3, 3) ]

param_to_tune=[
    {'kernel':['rbf'], 'alpha':alpha_list, 'gamma':gamma_list},
]

# ============== grid search CV

gscv = GridSearchCV(
    KernelRidge(), param_to_tune, cv=ds_splits, scoring='neg_mean_squared_error'
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
