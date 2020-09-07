import numpy as numpy
import subprocess as subp
import sys
from sklearn.svm import SVR
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

gamma_list = [10 ** x for x in range(-3, 3) ]
C_list = [10 ** y for y in range(0, 3)]
epsilon_list = [0.001]

param_to_tune=[
    {'kernel':['rbf'], 'gamma':gamma_list, 'C':C_list, 'epsilon':epsilon_list, },
]

# ============== grid search CV

gscv = GridSearchCV(
    SVR(), param_to_tune, cv=ds_splits, scoring='neg_mean_squared_error'
)

gscv.fit(X_train, y_train)

# ============== write results

logout.write("Best parameters set found on development set:\n\n")
for i in (gscv.best_params_):
    logout.write('{0}:{1}$'.format(str(i), str(gscv.best_params_[i])))
logout.write("\n\nGrid scores on development set:\n\n")
