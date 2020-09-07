# there are 2 testing cases.
# 	1. 18 ds are all used as t-v set, no testing set.
#	2. 17 ds are used as t-v set, 1 ds as testing set, recursively
# the former one is used for dynamic   contribution
# ... later  ... .. .... ... intrinsic ............
# Zilin.
#

import subprocess as subp

for basis in ['sccdftb', '631G', '631++Gdp', '631+Gd', 'dftd631++Gdp', '6311++Gdp', 'dftd6311++Gdp']:

    for modelname in ['svr', 'gpr', 'krr', ]:
        subp.call('mkdir -p {0}.tuning/{1}'.format(basis, modelname, ), shell=True)

        subp.call('cp sh.job sh.job.temp', shell=True)
        subp.call("sed -i -e 's/jjjj/{0}/g' sh.job.temp".format(basis + '_' + modelname +'.a'), shell=True)
        subp.call("sed -i -e 's/1111/{0}/g' sh.job.temp".format('cv_'+modelname+'.py'), shell=True)
        subp.call("sed -i -e 's/2222/..\/..\/7.feat.sele\/0.man.sele\/0.ml_dat.build\/{0}.ml_dat\/ds_all.X_train.npy/g' sh.job.temp".format(basis), shell=True)
        subp.call("sed -i -e 's/3333/..\/..\/7.feat.sele\/0.man.sele\/0.ml_dat.build\/{0}.ml_dat\/ds_all.y_train.npy/g' sh.job.temp".format(basis), shell=True)
        subp.call("sed -i -e 's/4444/..\/..\/7.feat.sele\/0.man.sele\/0.ml_dat.build\/{0}.ml_dat\/ds_all.group_labels.npy/g' sh.job.temp".format(basis), shell=True)
        subp.call("sed -i -e 's/5555/{0}.tuning\/{1}\/tune.ds_all.log/g' sh.job.temp".format(basis, modelname), shell=True)
        subp.call("sbatch sh.job.temp", shell=True)
        
        for ds_id in range(18):
            subp.call('cp sh.job sh.job.temp', shell=True)
            subp.call("sed -i -e 's/jjjj/{0}/g' sh.job.temp".format(basis + '_' + modelname +'.' + str(ds_id)), shell=True)
            subp.call("sed -i -e 's/1111/{0}/g' sh.job.temp".format('cv_'+modelname+'.py'), shell=True)
            subp.call("sed -i -e 's/2222/..\/..\/7.feat.sele\/0.man.sele\/0.ml_dat.build\/{0}.ml_dat\/ds_{1}\/ds_{1}.X_train.npy/g' sh.job.temp".format(basis, ds_id, ), shell=True)
            subp.call("sed -i -e 's/3333/..\/..\/7.feat.sele\/0.man.sele\/0.ml_dat.build\/{0}.ml_dat\/ds_{1}\/ds_{1}.y_train.npy/g' sh.job.temp".format(basis, ds_id, ), shell=True)
            subp.call("sed -i -e 's/4444/..\/..\/7.feat.sele\/0.man.sele\/0.ml_dat.build\/{0}.ml_dat\/ds_{1}\/ds_{1}.group_labels.npy/g' sh.job.temp".format(basis, ds_id, ), shell=True)
            subp.call("sed -i -e 's/5555/{0}.tuning\/{1}\/tune.ds_{2}.log/g' sh.job.temp".format(basis, modelname, ds_id, ), shell=True)
            subp.call("sbatch sh.job.temp", shell=True)
