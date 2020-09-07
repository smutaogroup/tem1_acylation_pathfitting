import subprocess as subp

for basis in ['sccdftb', '631G', '631++Gdp', '631+Gd', 'dftd631++Gdp', '6311++Gdp', 'dftd6311++Gdp']: 

    for modelname in ['svr', 'gpr', 'krr', ]:
        subp.call('cp sh.job sh.job.temp', shell=True)
        subp.call("sed -i -e 's/####/{0}/g' sh.job.temp".format(basis[0:4] + '_' + modelname), shell=True)
        subp.call("sed -i -e 's/@@@@/{0}/g' sh.job.temp".format('cv_'+modelname+'.py'), shell=True)
        subp.call("sed -i -e 's/#@#@/..\/..\/7.feat.sele\/2.rfecv.sele\/{0}.ml_dat/g' sh.job.temp".format(basis), shell=True)
        subp.call("sed -i -e 's/@#@#/.\/{0}.tuning\/{1}/g' sh.job.temp".format(basis, modelname), shell=True)
        
        #../../7.feature.filter/sccdftb.ml_dat/
        subp.call("sbatch sh.job.temp", shell=True)
