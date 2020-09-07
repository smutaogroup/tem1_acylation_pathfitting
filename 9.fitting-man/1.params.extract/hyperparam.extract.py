# extract tuned results - what a crapy script.
# Zilin Song, 18OCT2019
#
import subprocess as subp

def output_log(basis, method, ds_id, ):
    subp.call('mkdir -p {0}.tuned/{1}'.format(basis, method, ), shell=True)

    indir = '../0.tuning.params/{0}.tuning/{1}/tune.ds_{2}.log'.format(basis, method, ds_id, )
    outdir = '{0}.tuned/{1}/tuned.ds_{2}.log'.format(basis, method, ds_id, )
    
    print('Processing: {0}'.format(indir, ))

    fi = open(indir, 'r', )
    fo = open(outdir, 'w', )

    lines = fi.readlines()
    if len(lines)>3:
        target_line = lines[2]      # third line of each log file is what we need.
        key_values = target_line.split("$")

        v_list = []
                
        for k_v in key_values:
                    
            if k_v != "\n":
                v_list.append(k_v.split(':')[1])

        for v in v_list:
            fo.write("{0}*".format(str(v)))

    fo.write("\n")

for basis in ['sccdftb', '631G', '631++Gdp', '631+Gd', 'dftd631++Gdp', '6311++Gdp', 'dftd6311++Gdp']:
    in_dir = "../0.tuning.params/{0}.tuning".format(basis)
    
    for method in ['gpr', 'krr', 'svr',]:
        output_log(basis, method, 'all', )

        for ds_id in range(18):
            output_log(basis, method, str(ds_id))
