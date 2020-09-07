# extract tuned hyperparameters - what a crapy script.
# Zilin Song, 18OCT2019
#

import subprocess as subp

medthod_list = ['gpr', 'krr', 'svr',] # 'snn']
basis_list = ['sccdftb', '631G', '631++Gdp', '631+Gd', 'dftd631++Gdp', '6311++Gdp', 'dftd6311++Gdp'] 


for basis in basis_list:
    in_dir = "../0.tuning.params/{0}.tuning".format(basis)
    
    for method in medthod_list:
        subp.call('mkdir -p {0}.tuned'.format(basis), shell=True)
        file_out = open('{0}.tuned/{1}.param'.format(basis, method),'w')

        for i in range(18):
            print('Processing: {0}/{1}/ds_{2}.tune.log'.format(in_dir, method, str(i)))
            file_in = open('{0}/{1}/ds_{2}.tune.log'.format(in_dir, method, str(i)), 'r')
            lines = file_in.readlines()

            if len(lines)>3:
                target_line = lines[2]      # third line of each log file is what we need.
                key_values = target_line.split("$")

                v_list = []
                
                for k_v in key_values:
                    
                    if k_v != "\n":
                        v_list.append(k_v.split(':')[1])

                for v in v_list:
                    file_out.write("{0}*".format(str(v)))

            file_out.write("\n")
