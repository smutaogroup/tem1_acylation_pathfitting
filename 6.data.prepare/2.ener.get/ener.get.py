# get energy profile of replicas from each pathway 
# Zilin Song, 2019OCT1
#

def build_sccdftb_path_dir(state, pathid):
    '''build string of directory to sccdftb pathway profile.

    progress.dat
    energy.dat
    '''
    path_dat_dir = ''

    if state == 'reactant':
        path_dat_dir = '../../5.0.reactant/3.path.acylation/2.path.extract/path.' + str(pathid) + '.dat/'
    elif state == 'ts':
        path_dat_dir = '../../5.1.ts/3.path.acylation/2.path.extract/path.' + str(pathid) + '.dat/'
    elif state == 'nhinter':
        path_dat_dir = '../../5.2.nhinter/3.path.acylation/2.path.extract/path.' + str(pathid) + '.dat/'
    else:
        print('wrong state specified.\n')
        exit()
    
    return path_dat_dir + 'energy.dat', path_dat_dir + 'progress.dat'

def build_b3lyp_path_dir(method, state, pathid):
    '''build string of directory to DFT pathway profile.

    progress.dat
    energy.dat
    '''
    base_dat_dir = ''

    if state == 'reactant':
        base_dat_dir = '../../5.3.reactant.b3lyp'
    elif state == 'ts':
        base_dat_dir = '../../5.4.ts.b3lyp'
    elif state == 'nhinter':
        base_dat_dir = '../../5.5.nhinter.b3lyp'
    else:
        print('wrong state specified.\n')
        exit()
    
    prog_dir = '{0}/3.path.extract/path.{1}.dat/progress.dat'.format(base_dat_dir, pathid)
    ener_dir = ''

    if method == '631G':
        ener_dir = '{0}/4.631G.sp.calc/profile.path/path.{1}.ene'.format(base_dat_dir, pathid)
    elif method == '631+Gd':
        ener_dir = '{0}/6.631+Gd.sp.calc/profile.path/path.{1}.ene'.format(base_dat_dir, pathid)
    elif method == '631++Gdp':
        ener_dir = '{0}/5.631++Gdp.sp.calc/profile.path/path.{1}.ene'.format(base_dat_dir, pathid)
    elif method == 'dftd631++Gdp':
        ener_dir = '{0}/7.dftd_631++Gdp.sp.calc/profile.path/path.{1}.ene'.format(base_dat_dir, pathid)
    elif method == '6311++Gdp':
        ener_dir = '{0}/9.6311++Gdp.sp.calc/profile.path/path.{1}.ene'.format(base_dat_dir, pathid)
    elif method == 'dftd6311++Gdp':
            ener_dir = '{0}/8.dftd_6311++Gdp.sp.calc/profile.path/path.{1}.ene'.format(base_dat_dir, pathid)

    return ener_dir, prog_dir

def get_path_dat(method, state, pathid):
    '''return energy profile of pathway.
    '''
    if method == 'sccdftb':
        energy_dir, progress_dir = build_sccdftb_path_dir(state, pathid) 
    else:
        energy_dir, progress_dir = build_b3lyp_path_dir(method, state, pathid)

    ener_lines = open(energy_dir, 'r').readlines()

    path_profile = []
    ener_r1 = -9999.9999
    for i in range(2, len(ener_lines)):
        
        if i == 2:
            ener_r1 = float(ener_lines[i].split()[1])
        
        path_profile.append(
            (
                i-1, 
                round(float(ener_lines[i].split()[1]) - ener_r1, 4), 
            )
        )
    
    return path_profile

def main(method):
    path_list = [
        ('reactant', [112,326,634,1570,1828,1966]), # reactant pathids
        ('ts',       [651,713,870,1267,1376,1826]), # ts pathids
        ('nhinter',  [390,663,728,1178,1221,1453]), # nhinter pathids
    ]
    
    path_profiles = []

    for state, pathids in path_list:

        for pid in pathids:
            path_dat = get_path_dat(method, state, pid)
            path_profiles.append(
                    (
                        state,
                        pid,
                        path_dat,
                    )
                )

    fo = open('path_ener.{0}.dat'.format(method), 'w')

    for state, pid, path_profile in path_profiles:
        fo.write('===begin===\n## state: {0}\n## pathid: {1}\n'.format(state, str(pid)))

        for i, ener, in path_profile:
            fo.write('{0:4}\t{1:10}\n'.format(i, ener, ))
    
    fo.write('===+end+===\n')

if __name__ == "__main__":
    for m in ['sccdftb', '631G', '631++Gdp', '631+Gd', 'dftd631++Gdp', '6311++Gdp', 'dftd6311++Gdp']:
        main(m)
