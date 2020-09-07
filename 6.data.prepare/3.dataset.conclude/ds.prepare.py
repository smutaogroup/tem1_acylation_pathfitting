# this script generate the dataset for the full '105' features
# Zilin Song, 19SEP2019
#

import numpy as numpy
import subprocess as subp

def load_energy(energy_dir):
    '''load energies, return format:

    [ ( state, pathid, [ (energy, progress)
                         (energy, progress)
                         (energy, progress) 
                         ... <= 50 tuples <= 50 replicas
                       ],
      ),
      ( state, pathid, [ (energy, progress)
                         (energy, progress)
                         (energy, progress)
                         ...  <= 50 tuples <= 50 replicas
                       ],
      ),
      ...
    ]
    '''
    
    energy_list = []
    in_file = open(energy_dir, 'r')
    
    # init param
    pathway = []
    state = 'none'
    pathid = 'NaN'
    
    line = in_file.readline()
    while line:
        
        if line == '===begin===\n':
            
            if len(pathway) > 1 and state != 'none' and pathid != 'NaN':
                energy_list.append(
                    (
                        state,
                        pathid,
                        pathway[:],
                    )
                )
            pathway = []
            state = in_file.readline().split().pop()
            pathid = in_file.readline().split().pop()
                
        elif line == '===+end+===\n':
            energy_list.append(
                (
                    state,
                    pathid,
                    pathway[:],
                )
            )
        else:
            words = line.split()
            pathway.append(
                (
                    float(words[1]),
                    0, #float(words[2]),
                )
            )
        
        line = in_file.readline()
        
    return energy_list

def load_structure(struc_dir, replabel_dir):
    '''load list of features (interatomic distances), pure numbers.

    [ (state, pathid, repid, distlist[]) ]
    '''
    
    dist_array = numpy.load(struc_dir)
    
    dist_list = dist_array.tolist()
    in_labelfile = open(replabel_dir, 'r')
    
    reps = []
    for line in in_labelfile.readlines():
        words = line.split()[1].split('&')
        # states, pathids, reps
        reps.append( (words[0], words[1], words[2]) )
    
    structure_list = []
    
    if len(dist_list) != len(reps):
        print('data structure mismatch.')
        exit()
    else:
        for i in range(len(reps)):
            structure_list.append( 
                (
                    reps[i][0],
                    reps[i][1],
                    reps[i][2],
                    dist_list[i],
                ) 
            )
    
    return structure_list

def correlate_distlist(energy_list, structure_list):
    '''sort the structure list such that the order is the same with energy list

    data structure of energy_list:
    [ ( state, pathid, [ r1(energy, progress)
                         r2(energy, progress)
                         r3(energy, progress) 
                         ... <= 50 tuples <= 50 replicas
                       ],
      ),
      ( state, pathid, [ r1(energy, progress)
                         r2(energy, progress)
                         r3(energy, progress)
                         ...  <= 50 tuples <= 50 replicas
                       ],
      ),
      ...
    ]
    
    data structure of structure_list:
    [ (state, pathid, repid, distlist[]) ]
    
    data structure of sorted_distlist:
    [ (state, pathid, [ r1dist_[ ],
                        r2dist_[ ],
                        r3dist_[ ],
                      ],
      ),
      (state, pathid, [ r1dist_[ ],
                        r2dist_[ ],
                        r3dist_[ ],
                      ],
      ),    
      (state, pathid, [ r1dist_[ ],
                        r2dist_[ ],
                        r3dist_[ ],
                      ],
      ),
      ...
    ]
    '''
    
    sorted_distlist = []
    
    for energy in energy_list:
        state = energy[0]
        pathid = energy[1]
        temp_struc_list = []
        
        for i in range(1,51):
            
            for structure in structure_list:
                
                if structure[0] == state and structure[1] == pathid and structure[2] == str(i):
                    temp_struc_list.append(structure[3])
                    
        sorted_distlist.append(
            (
                state,
                pathid,
                temp_struc_list,
            )
        )
    
    return sorted_distlist

def get_path_ener(ener_list, state, pathid):
    '''get all energy of a certain paths

    data structure of ener_list:

    [ ( state, pathid, [ r1(energy, progress)
                         r2(energy, progress)
                         r3(energy, progress) 
                         ... <= 50 tuples <= 50 replicas
                       ],
      ),
      ( state, pathid, [ r1(energy, progress)
                         r2(energy, progress)
                         r3(energy, progress)
                         ...  <= 50 tuples <= 50 replicas
                       ],
      ),
      ...
    ]
    '''

    e_list = []
    
    for ener in ener_list:
        
        if ener[0] == state and ener[1] == str(pathid):
            e_list = ener[2]
    
    return [x[0] for x in e_list]

def get_path_progress(ener_list, state, pathid):
    '''get all energy of a certain paths

    data structure of ener_list:

    [ ( state, pathid, [ r1(energy, progress)
                         r2(energy, progress)
                         r3(energy, progress) 
                         ... <= 50 tuples <= 50 replicas
                       ],
      ),
      ( state, pathid, [ r1(energy, progress)
                         r2(energy, progress)
                         r3(energy, progress)
                         ...  <= 50 tuples <= 50 replicas
                       ],
      ),
      ...
    ]
    '''

    e_list = []
    
    for ener in ener_list:
        
        if ener[0] == state and ener[1] == str(pathid):
            e_list = ener[2]
    
    return [x[1] for x in e_list]

def get_path_dist(dist_list, state, pathid):
    '''get all geometry data of a certain path

    data structure of dist_list:

    [ (state, pathid, [ r1dist_[ ],
                        r2dist_[ ],
                        r3dist_[ ],
                      ],
      ),
      (state, pathid, [ r1dist_[ ],
                        r2dist_[ ],
                        r3dist_[ ],
                      ],
      ),    
      (state, pathid, [ r1dist_[ ],
                        r2dist_[ ],
                        r3dist_[ ],
                      ],
      ),
      ...
    ]
    
    '''
    d_list = []
    
    for dist in dist_list:
        
        if dist[0] == state and dist[1] == str(pathid):
            d_list = dist[2]
    
    return d_list

def dataset_conclude(dist_list, ener_list,):
    '''get overall dataset
    '''

    X = [] # feature
    y = [] # results
    
    if len(ener_list) != len(dist_list):
        print('data length inequivalent: dist_list vs. ener_list.\nExisting...')
        exit()
    
    for i in range(len(ener_list)):
        state = ener_list[i][0]
        pathid = ener_list[i][1]
        energies = ener_list[i][2]
        dists = get_path_dist(dist_list, state, pathid)
        
        if len(energies) == len(dists):
            
            for j in range(len(energies)):
                X.append(dists[j])
                y.append(energies[j][0])
    
    return X, y


    '''perform a feature elimination. return support_ and ranking_
    '''

    estimator = SVR(kernel='linear')
    selector = RFE(estimator, n_features_to_select=1, step=1,)
    selector = selector.fit(ds_X, ds_y)

    return selector.support_, selector.ranking_

def gen_leave_out_list(len_list):
    '''returns:
    [
        (0, [1,2,3,4,5,...len_list]),
        (1, [0,2,3,4,5,...len_list]),
        (2, [0,1,3,4,5,...len_list]),
        (3, [0,1,2,4,5,...len_list]),
        ...
    ]
    '''
    leave_one_out_list = []

    for i in range(0, len_list):
        retained_list = []

        for j in range(0, len_list):
            
            if i != j:
                retained_list.append(j)
        
        leave_one_out_list.append(
            (i, retained_list)
        )

    return leave_one_out_list

def main(basis):
    
    method = 'none'
    if basis != 'sccdftb':
        method = 'b3lyp'
    else:
        method = 'sccdftb'

    ener_dir = '../2.ener.get/path_ener.{0}.dat'.format(basis)
    struc_dir = '../1.bondlen.get/{0}.bond_mat.npy'.format(method)
    rep_label_dir = '../1.bondlen.get/replica_labels.dat'
    bon_label_dir = '../0.bondlen.prepare/1.bond_labels.gen/{0}.bond_labels.dat'.format(method)

    output_dir = './{0}.path_dat'.format(basis)

    subp.call('mkdir -p {0}'.format(output_dir), shell=True)
    subp.call('cp {0} {1}/{2}.bond_labels.dat'.format(bon_label_dir, output_dir, basis), shell=True)

    assign_log = open(output_dir + '/pathid.assigned.log', 'w')

    path_info = [
        ('reactant', [112,326,634,1570,1828,1966]), # reactant pathids
        ('ts',       [651,713,870,1267,1376,1826]), # ts pathids
        ('nhinter',  [390,663,728,1178,1221,1453]), # nhinter pathids
    ]

    ener = load_energy(ener_dir)
    struc = correlate_distlist(ener, load_structure(struc_dir, rep_label_dir))

    pid = 0
    path_list = []      # [(pid, X, y), (pid, X, y), ... ] len = 18, pid = int(0-17)
    
    for state, pathids in path_info:

        for pathid in pathids:
            path_ener = get_path_ener(ener, state, pathid)
            path_progress = get_path_progress(ener, state, pathid)
            path_struc = get_path_dist(struc, state, pathid)
            path_list.append(
                (pid, path_struc, path_ener, path_progress)
            )
            assign_log.write('dataset_{0}\t:  state: {1}; path_id: {2}\n'.format(str(pid), state, str(pathid)))
            pid += 1

    pathid_assign = gen_leave_out_list(len(path_list))

    ds_id = 0

    for leave_out_id, retained_ids in pathid_assign:
        
        X_test = path_list[leave_out_id][1]
        y_test = path_list[leave_out_id][2]
        test_progress = path_list[leave_out_id][3]

        X_train = []
        y_train = []
        train_progress = []
        group_ids = []
        
        current_group_id = 0
        for rid in retained_ids:

            for i in range(len(path_list[rid][1])):

                X_train.append(path_list[rid][1][i])
                y_train.append(path_list[rid][2][i])
                train_progress.append(path_list[rid][3][i])
                group_ids.append(current_group_id)
        
            current_group_id += 1
        
        numpy.save('{0}/ds_{1}.X_test'.format(output_dir, leave_out_id), X_test)
        numpy.save('{0}/ds_{1}.y_test'.format(output_dir, leave_out_id), y_test)
        numpy.save('{0}/ds_{1}.test_progress'.format(output_dir, leave_out_id), test_progress)

        numpy.save('{0}/ds_{1}.X_train'.format(output_dir, leave_out_id), X_train)
        numpy.save('{0}/ds_{1}.y_train'.format(output_dir, leave_out_id), y_train)
        numpy.save('{0}/ds_{1}.train_progress'.format(output_dir, leave_out_id), train_progress)
        
        numpy.save('{0}/ds_{1}.group_labels'.format(output_dir, leave_out_id), group_ids)

        ds_id += 1
    
    X=[]
    y=[]
    gids = []
    curr_gid = 0
    for did in range(18):
        for i in range(len(path_list[did][1])):
            X.append(path_list[did][1][i])
            y.append(path_list[did][2][i])
            gids.append(curr_gid)
        curr_gid += 1


    numpy.save('{0}/ds_all.X'.format(output_dir), X)
    numpy.save('{0}/ds_all.y'.format(output_dir), y)
    numpy.save('{0}/ds_all.group_labels'.format(output_dir), gids)


if __name__ == "__main__":
    for basis in ['sccdftb', '631G', '631++Gdp', '631+Gd', 'dftd631++Gdp', '6311++Gdp', 'dftd6311++Gdp']:
        main(basis)
