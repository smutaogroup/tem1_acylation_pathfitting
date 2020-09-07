# get all bond lengths from the pdb files 
# Zilin Song, 21SEP2019
#

import numpy as numpy

def get_bond_labels(bond_label_dir):
    '''get all bond labels from file.
    remove the swtich line at the end of each line.
    '''

    lines = open(bond_label_dir, 'r').readlines()

    bond_labels = []
    for line in lines:
        bond_labels.append(line.replace('\n', ''))
    
    return bond_labels

def calc_distance(at0, at1):
    '''calculate pairwise distance between atom0 & atom1.
    '''
    x_dist = at0[1][0] - at1[1][0]
    y_dist = at0[1][1] - at1[1][1]
    z_dist = at0[1][2] - at1[1][2]
    return round((x_dist**2 + y_dist**2 + z_dist**2)**(0.5), 8)

def get_atom_list(pdb_dir):
    '''get qm atoms in pdb file
    '''
    pdb_in = open(pdb_dir, 'r')
    lines = pdb_in.readlines()

    atom_list = []
    for line in lines:
        words = line.split()

        if len(words) >= 5 and words[0] == 'ATOM':
            atomname = words[2]
            resname = words[3]
            resid = words[4]

            coor_x = float(words[5])
            coor_y = float(words[6])
            coor_z = float(words[7])

            atom = ( 
                    resname + '_' + resid + '_' + atomname, 
                    (coor_x, coor_y, coor_z),
                )

            atom_list.append(atom)
    
    return atom_list

def get_bond_length(bond_label, atom_list):
    '''calculate bond length
    '''
    at0_label = bond_label.split('=')[0]
    at1_label = bond_label.split('=')[1]

    at0 = None
    at1 = None
    
    for at in atom_list:
        
        if at[0] == at0_label:
            at0 = at
        
        if at[0] == at1_label:
            at1 = at
    
    return calc_distance(at0, at1)

def build_pdb_dir(method, state, pathid, repid):
    '''return the string corresponding to pdb directory.
    '''
    return '../0.bondlen.prepare/0.pdbs.gen/{0}/{1}/path.{2}.dat/r{3}.pdb'.format(method, state, str(pathid), str(repid))

def get_bond_list(pdb_dir, bond_labels):
    '''get distances from pdb file
    '''
    print('Processing: {0}'.format(pdb_dir))

    atom_list = get_atom_list(pdb_dir)

    bond_list = []
    for bond_label in bond_labels:

        bond_length = get_bond_length(bond_label, atom_list)
        bond_list.append(
            (
                bond_label,
                bond_length,
            )
        )
    
    return bond_list

def normalize_bond_list(ref_bond_list, bond_list_to_normalize):
    '''normalize distance, all distance should be substracted by r1 distance.
    '''
    normalized_bond_list = []

    if len(bond_list_to_normalize) != len(ref_bond_list):
        print('bond list length mismatch.\n')
        exit()
    
    for i in range(len(bond_list_to_normalize)):

        if bond_list_to_normalize[i][0] != ref_bond_list[i][0]:
            print('dist label mismatch.\n')
            exit()

        #nor_dist = round((bond_list_to_normalize[i][1] - ref_bond_list[i][1]) / ref_bond_list[i][1] * 100, 4)
        nor_dist = round((bond_list_to_normalize[i][1] - ref_bond_list[i][1]), 4)
        
        normalized_bond_list.append( 
            (
                bond_list_to_normalize[i][0],
                nor_dist,
            )
        )

    return normalized_bond_list

def check_data(path_bond_mat):
    '''check if data is correctly produced in the following structure:

    [   ( 'state&pathid&repid', [   ( 'distancelabel', distance ),
                                        ( 'distancelabel', distance ),
                                        ( 'distancelabel', distance ),
                                        ...
                                    ],
            ),
            ( 'state&pathid&repid', [   ( 'distancelabel', distance ),
                                        ( 'distancelabel', distance ),
                                        ( 'distancelabel', distance ),
                                        ...
                                    ],
            ),
            ...
        ]
    '''
    ref_bond_label_list = [x[0] for x in path_bond_mat[0][1]]

    for rep_label, bond_list in path_bond_mat:

        if (len(bond_list)) != len(ref_bond_label_list):
            print('label number mismatch.\n')
            exit()

        for i in range(len(bond_list)):

            if ref_bond_label_list[i] != bond_list[i][0]:

                pirnt(rep_label + ':ref-' + ref_bond_label_list[i] + 'act-' + bond_list[i][0])
                exit()

def output(method, path_bond_mat):
    '''Write the data into several files.
    '''

    fo_rep_labels = open('replica_labels.dat', 'w')
    fo_numpy = '{0}.bond_mat'.format(method)

    ref_dist_labels = [x[0] for x in path_bond_mat[0][1]]

    bond_mat = []
    i = 0
    for replica_label, dist_list in path_bond_mat:
        fo_rep_labels.write(str(i) + '  ' + replica_label + '\n')
        i+=1
        tmp_distlist = []

        for dist in range(len(dist_list)):
            
            if dist_list[dist][0] == ref_dist_labels[dist]:
                tmp_distlist.append(dist_list[dist][1])
        
        bond_mat.append(tmp_distlist[:])

    numpy.save(fo_numpy, numpy.asarray(bond_mat))

def main(method):
    # path_list = [
    #     ('reactant', [112]), # nhinter pathids
    # ]
    # # list of pathids => 2d-list
    path_list = [
        ('reactant', [112,326,634,1570,1828,1966]), # reactant pathids
        ('ts',       [651,713,870,1267,1376,1826]), # ts pathids
        ('nhinter',  [390,663,728,1178,1221,1453]), # nhinter pathids
    ]


    # no. replicas
    rep_count = 50
    # bonded atom pairs
    bond_labels =get_bond_labels('../0.bondlen.prepare/1.bond_labels.gen/{0}.bond_labels.dat'.format(method))

    path_bond_mat = []

    for state, pathids in path_list:
        
        for pid in pathids:
            ref_bond_list = get_bond_list(build_pdb_dir(method, state, pid, 1), bond_labels)
                       
            for rid in range(1, rep_count+1):
                bond_list = get_bond_list(build_pdb_dir(method, state, pid, rid), bond_labels)
                normalized_bond_list = normalize_bond_list(ref_bond_list, bond_list)
                path_bond_mat.append(
                    (
                        state + '&' + str(pid) + '&' + str(rid),
                        normalized_bond_list,
                    )
                )
    
    check_data(path_bond_mat)
    output(method, path_bond_mat)

if __name__ == "__main__":
    main('b3lyp')
    main('sccdftb')
