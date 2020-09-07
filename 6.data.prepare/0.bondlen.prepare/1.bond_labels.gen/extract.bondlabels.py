# get bonding pairs info from 'qm.psf', output to 'xxx.bond_labels.dat'
# note that only bonds in the QM region are identified.
# chemical bonds: 	interpret directly from psf
# H-bonds:		get by Baker_Hubbard criteria from Mdtraj
# extra reactive bonds:	manually specified, PNM-C7 == SER-Ogamma 
# Zilin Song, 11SEP2019
# 

import mdtraj as mdtraj

def get_atom_list(psf_dir):
    '''get list of atoms: ( (indice, resname_resid_atname) )
    '''
    atom_list = []
    is_atom_list = False

    lines =  open(psf_dir, 'r').readlines()

    for line in lines:
        words = line.split()

        if len(words) >= 2 and words[1] == '!NATOM':
            is_atom_list = True
            print(words)

        if len(words) >= 2 and words[1] == '!NBOND:':
            is_atom_list = False
            print(words)
            
        if is_atom_list == True and len(words) == 11:
            indice = words[0]
            resid = words[2]
            resname = words[3]
            atname = words[4]

            atom_list.append(
                (
                    indice, 
                    resname + '_' + resid + '_' + atname,
                )
            )
    
    return atom_list

def get_chemical_bonds(psf_dir):
    '''get list of chemically bonded atom pairs: ( (resname_resid_atname, resname_resid_atname) )
    '''
    atom_list = get_atom_list(psf_dir)

    bond_indices_list = []  # ( ( indice0, indice1 ) )
    
    lines =  open(psf_dir, 'r').readlines()
    is_bond_list = False
    for line in lines:
        words = line.split()

        if len(words) >= 2 and words[1] == '!NBOND:':
            is_bond_list = True

        if len(words) >= 3 and words[1] == '!NTHETA:':
            is_bond_list = False
        
        if is_bond_list == True and len(words) > 3:
            atom_count = len(words)
            
            bond_count = atom_count / 2

            for i in range(int(bond_count)):
                at1 = words[i * 2]
                at2 = words[i * 2 + 1]
                bond_indices_list.append(
                    (
                        at1, 
                        at2,
                    )
                )
    
    bond_list = []
    for at0, at1 in bond_indices_list:
        name0 = 'None'
        name1 = 'None'

        for at_index, at_name in atom_list:

            if at0 == at_index:
                name0 = at_name

            if at1 == at_index:
                name1 = at_name
        
        if (name0 != 'TIP3_290_H1' and name1 != 'TIP3_290_H2'):
            bond = name0 + '=' + name1
            bond_list.append(bond)
    
    return bond_list

def get_pdb_atom_list(pdb_dir):
    '''get atom list of pdb files
    '''
    lines = open(pdb_dir, 'r').readlines()

    atom_list = []
    for line in lines:
        words = line.split()

        if words[0] == 'ATOM':
            indice = words[1]
            atname = words[2]
            resname = words[3]
            resid = words[4]

            atom_list.append(
                (
                    indice,     # this is a string, needs to be int() for compare
                    resname + '_' + resid + '_' + atname,
                )
            )
    
    return atom_list

def get_h_bonds(pdb_dir):
    '''get bond list of H-Bonds
    '''

    bond_list = []
    atom_list = get_pdb_atom_list(pdb_dir)
    
    topo = mdtraj.load_pdb(pdb_dir)

    # hbon_list = [ [ donor_indice, H_indice, acceptor_indice ] ], indice = pdbindice-1 
    hbon_list = mdtraj.baker_hubbard(topo, freq=0, exclude_water=False, periodic=False, sidechain_only=False)

    for hbon in hbon_list:
        H_indice = hbon[1]
        acceptor_indice = hbon[2]

        at_h = 'None'
        at_acc = 'None'

        for at_indice, at_name in atom_list:

            if H_indice == (int(at_indice)-1):
                at_h = at_name

            if acceptor_indice == (int(at_indice)-1):
                at_acc = at_name
            
        bond = at_h + '=' + at_acc

        bond_list.append(bond)

    print('Processing: ' + pdb_dir + '\tDetected H bonds: ' + str(len(bond_list)))

    return bond_list

def build_pdb_dir(method, state, pathid, repid):
    '''return the string corresponding to pdb directory.
    '''
    return '../0.pdbs.gen/{0}/{1}/path.{2}.dat/r{3}.pdb'.format(method, state, str(pathid), str(repid))

def remove_reduntant(h_bond_lists):
    '''get non-reduntant h-bond list for all pdb conformations.
    '''
    h_bonds = []

    for h_bond_list in h_bond_lists:

        for h_bond in h_bond_list:
            
            is_reduntant = False

            at0 = h_bond.split('=')[0]
            at1 = h_bond.split('=')[1]

            for included_h_bond in h_bonds:
                included_at0 = included_h_bond.split('=')[0]
                included_at1 = included_h_bond.split('=')[1]

                if (at0 == included_at0 and at1 == included_at1) or (at0 == included_at1 and at1 == included_at0):
                    is_reduntant = True
            
            if is_reduntant == False:

                h_bonds.append(
                    at0 + '=' + at1
                )
    
    return h_bonds

def output(chemical_bonds, h_bonds, reactive_bonds, method):
    '''output file.
    '''
    fo = open('{0}.bond_labels.dat'.format(method), 'w')

    for chemical_bond in chemical_bonds:
        fo.write(chemical_bond + '\n')
    for h_bond in h_bonds:
        fo.write(h_bond + '\n')
    for r_bond in reactive_bonds:
        fo.write(r_bond + '\n')

def main():
    # list of pathids => 2d-list
    path_list = [
        ('reactant', [112,326,634,1570,1828,1966]), # reactant pathids
        ('ts',       [651,713,870,1267,1376,1826]), # ts pathids
        ('nhinter',  [390,663,728,1178,1221,1453]), # nhinter pathids
    ]
    # no. replicas
    rep_count = 50
    
    method = 'b3lyp' # sccdftb / b3lyp
    chemical_bonds = get_chemical_bonds('qm.psf')
    
    h_bond_lists = []
    for state, pathids in path_list:
        
        for pid in pathids:
            
            for rid in range(1, rep_count+1):
                pdb_dir = build_pdb_dir(method, state, str(pid), str(rid))
                h_bond_lists.append(get_h_bonds(pdb_dir))
    
    h_bonds = remove_reduntant(h_bond_lists)
    
    reactive_bonds = ['SER_70_OG=PNM_523_C7']

    output(chemical_bonds, h_bonds, reactive_bonds, method)

if __name__ == "__main__":
    main()
