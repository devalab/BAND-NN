import numpy as np

def get_features(conformer,S,bond_connectivity_list):
    conformer=np.array(conformer)
    nonbondcutoff = 6
    
    bonds = generate_bondconnectivty_matrix(bond_connectivity_list)
    
    #Calculate the atomic environment vector for each atom
    atomic_envs = generate_atomic_env(bonds, S)

    #Calculate the sets of bonds and bond values
    bondlist, bonddistances = generate_bond_data(conformer, bonds)
    bondfeatures = generate_bond_features(bonddistances,bondlist,atomic_envs)

    #Calculate the 3 atom angle sets and angle values
    angles_list, angles = generate_angle_data(conformer, bonds)
    anglefeatures = generate_angle_features(angles,angles_list,atomic_envs,bondlist,bonddistances)

    #Calculate  4 atom dihedral sets and dihedral values 
    dihedral_list, dihedral_angles = generate_dihedral_data(conformer,bonds)
    dihedralfeatures = generate_dihedralangle_features(dihedral_angles, dihedral_list, atomic_envs, bondlist, bonddistances, angles_list, angles)

    # Calculate the list of Non-bonds
    nonbond_list, nonbonddistances = generate_nonbond_data(conformer, bonds, nonbondcutoff)
    nonbondfeatures = generate_bond_features(nonbonddistances,nonbond_list,atomic_envs)

    # Zipping the data
    features = {}
    features['bonds'] = np.array(bondfeatures)
    features['angles'] = np.array(anglefeatures)
    features['nonbonds'] = np.array(nonbondfeatures)
    features['dihedrals'] = np.array(dihedralfeatures)
    return features     


def generate_bondconnectivty_matrix(bond_connectivity_list):
    bond_matrix = [[0 for i in range(len(bond_connectivity_list))] for j in range(len(bond_connectivity_list))]
    for i1 in range(len(bond_connectivity_list)):
        for i2 in bond_connectivity_list[i1]:
            bond_matrix[i1][i2] = 1
            bond_matrix[i2][i1] = 1
    return bond_matrix


def generate_atomic_env(bonds, S):
    atomic_envs = []
    for i in range(len(bonds)):
        atom_id = {'H':0, 'C':1, 'O':2, 'N':3 }
        atomtype = [0,0,0,0]
        atomtype[atom_id[S[i]]]  = 1 
        immediate_neighbour_count = [0,0,0,0]
        for j in range(len(bonds[i])):
            if(bonds[i][j] > 0):
                immediate_neighbour_count[atom_id[S[j]]] += 1
        atomic_envs.append(atomtype + immediate_neighbour_count)
    return atomic_envs


def generate_bond_data(conformer,bonds):
    #Calculate the paiwise-distances among the atoms
    distance = [[0 for i in range(len(conformer))] for j in range(len(conformer))]
    for i in range(len(conformer)):
        for j in range(len(conformer)):
            distance[i][j] = np.linalg.norm(conformer[i]-conformer[j])

    bondlist = []
    bonddistances = []
    for i in range(len(bonds)):
        for j in range(i):
            if(bonds[i][j] is 1):
                bondlist.append([i,j])
                bonddistances.append(distance[i][j])

    return bondlist, bonddistances

def generate_bond_features(bonddistances, bondlist, atomtype):
    labels = []
    for bond in range(len(bondlist)):
        bond_feature = []
        if(atomtype[bondlist[bond][0]] > atomtype[bondlist[bond][1]]):
            bond_feature += atomtype[bondlist[bond][0]] + atomtype[bondlist[bond][1]]
        else:
            bond_feature += atomtype[bondlist[bond][1]] + atomtype[bondlist[bond][0]]
        bond_feature.append(bonddistances[bond])
        labels.append(bond_feature)
    return labels


def generate_angle_data(conformer,bonds):
    angles_list = []
    for i in range(len(conformer)):
        for j in range(len(conformer)):
            for k in range(len(conformer)):
                if(j!=i and j!=k and i>k and bonds[i][j]!=0 and bonds[j][k]!=0):
                    angles_list.append([i,j,k])

    angles = []
    for angle_triplet in angles_list:
        angle = get_angle(conformer[angle_triplet[0]], conformer[angle_triplet[1]], conformer[angle_triplet[2]])
        angles.append(angle)
    return angles_list, angles


def get_angle(coor1,coor2,coor3):
    ba =coor1 - coor2   
    bc = coor3 - coor2 
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    if cosine_angle > 1.0:
        cosine_angle=1.0
    elif cosine_angle < -1.0:
        cosine_angle=-1.0
    if cosine_angle <=1.0 and cosine_angle>=-1.0:
        angle = np.arccos(cosine_angle)
        if angle > np.pi:
            angle=2*(np.pi)-angle
    return angle

def generate_angle_features(angles, angletype, atomtype,bondlist,bonddistances):
    labels = []
    for angle in range(len(angletype)):
        anglefeature = []
        if(atomtype[angletype[angle][0]] > atomtype[angletype[angle][2]]):
            anglefeature += atomtype[angletype[angle][0]] + atomtype[angletype[angle][2]]
            bondlen1 = get_bondlen(angletype[angle][0],angletype[angle][1],bondlist,bonddistances)
            bondlen2 = get_bondlen(angletype[angle][1],angletype[angle][2],bondlist,bonddistances)
        else:
            anglefeature += atomtype[angletype[angle][2]] + atomtype[angletype[angle][0]]
            bondlen1 = get_bondlen(angletype[angle][1],angletype[angle][2],bondlist,bonddistances)
            bondlen2 = get_bondlen(angletype[angle][0],angletype[angle][1],bondlist,bonddistances)

        anglefeature += atomtype[angletype[angle][1]]
        anglefeature += ([angles[angle],bondlen1,bondlen2])
        labels.append(anglefeature)
    return labels

def get_bondlen(i1,i2,bondtypelist,bondlenlist):
    try:
        index = bondtypelist.index([i1,i2])
    except:
        index = bondtypelist.index([i2,i1])
    return bondlenlist[index]

def generate_nonbond_data(conformer,bonds,nonbondcutoff):
    #Calculate the paiwise-distances among the atoms
    distance = [[0 for i in range(len(conformer))] for j in range(len(conformer))]
    for i in range(len(conformer)):
        for j in range(len(conformer)):
            distance[i][j] = np.linalg.norm(conformer[i]-conformer[j])
    nonbond_distances = []
    nonbond_list = []
    for i in range(len(conformer)):
        for j in range(len(conformer)):
            if(i > j and distance[i][j] <  nonbondcutoff and (bonds[i][j] == 0 ) ):
                nonbond_list.append([i,j])
                nonbond_distances.append(distance[i][j])
    return nonbond_list, nonbond_distances

def generate_dihedral_data(conformer,bonds):
    dihedral_list= []
    for i in range(len(conformer)):
        for j in range(len(conformer)):
            for k in range(len(conformer)):
                for l in range(len(conformer)):
                    if( i>l and i!=j and i!=k and j!=k and j!=l and k!=l and bonds[i][j] == 1 and bonds[j][k]==1 and bonds[k][l]==1):
                        dihedral_list.append([i,j,k,l])
                        
    dihedrals = []
    for dihed in dihedral_list:
        dihedral_angle = get_dihedral(conformer[dihed[0]],conformer[dihed[1]],conformer[dihed[2]],conformer[dihed[3]])
        dihedrals.append(dihedral_angle) 
    return dihedral_list,dihedrals

def get_dihedral(p0, p1, p2, p3):
    b0=p0-p1
    b1=p2-p1
    b2=p3-p2

    b0xb1 = np.cross(b0,b1)
    b1xb2 = np.cross(b2,b1)
    
    b0xb1_x_b1xb2 = np.cross(b0xb1,b1xb2)
    y = np.dot(b0xb1_x_b1xb2, b1)*(1.0/np.linalg.norm(b1))
    x = np.dot(b0xb1, b1xb2)
    return np.arctan2(y, x)

def get_angleval(i1,i2,i3,angletypelist,anglevallist):
    try:
        index = angletypelist.index([i1,i2,i3])
    except:
        index = angletypelist.index([i3,i2,i1])

    return anglevallist[index]

def generate_dihedralangle_features(dihedral_angles, dihedral_list, atomtype,bondtypelist,bondlenlist,angletypelist,anglevallist):
    labels = []
    for dihedral in range(len(dihedral_angles)):
        dihedral_feature = []
        if(atomtype[dihedral_list[dihedral][0]] > atomtype[dihedral_list[dihedral][3]]):
            index1 = 0
            index2 = 1
            index3 = 2
            index4 = 3
        else:
            index1 = 3
            index2 = 2
            index3 = 1
            index4 = 0

        dihedral_feature += atomtype[dihedral_list[dihedral][index1]] + atomtype[dihedral_list[dihedral][index2]]
        dihedral_feature += atomtype[dihedral_list[dihedral][index3]] + atomtype[dihedral_list[dihedral][index4]]
        bondlen1 = get_bondlen(dihedral_list[dihedral][index1],dihedral_list[dihedral][index2],bondtypelist,bondlenlist)
        bondlen2 = get_bondlen(dihedral_list[dihedral][index2],dihedral_list[dihedral][index3],bondtypelist,bondlenlist)
        bondlen3 = get_bondlen(dihedral_list[dihedral][index3],dihedral_list[dihedral][index4],bondtypelist,bondlenlist)
        angleval1 = get_angleval(dihedral_list[dihedral][index1],dihedral_list[dihedral][index2],dihedral_list[dihedral][index3],angletypelist,anglevallist)
        angleval2 = get_angleval(dihedral_list[dihedral][index2],dihedral_list[dihedral][index3],dihedral_list[dihedral][index4],angletypelist,anglevallist)

        dihedral_feature += (dihedral_angles[dihedral],angleval1,angleval2,bondlen1,bondlen2,bondlen3)
        labels.append(dihedral_feature)
    return labels
