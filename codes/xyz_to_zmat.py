import numpy as np

def get_zmat_from_coordinates(xyzarr):
    xyzarr = np.array(xyzarr)
    distmat = distance_matrix(xyzarr)
    zmat = []
    npart, ncoord = xyzarr.shape
    rlist = []
    alist = []
    dlist = []
    rconnect = []
    aconnect = []
    dconnect = []

    if npart > 0:
        
        if npart > 1:
            # and the second, with distance from first
            rlist.append(distmat[0][1])
            zmat.append([1,rlist[0]])
            rconnect.append(1)
            
            if npart > 2:
                rconnect.append(1)    
                rlist.append(distmat[0][2])
                alist.append(angle(xyzarr, 2, 0, 1))
                aconnect.append(2)
                zmat.append([1, rlist[1], 2, alist[0]])

                
                if npart > 3:
                    for i in range(3, npart):
                        rconnect.append(i-2)
                        aconnect.append(i-1)
                        dconnect.append(i)
                        rlist.append(distmat[i-3][i])
                        alist.append(angle(xyzarr, i, i-3, i-2))
                        dlist.append(dihedral(xyzarr, i, i-3, i-2, i-1))
                        zmat.append([i-2, rlist[i-1], i-1, alist[i-2], i, dlist[i-3]])
    zparams = rlist+alist+dlist
    zconnect  = [rconnect,aconnect,dconnect]
    return (zparams, zconnect)

def get_coordinates_from_zmat(zparams, zconnect):
    
    rconnect = zconnect[0]
    aconnect = zconnect[1]
    dconnect = zconnect[2]
    
    rlist=[]
    alist=[]
    dlist=[]
    zparams = zparams.tolist()
    for i in range(len(rconnect)):
        rlist.append(zparams.pop(0))
    for i in range(len(aconnect)):
        alist.append(zparams.pop(0))
    for i in range(len(dconnect)):
        dlist.append(zparams.pop(0))
    
    npart = len(rconnect) + 1

    xyzarr = np.zeros([npart, 3])
    if (npart > 1):
        xyzarr[1] = [rlist[0], 0.0, 0.0]

    if (npart > 2):
        i = rconnect[1] - 1
        j = aconnect[0] - 1
        r = rlist[1]
        theta = alist[0] 
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a_i = xyzarr[i]
        b_ij = xyzarr[j] - xyzarr[i]
        if (b_ij[0] < 0):
            x = a_i[0] - x
            y = a_i[1] - y
        else:
            x = a_i[0] + x
            y = a_i[1] + y
        xyzarr[2] = [x, y, 0.0]

    for n in range(3, npart):
        r = rlist[n-1]
        theta = alist[n-2]
        phi = dlist[n-3] - np.pi
        
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)

        x = r * cosTheta
        y = r * cosPhi * sinTheta
        z = r * sinPhi * sinTheta
        
        i = rconnect[n-1] - 1
        j = aconnect[n-2] - 1
        k = dconnect[n-3] - 1
        a = xyzarr[k]
        b = xyzarr[j]
        c = xyzarr[i]
        
        ab = b - a
        bc = c - b
        bc = bc / np.linalg.norm(bc)
        nv = np.cross(ab, bc)
        nv = nv / np.linalg.norm(nv)
        ncbc = np.cross(nv, bc)
        
        new_x = c[0] - bc[0] * x + ncbc[0] * y + nv[0] * z
        new_y = c[1] - bc[1] * x + ncbc[1] * y + nv[1] * z
        new_z = c[2] - bc[2] * x + ncbc[2] * y + nv[2] * z
        xyzarr[n] = [new_x, new_y, new_z]
    
    return (xyzarr)  


def angle(xyzarr, i, j, k):
    rij = xyzarr[i] - xyzarr[j]
    rkj = xyzarr[k] - xyzarr[j]
    cos_theta = np.dot(rij, rkj)
    sin_theta = np.linalg.norm(np.cross(rij, rkj))
    theta = np.arctan2(sin_theta, cos_theta)
    return theta

def dihedral(xyzarr, i, j, k, l):
    rji = xyzarr[j] - xyzarr[i]
    rkj = xyzarr[k] - xyzarr[j]
    rlk = xyzarr[l] - xyzarr[k]
    v1 = np.cross(rji, rkj)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(rlk, rkj)
    v2 = v2 / np.linalg.norm(v2)
    m1 = np.cross(v1, rkj) / np.linalg.norm(rkj)
    x = np.dot(v1, v2)
    y = np.dot(m1, v2)
    chi = np.arctan2(y, x)
    return chi



def distance_matrix(xyzarr):
    npart, ncoord = xyzarr.shape
    dist_mat = np.zeros([npart, npart])
    for i in range(npart):
        for j in range(0, i):
            rvec = xyzarr[i] - xyzarr[j]
            dist_mat[i][j] = dist_mat[j][i] = np.sqrt(np.dot(rvec, rvec))
    return dist_mat
