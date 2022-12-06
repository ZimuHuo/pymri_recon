import numpy as np
import sys
sys.path.insert(1, '../')
import util.io as IO
from tqdm.notebook import tqdm
from scipy.spatial import Voronoi, ConvexHull, convex_hull_plot_2d
'''
Formula from Jack et al.(1991)
According to Jackson 1991, on selection of the colvolution kernel page 3
u is only defined for |u|<w/2  
The convolution kernel is symmetrical, so only half part is computed, and it is also 
presampled with oversampling ratio of 2 for faster computation, check Betty's paper for lower oversampling ratio. 
'''

# Generic kb kernel 
def kb(u, width, beta):
    u = beta*np.sqrt(1-(2*u/width)**2)
    u = np.i0(u)/width
    return u 



def KaiserBesselwindow(width, length,overgridfactor):
    w = width
    l = length
    alpha = overgridfactor
    beta = np.pi*np.sqrt(w**2/alpha**2*(alpha-1/2)**2-0.8)
    # from betty, 2005, on rapid griding algorithms 
    
    u = np.arange(0,l,1)/(l-1)*w/2
    #According to Jackson 1991, on selection of the colvolution kernel page 3
    #u is only defined for |u|<w/2

    window = kb(u, w, beta)
    window = window/window[0]

    return window




'''
standard griding
'''
def gridding(mat, data, traj, dcf):
    gridsize = mat.shape[0]
    Kernallength = 32
    kernalwidth = 5 
    window = KaiserBesselwindow(kernalwidth, Kernallength, 1.375)  
    kwidth = kernalwidth / 2 / gridsize
    gridcenter = gridsize / 2
    for n, weight in enumerate(dcf):
        kx = traj[n,0]
        ky = traj[n,1]
        xmin = int((kx - kwidth) * gridsize + gridcenter)
        xmax = int((kx + kwidth) * gridsize + gridcenter) + 1
        ymin = int((ky - kwidth) * gridsize + gridcenter)
        ymax = int((ky + kwidth) * gridsize + gridcenter) + 1
        if (xmin < 0):
            xmin = 0
        if (xmax >= gridsize):
            xmax = gridsize
        if (ymin < 0):
            ymin = 0
        if (ymax >= gridsize):
            ymax = gridsize
        for x in range(xmin, xmax):
            dx = (x - gridcenter) / gridsize - kx
            for y in range(ymin, ymax):
                dy = (y - gridcenter) / gridsize - ky
                d = np.sqrt(dx ** 2 + dy ** 2)
                if (d < kwidth):
                    idx = d / kwidth * (Kernallength - 1)
                    idxint = int(idx)
                    frac = idx - idxint
                    kernal = window[idxint] * (1 - frac) + window[idxint + 1] * frac
                    mat[x, y] += kernal * weight * data[n]
    return mat



'''
Equation 19
(W * phi) * R from  Jim Pipe et al. (1999)
It simply means convolve weight with kernel on to R, which is a cartesian grid
Complexity O(2pi L^2 N) 
'''
    
def grid(traj, dcf, gridsize = 256):
    mat = np.zeros([gridsize, gridsize], dtype=complex)
    gridsize = mat.shape[0]
    Kernallength = 32
    kernalwidth = 5 
    window = KaiserBesselwindow(kernalwidth, Kernallength, 1.375)  
    kwidth = kernalwidth / 2 / gridsize
    gridcenter = gridsize / 2
    for n, weight in enumerate(dcf):
        kx = traj[n,0]
        ky = traj[n,1]
        xmin = int((kx - kwidth) * gridsize + gridcenter)
        xmax = int((kx + kwidth) * gridsize + gridcenter) + 1
        ymin = int((ky - kwidth) * gridsize + gridcenter)
        ymax = int((ky + kwidth) * gridsize + gridcenter) + 1
        if (xmin < 0):
            xmin = 0
        if (xmax >= gridsize):
            xmax = gridsize
        if (ymin < 0):
            ymin = 0
        if (ymax >= gridsize):
            ymax = gridsize
        for x in range(xmin, xmax):
            dx = (x - gridcenter) / gridsize - kx
            for y in range(ymin, ymax):
                dy = (y - gridcenter) / gridsize - ky
                d = np.sqrt(dx ** 2 + dy ** 2)
                if (d < kwidth):
                    idx = d / kwidth * (Kernallength - 1)
                    idxint = int(idx)
                    frac = idx - idxint
                    kernal = window[idxint] * (1 - frac) + window[idxint + 1] * frac
                    mat[x, y] += kernal * weight
    return mat

'''
Equation 19
(((W * phi) * R) *phi) * S = w from  Jim Pipe et al. (1999)
It simply means convolve weight with kernel on to R, which is a cartesian grid
then re-sample back to the weigth vector w using the kernel phi from trajectory S
Complexity also O(2pi L^2 N) 
'''
def degrid(mat,traj):
    gridsize = mat.shape[0]
    gridcenter = (gridsize / 2)
    weight = np.zeros(traj.shape[0])
    Kernallength = 32
    kernalwidth = 5 
    window = KaiserBesselwindow(kernalwidth, Kernallength, 1.375)  
    kwidth = kernalwidth / 2 / gridsize
    for n, loc in enumerate(traj):
        kx = loc[0]
        ky = loc[1]
        xmin = int((kx - kwidth) * gridsize + gridcenter)
        xmax = int((kx + kwidth) * gridsize + gridcenter) + 1
        ymin = int((ky - kwidth) * gridsize + gridcenter)
        ymax = int((ky + kwidth) * gridsize + gridcenter) + 1
        if (xmin < 0):
            xmin = 0
        if (xmax >= gridsize):
            xmax = gridsize 
        if (ymin < 0):
            ymin = 0
        if (ymax >= gridsize):
            ymax = gridsize 
        for x in range(xmin, xmax):
            dx = (x - gridcenter) / gridsize - kx
            for y in range(ymin, ymax):
                dy = (y - gridcenter) / gridsize - ky
                d = np.sqrt(dx ** 2 + dy ** 2)
                if (d < kwidth):
                    idx = d / kwidth * (Kernallength - 1)
                    idxint = int(idx)
                    frac = idx - idxint
                    kernal = window[idxint] * (1 - frac) + window[idxint + 1] * frac
                    weight[n] += np.abs(mat[x, y]) * (kernal)
    return weight

'''
The mean loop:
Equation 19
(((W * phi) * R) *phi) * S = w from  Jim Pipe et al. (1999)
'''
def pipedcf(traj, ns):
    dcf = np.ones(ns)
    for i in tqdm(range(10)):
        mat = grid(traj, dcf)
        newdcf = degrid(mat, traj)
        dcf = dcf / newdcf
    return dcf

def voronoidcf(points, threshold = 95):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = 0
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    vol[vol > np.percentile(vol,threshold)] = 0
    return vol
