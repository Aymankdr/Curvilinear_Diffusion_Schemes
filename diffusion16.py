#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:30:29 2024

@author: khaddari
"""

import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import lil_matrix, csr_matrix, eye, diags
from scipy.sparse.linalg import spsolve, eigs, norm
#from diffusion7 import colormap
import diffusion7 as df7
from diffusion14 import spherical, colormap
import assimilation1 as as1
import netCDF4 as nc
import cartopy.crs as ccrs

# intervals in x-, y- directions,
nx, ny = 360, 157
#nx,ny=30,30

## Constants
D = 0.6
dt = 1.0
S = 10 # + int(D**2 * (dx2 + dy2)/(dx2 * dy2))
k0 = D**2/((2*S-4)*dt) # kappa_max



## Boundary Condition
BC = 'periodic'

## Diffusivity
diffusivity='isotropic_and_homogeneous'
#diffusivity='anisotropic_and_inhomogeneous'

## Gradient expression
finite_diff_scheme = 'symmetric'
finite_diff_scheme = 'adaptable'

######################################
## Curvilinear coordinates (sphere) ##
######################################
n1, n2 = nx, ny
N = n1 * (n2 + 1)

## Sphere
R = 2 # radius

theta_secu = np.deg2rad(10) 
theta_init, theta_fin = theta_secu, np.pi - theta_secu

phi_redu = np.deg2rad(0)
dphi, dtheta = (2*np.pi-phi_redu)/n1, (np.pi - 2*theta_secu)/n2

def e1(i,j): return (R*dphi)
#def e1(i,j): return (R*np.sin(j*dtheta + theta_secu)*dphi)
def e2(i,j): return (R*dtheta)

if diffusivity=='isotropic_and_homogeneous':
    def k11(i,j): return k0 #k0*i/n1
    def k22(i,j): return k0 #k0*j/n2
    def k12(i,j): return k0/3.6 #k0*i*j/(n1*n2)
if diffusivity=='anisotropic_and_inhomogeneous':
    def k11(i,j): return 4*k0*(i-n1//2)**2/n1**2
    def k22(i,j): return 4*k0*(j-n2//2)**2/n2**2
    def k12(i,j): return 8*k0*(i-n1//2)**2*(j-n2//2)**2/(n1*n2)**2

def a11(i,j): return e2(i,j)/e1(i,j)*k11(i,j)
def a22(i,j): return e1(i,j)/e2(i,j)*k22(i,j)
def a12(i,j): return k12(i,j)

## Initial conditions - Amplitudes
Tcool, Thot = 0, 1

# Initial conditions - Dirac
def Dirac(i,j,normal=False):
    v = np.zeros(N)
    if normal:  v[j + (ny+1) * i] = 1.0
    else: v[j + (ny+1) * i] = 1.0 / (e1(i+0.5,j+0.5) * e2(i+0.5,j+0.5))
    return v

## Masks

def mask1(n1,n2):
    Mask = np.zeros((n1,n2),dtype=bool)
    N = n2
    # Triangle
    for j in range(N//4):
        Mask[N//4 + j, N//2 + j : 3*N//4] = True
    # Square
    for j in range(N//4):
        Mask[N//2 + j, 3*N//8 : 5*N//8] = True
    return Mask

def Mx(n1,n2,mask=mask1):
    Mask = mask
    Inter= np.logical_or(Mask[1:,:], Mask[:-1,:])
    return 0.5 * np.logical_not(Inter)

def My(n1,n2,mask=mask1):
    Mask = mask
    Inter= np.logical_or(Mask[:,1:], Mask[:,:-1])
    return 0.5 * np.logical_not(Inter)

def Mc(n1,n2,mask=mask1):
    Mask = mask
    Inter= np.logical_or(np.logical_or(Mask[1:,1:], Mask[1:,:-1]),\
            np.logical_or(Mask[:-1,1:], Mask[:-1,:-1]))
    return np.logical_not(Inter)

def Mp(n1,n2,mask=mask1): # mask push
    Mask = mask
    #Inter= Mask[1:,1:]
    Inter= np.logical_and(np.logical_and(Mask[1:,1:], Mask[1:,:-1]),\
            np.logical_and(Mask[:-1,1:], Mask[:-1,:-1]))
    return np.logical_not(Inter)

def Mdiru(n1,n2,mask=mask1):
    Mask = mask
    Inter= np.logical_or(Mask[:-1,:-1], Mask[1:,1:])
    return 0.25 * np.logical_not(Inter) 

def Mdirv(n1,n2,mask=mask1):
    Mask = mask
    Inter= np.logical_or(Mask[1:,:-1], Mask[:-1,1:])
    return 0.25 * np.logical_not(Inter) 

## Zoom
def zoom(global_field, domain):
    """
    Select the values global_field inside the area defined by domain

    Parameters
    ----------
    global_field : array
        Array of dimensions (nlat,nlon) representing any global geophysical field.
    
    domain : tuple
        Domain on which the operator is applied. The domain must be delimited
        by two parallels and two meridians (i.e. it is a 'curved' trapeze). It
        is specified as: (min latitude, max latitude, min longitude, max longitude). 
        The latitudes are specified between -90 and 90 and should be increasing.
        The longitudes are specified between 0 and 360 and do not have to be
        increasing, e.g. (-10, 10, 350, 10) is accepted. 

    Returns
    -------
    filt_mask : array
        Filtered array of booleans equal to 1 on land.

    """
    assert domain[0]<=90 and domain[0]>=-90
    assert domain[1]<=90 and domain[1]>=-90
    assert domain[2]<=360 and domain[2]>=0
    assert domain[3]<=360 and domain[3]>=0

    nlat, nlon = global_field.shape

    nlon_zoom = domain[3] - domain[2] if domain[3] > domain[2] else 360- domain[2] + domain[3]
    
    shifted_field = np.roll(global_field, shift = - domain[2], axis = 1)
    
    return shifted_field[domain[0] + 90: domain[1] + 90, :nlon_zoom]

## Domain
domain = (-80, 80, 0, 360)



# Generate land/sea mask
# -----------------------------------------------------------------------------
# Import from the netCDF file the global 1 degree water ratio map. The function
# zoom select the geographical area of interest
with nc.Dataset("water_ratio.nc", 'r') as data:
    water_ratio = zoom(data['water_ratio_1_1'][:], domain)


# Create a mask from the water ratio 
geo_mask_struc = water_ratio<=0.02

geo_mask = geo_mask_struc.data.T

tri_mask = df7.mask1(n1, n2+1)

## Matrices (Energy FD Scheme)

A11 = np.array([[a11(i+0.5,j+0.5) for j in range(n2)] for i in range(n1)])
A22 = np.array([[a22(i+0.5,j+0.5) for j in range(n2)] for i in range(n1)])
A12 = np.array([[a12(i+0.5,j+0.5) for j in range(n2)] for i in range(n1)])

A11r = np.array([[a11(i+0.5,j) for j in range(n2+1)] for i in range(n1)])
A22r = np.array([[a22(i,j+0.5) for j in range(n2)] for i in range(n1+1)])

E1  = np.array([[e1(i,j) for j in range(n2+1)] for i in range(1,n1+1)])
E2  = np.array([[e2(i,j) for j in range(n2+1)] for i in range(1,n1+1)])
E1E2 = E1 * E2 # G[i,j] = G_i+1,j

## Construct K's coefficients
Aa11 = np.zeros((n1+2,n2+2)); Aa11[1:-1,1:-1] = A11; Aa11[0,:] = Aa11[-2,:]; Aa11[-1,:] = Aa11[1,:]
Aa22 = np.zeros((n1+2,n2+2)); Aa22[1:-1,1:-1] = A22; Aa22[0,:] = Aa22[-2,:]; Aa22[-1,:] = Aa22[1,:]
Aa12 = np.zeros((n1+2,n2+2)); Aa12[1:-1,1:-1] = A12; Aa12[0,:] = Aa12[-2,:]; Aa12[-1,:] = Aa12[1,:]

Aa11r = np.zeros((n1+2,n2+1)); Aa11r[1:-1,:] = A11r;
Aa11r[0,:] = Aa11r[-2,:]; Aa11r[-1,:] = Aa11r[1,:]
Aa22r = np.zeros((n1+1,n2+2)); Aa22r[:,1:-1] = A22r


def neumann_mask(mask):
    mc   = np.logical_not(mask[:,:-1])
    #mc   = np.logical_not(mc) * mc
    global Aa11, Aa12, Aa22, Aa11r, Aa22r
    
    Aa11 = mc * Aa11; Aa12 = mc * Aa12; Aa22 = mc * Aa22
    Aa11r = mc[:,:-1] * Aa11r; Aa22r = mc[:-1] * Aa22r
    
#neumann_mask(geo_mask)

if finite_diff_scheme == 'symmetric':
    Kcc = Aa11[1:,1:] + 2 * Aa12[1:,1:] + Aa22[1:,1:] +\
        Aa11[:-1,1:] - 2 * Aa12[:-1,1:] + Aa22[:-1,1:] +\
        Aa11[1:,:-1] - 2 * Aa12[1:,:-1] + Aa22[1:,:-1] +\
        Aa11[:-1,:-1] + 2 * Aa12[:-1,:-1] + Aa22[:-1,:-1] 
    Krr = - Aa11[1:,1:] + Aa22[1:,1:] - Aa11[1:,:-1] + Aa22[1:,:-1]
    Kll = - Aa11[:-1,1:] + Aa22[:-1,1:] - Aa11[:-1,:-1] + Aa22[:-1,:-1]
    Kuu = Aa11[1:,1:] - Aa22[1:,1:] + Aa11[:-1,1:] - Aa22[:-1,1:]
    Kdd = Aa11[1:,:-1] - Aa22[1:,:-1] + Aa11[:-1,:-1] - Aa22[:-1,:-1]
    Kru = - Aa11[1:,1:] - Aa22[1:,1:] - 2 * Aa12[1:,1:]
    Klu = - Aa11[:-1,1:] - Aa22[:-1,1:] + 2 * Aa12[:-1,1:]
    Krd = - Aa11[1:,:-1] - Aa22[1:,:-1] + 2 * Aa12[1:,:-1]
    Kld = - Aa11[:-1,:-1] - Aa22[:-1,:-1] - 2 * Aa12[:-1,:-1]
    Kcc*=2;Krr*=2;Kll*=2;Kuu*=2;Kdd*=2;Kru*=2;Klu*=2;Krd*=2;Kld*=2
else:
    Kcc = 8 * (Aa11r[:-1,:] + Aa11r[1:,:] + Aa22r[:,:-1] + Aa22r[:,1:]) +\
        2 * (Aa12[:-1,:-1] - Aa12[:-1,1:]) +\
        2 * (Aa12[1:,1:] - Aa12[1:,:-1]) 
    Krr = - 8 * Aa11r[1:,:]
    Kll = - 8 * Aa11r[:-1,:]
    Kuu = - 8 * Aa22r[:,1:]
    Kdd = - 8 * Aa22r[:,:-1]
    Kru = - 2 * Aa12[1:,1:]
    Klu = 2 * Aa12[:-1,1:]
    Krd = 2 * Aa12[1:,:-1]
    Kld = - 2 * Aa12[:-1,:-1]

## Construct M's coefficients
#Gmet = np.zeros((n1+1,n2+2)); Gmet[1:-1,1:-1] = E1E2
#Gmoy = Gmet[1:,1:]+Gmet[1:,:-1]+Gmet[:-1,1:]+Gmet[:-1,:-1]
Gmet = E1E2; Gmet[1:-1,1:-1] *= 4;
Gmet[1:-1,0] *= 2; Gmet[1:-1,-1] *= 2; Gmet[0,1:-1] *= 2; Gmet[0,1:-1] *= 2 

## Construct K
def build_K():
    K = lil_matrix((N,N))
    for i in range(n1):
        for j in range(n2+1):
            k = j + i * (n2 + 1) # true index is (i+1,j)
            # Logicals
            ru = (i == n1 - 1 or j == n2); lu = (i == 0 or j == n2)
            rd = (i == n1 - 1 or j == 0); ld = (i == 0 or j == 0)
            rr = (i == n1 - 1); ll = (i == 0)
            uu = (j == n2); dd = (j == 0)
            cc = (0 < i < n1 - 1 and 0 < j < n2)
            
            K[k,k] = Kcc[i+1,j]
            if not uu: K[k,k+1] = Kuu[i+1,j]
            if not dd: K[k,k-1] = Kdd[i+1,j]
            if not rr: K[k,k+n2+1] = Krr[i+1,j]
            else: K[k,k+n2+1 - N] = Krr[i+1,j]
            if not ll: K[k,k-n2-1] = Kll[i+1,j]
            else: K[k,k-n2-1 + N] = Kll[i+1,j]
            if not rr and not ru: K[k,k+n2+2] = Kru[i+1,j]
            elif rr and (j!=n2): K[k,k+n2+2 - N] = Kru[i+1,j]
            if not ll and not ld: K[k,k-n2-2] = Kld[i+1,j]
            elif ll and (j!=0): K[k,k-n2-2 + N] = Kld[i+1,j]
            if not rr and not rd: K[k,k+n2] = Krd[i+1,j]
            elif rr and (j!=0): K[k,k+n2 - N] = Krd[i+1,j]
            if not ll and not lu: K[k,k-n2] = Klu[i+1,j]
            elif ll and (j!=n2): K[k,k-n2 + N] = Klu[i+1,j]
            
    K = K.tocsr()
    return K
        
## Construct M
Gmet *= 2
def build_M():
    M = Gmet.flatten()
    return M
Mdir = Gmet.flatten()
Minv = 1/Mdir

## Mask tools
def create_stack(mask):
    mc = np.logical_not(mask)
    #mc = mask().T[1:,1:]
    global index_stack, stack_size
    stack_size = sum(sum(mc))
    index_stack=[]
    for i in range(nx):
        for j in range(ny+1):
            if mc[i,j]:
                k = j + i*(ny+1)
                index_stack.append(k)
    return index_stack

## Energy scheme


def energy(u,up):
    return (M * u) @ up - 0.5 * (M * u) @ u - 0.5 * (K @ u) @ u

def energy_scheme(u,mask=None,nb_iter=S):
    E_list = []
    v = np.copy(u)
    ## Construct A=M+K
    M = build_M()
    K = build_K()
    A = diags(M) + K
    if mask is None:
        for s in range(nb_iter):
            y = M * v
            w = spsolve(A,y)
            e = (M * w) @ v - 0.5 * (M * w) @ w - 0.5 * (K @ w) @ w
            E_list.append(-e)
            v = w
    else:
        index_stack = create_stack(mask)
        # Create the A_small matrix
        K_small = K[index_stack, :][:, index_stack]
        M_small = M[index_stack]
        # Convert to csr_matrix for efficiency if needed
        K_small = csr_matrix(K_small)
        A_small = K_small + diags(M_small)
        # Extract the subvector
        v_small = v[index_stack]
        for m in range(nb_iter):
            y_small = M_small * v_small
            w_small = spsolve(A_small,y_small)
            e = (M_small * w_small) @ v_small -\
                0.5 * (M_small * w_small) @ w_small - 0.5 * (K_small @ w_small) @ w_small
            E_list.append(-e)
            v_small = w_small
        v[:] = 0
        v[index_stack] = v_small
    return v,E_list

## FE scheme
def b_form(corner1, corner2, method='fast'):
    if method=='fast':
        xc,yc = (corner1[0]+corner2[0])/2,(corner1[1]+corner2[1])/2
        e1c, e2c = e1(xc,yc), e2(xc,yc)
        diag_index = (corner1[0]-corner2[0])*(corner1[1]-corner2[1])
        x_diff,y_diff = abs(corner1[0]-corner2[0]),abs(corner1[1]-corner2[1])
        if diag_index == 0.0:
            if x_diff == 0.0 and y_diff == 0.0:
                return e1c * e2c * 4/9
            elif (x_diff == 1.0 and y_diff == 0.0) or (x_diff == 0.0 and y_diff == 1.0):
                return e1c * e2c * 1/9
        elif abs(diag_index) == 1.0:
            return e1c * e2c * 1/36
        else:
            return 0.0
    else:
        a,b = min(corner1[0]-1.,corner2[0]-1.),max(corner1[0]+1.,corner2[0]+1.)
        c,d = min(corner1[1]-1.,corner2[1]-1.),max(corner1[1]+1.,corner2[1]+1.)
        func = lambda x,y:phi(x,y,corner1)*phi(x,y,corner2)*df7.e1(x,y)*df7.e2(x,y)
        return dblquad(func, c, d, a, b)[0]

def a_form(corner1, corner2, method='fast'):
    if method=='fast':
        xc,yc = (corner1[0]+corner2[0])/2,(corner1[1]+corner2[1])/2
        a11c, a12c, a22c = a11(xc,yc), a12(xc,yc), a22(xc,yc)
        diag_index = (corner1[0]-corner2[0])*(corner1[1]-corner2[1])
        x_diff,y_diff = abs(corner1[0]-corner2[0]),abs(corner1[1]-corner2[1])
        if diag_index == 0.0:
            if x_diff == 0.0:
                if y_diff == 0.0:
                    return a11c * 4/3 + a22c * 4/3
                elif y_diff == 1.0:
                    return a11c * 1/3 - a22c * 2/3
                else:
                    return 0.0
            elif x_diff == 1.0:
                return - a11c * 2/3 + a22c * 1/3
            else:
                return 0.0
        elif diag_index == 1.0:
            return - a11c * 1/6 - a12c * 1/2 - a22c * 1/6
        elif diag_index == -1.0:
            return - a11c * 1/6 + a12c * 1/2 - a22c * 1/6
        else:
            return 0.0
    else:
        a,b = min(corner1[0]-1.,corner2[0]-1.),max(corner1[0]+1.,corner2[0]+1.)
        c,d = min(corner1[1]-1.,corner2[1]-1.),max(corner1[1]+1.,corner2[1]+1.)
        func1 = lambda x,y:phi(x,y,corner1)*phi(x,y,corner2)*df7.e1(x,y)*df7.e2(x,y)
        term1 = dblquad(func1, c, d, a, b)
        func2 = lambda x,y:df7.a11(x,y)*dx_phi(x,y,corner1)*dx_phi(x,y,corner2)+\
            df7.a12(x,y)*(dx_phi(x,y,corner1)*dy_phi(x,y,corner2)+dy_phi(x,y,corner1)*dx_phi(x,y,corner2))+\
                df7.a22(x,y)*dy_phi(x,y,corner1)*dy_phi(x,y,corner2)
        term2 = dblquad(func2, c, d, a, b)
        return (term1[0] + term2[0]) #,term1[1] + term2[1])

def construct_A(nx,ny,mask=None):
    N = nx * ny
    A = lil_matrix((N, N))
    if mask is None:
        for i in range(nx):
            for j in range(ny):
                k = j + i*ny
                A[k,k] = a_form([i,j], [i,j])
                if j>0: A[k,k-1] = a_form([i,j], [i,j-1])
                if j<ny-1: A[k,k+1] = a_form([i,j], [i,j+1])
                if i>0: A[k,k-ny] = a_form([i,j], [i-1,j])
                if i<nx-1: A[k,k+ny] = a_form([i,j], [i+1,j])
                if j>0 and i>0: A[k,k-ny-1] = a_form([i,j], [i-1,j-1])
                if j<ny-1 and i>0: A[k,k-ny+1] = a_form([i,j], [i-1,j+1])
                if j>0 and i<nx-1: A[k,k+ny-1] = a_form([i,j], [i+1,j-1])
                if j<ny-1 and i<nx-1: A[k,k+ny+1] = a_form([i,j], [i+1,j+1])
    else:
        mc = np.logical_not(mask).T
        #mc = mask().T[1:,1:]
        for i in range(nx):
            for j in range(ny):
                if mc[j,i]:
                    k = j + i*ny
                    A[k,k] = a_form([i,j], [i,j])
                    if j>0 and mc[j-1,i]: A[k,k-1] = a_form([i,j], [i,j-1])
                    if j<ny-1 and mc[j+1,i]: A[k,k+1] = a_form([i,j], [i,j+1])
                    if i>0 and  mc[j,i-1]: A[k,k-ny] = a_form([i,j], [i-1,j])
                    if i<nx-1 and mc[j,i+1]: A[k,k+ny] = a_form([i,j], [i+1,j])
                    if j>0 and i>0 and mc[j-1,i-1]: A[k,k-ny-1] = a_form([i,j], [i-1,j-1])
                    if j<ny-1 and i>0 and mc[j+1,i-1]: A[k,k-ny+1] = a_form([i,j], [i-1,j+1])
                    if j>0 and i<nx-1 and mc[j-1,i+1]: A[k,k+ny-1] = a_form([i,j], [i+1,j-1])
                    if j<ny-1 and i<nx-1 and mc[j+1,i+1]: A[k,k+ny+1] = a_form([i,j], [i+1,j+1])
    return A.tocsr()

def construct_B(nx,ny,mask=None):
    N = nx * ny
    B = lil_matrix((N, N))
    if mask is None:
        for i in range(nx):
            for j in range(ny):
                k = j + i*ny
                B[k,k] = b_form([i,j], [i,j])
                if j>0: B[k,k-1] = b_form([i,j], [i,j-1])
                if j<ny-1: B[k,k+1] = b_form([i,j], [i,j+1])
                if i>0: B[k,k-ny] = b_form([i,j], [i-1,j])
                if i<nx-1: B[k,k+ny] = b_form([i,j], [i+1,j])
                if j>0 and i>0: B[k,k-ny-1] = b_form([i,j], [i-1,j-1])
                if j<ny-1 and i>0: B[k,k-ny+1] = b_form([i,j], [i-1,j+1])
                if j>0 and i<nx-1: B[k,k+ny-1] = b_form([i,j], [i+1,j-1])
                if j<ny-1 and i<nx-1: B[k,k+ny+1] = b_form([i,j], [i+1,j+1])
    else:
        mc = np.logical_not(mask).T
        #mc = mask().T[1:,1:]
        for i in range(nx):
            for j in range(ny):
                if mc[j,i]:
                    k = j + i*ny
                    B[k,k] = b_form([i,j], [i,j])
                    if j>0 and mc[j-1,i]: B[k,k-1] = b_form([i,j], [i,j-1])
                    if j<ny-1 and mc[j+1,i]: B[k,k+1] = b_form([i,j], [i,j+1])
                    if i>0 and  mc[j,i-1]: B[k,k-ny] = b_form([i,j], [i-1,j])
                    if i<nx-1 and mc[j,i+1]: B[k,k+ny] = b_form([i,j], [i+1,j])
                    if j>0 and i>0 and mc[j-1,i-1]: B[k,k-ny-1] = b_form([i,j], [i-1,j-1])
                    if j<ny-1 and i>0 and mc[j+1,i-1]: B[k,k-ny+1] = b_form([i,j], [i-1,j+1])
                    if j>0 and i<nx-1 and mc[j-1,i+1]: B[k,k+ny-1] = b_form([i,j], [i+1,j-1])
                    if j<ny-1 and i<nx-1 and mc[j+1,i+1]: B[k,k+ny+1] = b_form([i,j], [i+1,j+1])
    return B.tocsr()

def FEscheme(u0,mask=None,diag_M=False,solver='conjgrad',nb_iter=S):
    global K,M
    K = construct_A(nx,ny+1,mask=mask)
    M = construct_B(nx,ny+1,mask=mask)
    C = M+K
    u = np.copy(u0)
    E_list = []
    if diag_M:
        # Compute the row sums
        row_sums = M.sum(axis=1).A1  # .A1 converts it to a 1D array
        # Create a diagonal matrix with these sums
        M = diags(row_sums)
        M = M.tocsr()
    if mask is None:
        for m in range(nb_iter):
            v = M.dot(u)
            if solver=='conjgrad': u = as1.conjgrad(C, v, v, nature=0)
            else: u = spsolve(C,v)
            e = (M @ u) @ v -\
                0.5 * (M @ u) @ u - 0.5 * (K @ u) @ u
            E_list.append(e)
    else:
        index_stack = create_stack(mask)
        # Create the A_small matrix
        K_small = K[index_stack, :][:, index_stack]
        M_small = M[index_stack, :][:, index_stack]
        # Convert to csr_matrix for efficiency if needed
        K_small = csr_matrix(K_small)
        M_small = csr_matrix(M_small)
        C_small = K_small + M_small
        # Extract the subvector
        u_small = u[index_stack]
        for m in range(nb_iter):
            v_small = M_small.dot(u_small)
            if solver=='conjgrad': u_small = as1.conjgrad(C_small, v_small, v_small, nature=0)
            else: u_small = spsolve(C_small,v_small)
            e = (M_small @ u_small) @ v_small -\
                0.5 * (M_small @ u_small) @ u_small - 0.5 * (K_small @ u_small) @ u_small
            E_list.append(e)
        u[:] = 0
        u[index_stack] = u_small
    return u, E_list

gamma_exists = False
def FEcorrelation(u0,mask=None,diag_M=False,solver='conjgrad',nb_iter=S,Ne=25):
    global K,M
    K = construct_A(nx,ny+1,mask=mask)
    M = construct_B(nx,ny+1,mask=mask)
    Ml = M.sum(axis=1)  # Sum the rows
    Ml = np.array(Ml).flatten()  # Convert to a 1D array
    # Compute the row sums
    row_sums = M.sum(axis=1).A1  # .A1 converts it to a 1D array
    # Create a diagonal matrix with these sums
    M = diags(row_sums)
    M = M.tocsr()
    C = M+K
    u = np.copy(u0)
    # randomisation
    Mlrac = np.sqrt(Ml)
    global gamma, gamma_exists
    if not gamma_exists:
        if mask is None:
            s = np.zeros(n1*(n2+1))
            for n in range(Ne):
                v0 = np.random.normal(size=n1*(n2+1))
                v1 = v0
                ## Layer 1
                v = v1
                for m in range(nb_iter//2):
                    if solver=='conjgrad': u = as1.conjgrad(C, v, v, nature=0)
                    else: u = spsolve(C,v)
                    v = Ml * u
                ## Layer 2
                u = v / Mlrac
                v2= u
                v3 = v2 * v2
                s += v3
                print("Step: ",n)
            v4 = s/Ne
            for i in range(n1*(n2+1)):
                if v4[i]==0.0: v4[i]=1.0
                #if mc[i]: v4[i]=1.0
            v5 = 1/np.sqrt(v4)
            v6 = v5
        else:
            index_stack = create_stack(mask)
            # Create the A_small matrix
            K_small = K[index_stack, :][:, index_stack]
            M_small = M[index_stack, :][:, index_stack]
            # Convert to csr_matrix for efficiency if needed
            K_small = csr_matrix(K_small)
            M_small = csr_matrix(M_small)
            C_small = K_small + M_small
            # Extract the subvector
            Ml_small= Ml[index_stack]
            Mlrac_small = Mlrac[index_stack]
            u_small = u[index_stack]
            ## Layer 1
            v_small = u_small
            ## beg rand
            s = np.zeros(len(index_stack))
            for n in range(Ne):
                v0 = np.random.normal(size=len(index_stack))
                v_small = v0
                for m in range(nb_iter//2):
                    if solver=='conjgrad': u_small = as1.conjgrad(C_small, v_small, v_small, nature=0)
                    else: u_small = spsolve(C_small,v_small)
                    v_small = Ml_small * u_small
                ## Layer 2
                u_small = v_small / Mlrac_small
                v3 = u_small * u_small
                s += v3
                print("Step: ",n)
            v4 = s/Ne
            v5 = 1 / np.sqrt(v4)
            v6 = np.zeros(nx*(ny+1))
            v6[index_stack] = v5
        # to global
        gamma = v6
    # Calculation
    u = gamma * np.copy(u0)
    #u = np.copy(u0)
    if diag_M:
        # Compute the row sums
        row_sums = M.sum(axis=1).A1  # .A1 converts it to a 1D array
        # Create a diagonal matrix with these sums
        M = diags(row_sums)
        M = M.tocsr()
    if mask is None:
        ## Layer 1
        v = u
        for m in range(nb_iter//2):
            if solver=='conjgrad': u = as1.conjgrad(C, v, v, nature=0)
            else: u = spsolve(C,v)
            v = Ml * u
        ## Layer 2
        v = u
        u = v / Ml
        ## Layer 3
        for m in range(nb_iter//2):
            v = Ml * u
            if solver=='conjgrad': u = as1.conjgrad(C, v, v, nature=0)
            else: u = spsolve(C,v)
        
    else:
        index_stack = create_stack(mask)
        # Create the A_small matrix
        K_small = K[index_stack, :][:, index_stack]
        M_small = M[index_stack, :][:, index_stack]
        # Convert to csr_matrix for efficiency if needed
        K_small = csr_matrix(K_small)
        M_small = csr_matrix(M_small)
        C_small = K_small + M_small
        # Extract the subvector
        Ml_small= Ml[index_stack]
        u_small = u[index_stack]
        ## Layer 1
        v_small = u_small
        for m in range(nb_iter//2):
            if solver=='conjgrad': u_small = as1.conjgrad(C_small, v_small, v_small, nature=0)
            else: u_small = spsolve(C_small,v_small)
            v_small = Ml_small * u_small
        ## Layer 2
        u_small = v_small / Ml_small
        ## Layer 3
        for m in range(nb_iter//2):
            v_small = Ml_small * u_small
            if solver=='conjgrad': u_small = as1.conjgrad(C_small, v_small, v_small, nature=0)
            else: u_small = spsolve(C_small,v_small)
        u[:] = 0
        u[index_stack] = u_small
    return gamma * u

## Finite differences Symmetric scheme
BC = ['Neumann','Neumann','Neumann','Neumann']

def fill_edges(W, nx, ny, is_mask = False):
    # North
    if is_mask == True:
        n1,n2 = len(W), len(W[0])
        M_BIG = np.ones((n1+2,n2+2),dtype=bool)
        M_BIG[1:-1,1:-1] = W
        return M_BIG
    
    if BC[0] == 'Dirichlet': 
        W[:,ny+1] = - W[:,ny]
    elif BC[0] == 'Neumann': 
        W[:,ny+1] = + W[:,ny]
    else: 
        print('method not available')
    # South
    if BC[1] == 'Dirichlet': 
        W[:,0] = - W[:,1]
    elif BC[1] == 'Neumann': 
        W[:,0] = + W[:,1]
    else: 
        print('method not available')
    # East
    if BC[2] == 'Dirichlet': 
        W[nx+1,:] = - W[nx,:]
    elif BC[2] == 'Neumann': 
        W[nx+1,:] = + W[nx,:]
    else: 
        print('method not available')
    # West
    if BC[3] == 'Dirichlet': 
        W[0,:] = - W[1,:]
    elif BC[3] == 'Neumann': 
        W[0,:] = + W[1,:]
    else: 
        print('method not available')
    return W

A11 = np.array([[a11(i,j) for j in range(n2+2)] for i in range(n1+1)])
A22 = np.array([[a22(i,j) for j in range(n2+2)] for i in range(n1+1)])
A12 = np.array([[a12(i,j) for j in range(n2+2)] for i in range(n1+1)])

E1  = np.array([[e1(i+0.5,j+0.5) for j in range(n2+1)] for i in range(n1)])
E2  = np.array([[e2(i+0.5,j+0.5) for j in range(n2+1)] for i in range(n1)])
E1E2 = E1 * E2

def previous_step_curv(v0, n1, n2, mask=None, mask_type='diag', scheme='symmetric'):
    W0 = v0.copy()
    if mask_type == 'holes': W0 = np.logical_not(mask[1:-1,1:-1].flatten()) * W0
    W0.resize((n1,n2))
    W  = np.zeros((n1+2,n2+2))
    W[1:-1,1:-1] = W0
    W  = fill_edges(W, n1, n2)
    global mx,my,mc,mdiru,mdirv

    # Rectangle centers
    if mask is None:
        diW   = 0.5*(W[1:,1:] + W[1:,:-1] - W[:-1,1:] - W[:-1,:-1])
        djW   = 0.5*(W[1:,1:] + W[:-1,1:] - W[1:,:-1] - W[:-1,:-1])
        # Vertex nods
        Qi    = - (A11 * diW + A12 * djW)
        Qj    = - (A12 * diW + A22 * djW)
        # Back to centers
        divQ  = (Qi[1:,1:] + Qi[1:,:-1] - Qi[:-1,1:] - Qi[:-1,:-1])/(2)
        divQ += (Qj[1:,1:] - Qj[1:,:-1] + Qj[:-1,1:] - Qj[:-1,:-1])/(2)
    else:
        if mask_type == 'ridge center':
            #mx    = Mx(mask)
            #my    = My(mask)
            diW   = mx[:,1:] * (W[1:,1:] - W[:-1,1:]) + mx[:,:-1] * (W[1:,:-1] - W[:-1,:-1])
            djW   = my[1:,:] * (W[1:,1:] - W[1:,:-1]) + my[:-1,:] * (W[:-1,1:] - W[:-1,:-1])
            # Vertex nods
            Qi    = - (A11 * diW + A12 * djW)
            Qj    = - (A12 * diW + A22 * djW)
            # Back to centers
            divQ  = my[1:-1,1:] * (Qi[1:,1:] - Qi[:-1,1:]) + my[1:-1,:-1] * (Qi[1:,:-1] - Qi[:-1,:-1])
            divQ += mx[1:,1:-1] * (Qj[1:,1:] - Qj[1:,:-1]) + mx[:-1,1:-1] * (Qj[:-1,1:] - Qj[:-1,:-1])
        elif mask_type == 'diag':
            #mc    = Mc(mask)
            m1 = mdiru + mdirv
            m2 = mdiru - mdirv
            diW   = (W[1:,1:] + W[1:,:-1] - W[:-1,1:] - W[:-1,:-1])
            djW   = (W[1:,1:] + W[:-1,1:] - W[1:,:-1] - W[:-1,:-1])
            # Masking diagonally / in vertices
            mdiW  = m1 * diW + m2 * djW
            mdjW  = m2 * diW + m1 * djW
            # Flux in Vertex nods
            Qi    = - (A11 * mdiW + A12 * mdjW)
            Qj    = - (A12 * mdiW + A22 * mdjW)
            # Masking again
            mQi   = m1 * Qi + m2 * Qj
            mQj   = m2 * Qi + m1 * Qj
            # Back to centers
            divQ  = (mQi[1:,1:] + mQi[1:,:-1] - mQi[:-1,1:] - mQi[:-1,:-1])
            divQ += (mQj[1:,1:] - mQj[1:,:-1] + mQj[:-1,1:] - mQj[:-1,:-1])
        elif mask_type == 'holes':
            diW   = 0.5*(W[1:,1:] + W[1:,:-1] - W[:-1,1:] - W[:-1,:-1])
            djW   = 0.5*(W[1:,1:] + W[:-1,1:] - W[1:,:-1] - W[:-1,:-1])
            # Vertex nods
            Qi    = - (A11 * diW + A12 * djW)
            Qj    = - (A12 * diW + A22 * djW)
            # Back to centers
            divQ  = (Qi[1:,1:] + Qi[1:,:-1] - Qi[:-1,1:] - Qi[:-1,:-1])/(2)
            divQ += (Qj[1:,1:] - Qj[1:,:-1] + Qj[:-1,1:] - Qj[:-1,:-1])/(2)
        else:
            #mc    = Mc(mask)
            diW   = (W[1:,1:] + W[1:,:-1] - W[:-1,1:] - W[:-1,:-1])
            djW   = (W[1:,1:] + W[:-1,1:] - W[1:,:-1] - W[:-1,:-1])
            # Vertex nods
            Qi    = mc * -0.5 * (A11 * diW + A12 * djW)
            Qj    = mc * -0.5 * (A12 * diW + A22 * djW)
            # Back to centers
            divQ  = (Qi[1:,1:] + Qi[1:,:-1] - Qi[:-1,1:] - Qi[:-1,:-1])/(2)
            divQ += (Qj[1:,1:] - Qj[1:,:-1] + Qj[:-1,1:] - Qj[:-1,:-1])/(2)
    
    W[1:-1,1:-1] += dt * divQ / E1E2
    
    W0 = W[1:-1,1:-1]
    v  = W0.copy() 
    v.resize(n1*n2)
    if mask_type == 'holes': v = np.logical_not(mask[1:-1,1:-1].flatten()) * v
    return v

def implicit_operator_curv(v0, mask=None, mask_type='diag', scheme='symmetric',nb_iter=S):
    w = v0.copy()
    global mx,my,mc,mdiru,mdirv
    if not mask is None:
        mx    = Mx(n1,n2,mask)
        my    = My(n1,n2,mask)
        mc    = Mc(n1,n2,mask)
        mdiru = Mdiru(n1,n2,mask)
        mdirv = Mdirv(n1,n2,mask)
    for m in range(nb_iter):
        previous_step_op = lambda v0:previous_step_curv(v0, n1, n2+1, mask=mask,\
                                                        mask_type=mask_type,scheme=scheme)
        w = as1.conjgrad(previous_step_op,w,w)
    return w

## Metric Matrix
gg = 1 / E1E2.flatten()
gd = np.sqrt(gg)

## Algorithm
def rand_inv_gamma(Ne,mask):
    s = np.zeros(n1*(n2+1))
    for n in range(Ne):
        v0 = np.random.normal(size=n1*(n2+1))
        v1 = gd * v0
        #v2 = implicit_operator_curv(v1,mask=mask,nb_iter=S//2,mask_type='holes')
        #v2 = FEscheme(v1,mask=mask,nb_iter=S//2)[0]
        v2 = energy_scheme(v1,mask=mask,nb_iter=S//2)[0]
        v3 = v2 * v2
        s += v3
        print("Step: ",n)
    v4 = s/Ne
    #mc = mask[1:-1,1:-1].flatten()
    for i in range(n1*(n2+1)):
        if v4[i]==0.0: v4[i]=1.0
        #if mc[i]: v4[i]=1.0
    v5 = 1/np.sqrt(v4)
    return v5

def algo(nb_iter=S,mask=None,u0=None,toshow='flat',Ne=25):
    '''
    if type(u0)!=type(np.zeros(2)): u0 = 0.25*(Dirac(nx//3,ny//2,True)+\
                                               Dirac(nx//3+1,ny//2,True)+\
                                               Dirac(nx//3,ny//2+1,True)+\
                                               Dirac(nx//3+1,ny//2+1,True))'''
    if type(u0)!=type(np.zeros(2)): u0 = Dirac(nx//2,ny//2,True)
    gamma = rand_inv_gamma(Ne, mask)
    #gamma = 1; #gg = 1
    E_list = []
    vv,E_list = energy_scheme(gg * (gamma * u0),mask=mask,nb_iter=nb_iter)
    #vv,E_list = FEscheme(gg * (gamma * u0),mask=mask,nb_iter=nb_iter)
    #vv,E_list = FEscheme(u0,mask=mask,nb_iter=nb_iter)
    #vv = implicit_operator_curv(gg * (gamma * u0),mask=mask,nb_iter=nb_iter,mask_type='holes')
    cc    = gamma * vv
    #cc = FEcorrelation(u0,mask=mask,nb_iter=nb_iter,Ne=Ne)
    #gamma = np.sqrt(1/gv.max())
    #kind  = gv.argmax()
    #vvv,E_list = energy_scheme(u0 * gamma,mask=mask,nb_iter=nb_iter)
    #gv    = gg * vvv
    #print(gamma,kind)
    #print(vv.dot(vv))
    #colormap(vv,nx=n1,ny=n2+1,nb_iter=nb_iter,mask=mask) 
    #plt.show()
    if toshow=='flat':
        colormap(cc,nx=n1,ny=n2+1,nb_iter=nb_iter,mask=mask) 
        plt.show()
    elif toshow=='energy':
        s_list = list(range(nb_iter))
        plt.plot(s_list,E_list)
        plt.show()
    else:
        spherical(vv, nx=n1, ny=n2+1, mask=mask, theta_secu=theta_secu)
        plt.show()
    return cc, E_list, gamma#, vv2
    
def sparsity_pattern(X):
    x = np.array(X.todense())
    fig, axs = plt.subplots(2, 2)
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]
    ax1.spy(x, markersize=5)
    ax2.spy(x, precision=1e-10, markersize=5)
    ax3.spy(x)
    ax4.spy(x, precision=1e-10)
    plt.show()
    
def random_spectral_radius(A,Minv):
    u=np.random.normal(size=N); s=(Minv*(A.dot(u))).dot(u)/(u.dot(u));
    return s




