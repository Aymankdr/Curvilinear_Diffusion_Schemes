#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:36:13 2024

@author: khaddari
"""

import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import assimilation1 as as1

# plate size, mm
Lx, Ly = 10., 7.
# intervals in x-, y- directions, mm
nx, ny = 50, 50
dx, dy = Lx/nx, Ly/ny
dx2, dy2 = dx*dx, dy*dy

## Constants
D = 0.15
dt = 1.0
M = 25 + int(D**2 * (dx2 + dy2)/(dx2 * dy2))
k0 = D**2/((2*M-4)*dt) # kappa_max

# Mesh
xedge = np.linspace(-dx/2, Lx + dx/2, nx + 2)
yedge = np.linspace(-dy/2, Ly + dy/2, ny + 2)

# Normalization diagonal Matrix
NDM = []
NDM_filled = False

# Thermal diffusivity of steel, mm2.s-1
case = 1
if case == 1:
    def kxx(x,y): return k0 #1*k0*y/Ly
    def kyy(x,y): return k0 #1*k0*x/Lx
    def kxy(x,y): return 0  #k0*x*y/(Lx*Ly)
elif case == 2:
    def kxx(x,y): return 1*k0*y/Ly
    def kyy(x,y): return 1*k0*x/Lx
    def kxy(x,y): return 0
else:
    def kxx(x,y): return 1*k0*y/Ly
    def kyy(x,y): return 1*k0*x/Lx
    def kxy(x,y): return k0*x*y/(Lx*Ly)



Kxx = np.array([[kxx(i*dx,j*dy) for j in range(ny+1)] for i in range(nx+1)])
Kyy = np.array([[kyy(i*dx,j*dy) for j in range(ny+1)] for i in range(nx+1)])
Kxy = np.array([[kxy(i*dx,j*dy) for j in range(ny+1)] for i in range(nx+1)])

# Amplitudes
Tcool, Thot = 0, 1

# Initial conditions - circle of radius r centred at (cx,cy) (mm)
r, cx, cy = 0.2*min(Lx,Ly), Lx/2, Ly/2
r2 = r**2
CIcase = 'modal'
CIcase = 'circle'
ang1, ang2 = 2, 2
if CIcase == 'modal':
    chosen_type = complex
else:
    chosen_type = float
            
def wi(x,y,r,cx,cy,case=CIcase):
    if case == 'circle':
        p2 = (x - cx)**2 + (y - cy)**2
        if p2 < r2:
            return Thot
        else: return Tcool
    else:
        z = x * ang1 + y * ang2
        return np.cos(z) + 1j * np.sin(z)
    
def U0(r,cx,cy,case=CIcase):
    u0 = Tcool * np.ones((nx, ny), dtype=chosen_type)
    for i in range(nx):
        for j in range(ny):
            u0[i,j] = wi((i+0.5)*dx,(j+0.5)*dy,r,cx,cy)
    u0.resize(nx*ny)
    return u0

u0 = U0(r,cx,cy)

# Boundary conditions

def north(x,CL='Neumann'):
    if CL=='Neumann':
        return 0
    if CL=='Dirichlet':
        return 0
    else: # condition mixte
        return 0
    
def south(x,CL='Neumann'):
    if CL=='Neumann':
        return 0
    if CL=='Dirichlet':
        return 0
    else: # condition mixte
        return 0

def east(y,CL='Neumann'):
    if CL=='Neumann':
        return 0
    if CL=='Dirichlet':
        return 0
    else: # condition mixte
        return 0
    
def west(y,CL='Neumann'):
    if CL=='Neumann':
        return 0
    if CL=='Dirichlet':
        return 0
    else: # condition mixte
        return 0

BC = ['Neumann','Neumann','Neumann','Neumann']

# Explicit scheme

def fill_edges(W, is_mask = False):
    # North
    if is_mask == True:
        n1,n2 = len(W), len(W[0])
        M_BIG = np.ones((n1+2,n2+2),dtype=bool)
        M_BIG[1:-1,1:-1] = W
        return M_BIG
    
    if BC[0] == 'Dirichlet': 
        W[:,ny+1] = 2*north(xedge,CL=BC[0]) - W[:,ny]
    elif BC[0] == 'Neumann': 
        W[:,ny+1] = dy*north(xedge,CL=BC[0]) + W[:,ny]
    else: 
        print('method not available')
    # South
    if BC[1] == 'Dirichlet': 
        W[:,0] = 2*south(xedge,CL=BC[1]) - W[:,1]
    elif BC[1] == 'Neumann': 
        W[:,0] = dy*south(xedge,CL=BC[1]) + W[:,1]
    else: 
        print('method not available')
    # East
    if BC[2] == 'Dirichlet': 
        W[nx+1,:] = 2*east(yedge,CL=BC[2]) - W[nx,:]
    elif BC[2] == 'Neumann': 
        W[nx+1,:] = dx*east(yedge,CL=BC[2]) + W[nx,:]
    else: 
        print('method not available')
    # West
    if BC[3] == 'Dirichlet': 
        W[0,:] = 2*west(yedge,CL=BC[3]) - W[1,:]
    elif BC[3] == 'Neumann': 
        W[0,:] = dx*west(yedge,CL=BC[3]) + W[1,:]
    else: 
        print('method not available')
    return W

def explicit_step(W):
    
    # Rectangle centers
    dxW   = (W[1:,1:] + W[1:,:-1] - W[:-1,1:] - W[:-1,:-1])/(2*dx)
    dyW   = (W[1:,1:] + W[:-1,1:] - W[1:,:-1] - W[:-1,:-1])/(2*dy)
    
    Qx    = - (Kxx * dxW + Kxy * dyW)
    Qy    = - (Kxy * dxW + Kyy * dyW)
    
    divQ  = (Qx[1:,1:] + Qx[1:,:-1] - Qx[:-1,1:] - Qx[:-1,:-1])/(2*dx)
    divQ += (Qy[1:,1:] - Qy[1:,:-1] + Qy[:-1,1:] - Qy[:-1,:-1])/(2*dy)
    
    W[1:-1,1:-1] = W[1:-1,1:-1] - dt * divQ
    
    W = fill_edges(W)
    W0 = W.copy()
    
    return W0

def adj_explicit_step(W):
    
    # Rectangle centers
    dxW   = (W[1:,1:] + W[1:,:-1] - W[:-1,1:] - W[:-1,:-1])/(2*dx)
    dyW   = (W[1:,1:] + W[:-1,1:] - W[1:,:-1] - W[:-1,:-1])/(2*dy)
    
    Qx    = - (Kxx * dxW + Kxy * dyW)
    Qy    = - (Kxy * dxW + Kyy * dyW)
    
    divQ  = (Qx[1:,1:] + Qx[1:,:-1] - Qx[:-1,1:] - Qx[:-1,:-1])/(2*dx)
    divQ += (Qy[1:,1:] - Qy[1:,:-1] + Qy[:-1,1:] - Qy[:-1,:-1])/(2*dy)
    
    W[1:-1,1:-1] = W[1:-1,1:-1] - dt * divQ
    
    W = fill_edges(W)
    W0 = W.copy()
    
    return W0

def previous_step(v0):
    W0 = v0.copy()
    W0.resize((nx,ny))
    W  = np.zeros((nx+2,ny+2),dtype=chosen_type)
    W[1:-1,1:-1] = W0
    W  = fill_edges(W)
    
    # Rectangle centers
    dxW   = (W[1:,1:] + W[1:,:-1] - W[:-1,1:] - W[:-1,:-1])/(2*dx)
    dyW   = (W[1:,1:] + W[:-1,1:] - W[1:,:-1] - W[:-1,:-1])/(2*dy)
    
    Qx    = - (Kxx * dxW + Kxy * dyW)
    Qy    = - (Kxy * dxW + Kyy * dyW)
    
    divQ  = (Qx[1:,1:] + Qx[1:,:-1] - Qx[:-1,1:] - Qx[:-1,:-1])/(2*dx)
    divQ += (Qy[1:,1:] - Qy[1:,:-1] + Qy[:-1,1:] - Qy[:-1,:-1])/(2*dy)
    
    W[1:-1,1:-1] += dt * divQ
    
    W0 = W[1:-1,1:-1]
    v  = W0.copy() 
    v.resize(nx*ny)
    return v

def explicit_operator(v0):
    w0 = v0.copy()
    w0.resize((nx,ny))
    w  = np.zeros((nx+2,ny+2))
    w[1:-1,1:-1] = w0
    w = fill_edges(w)
    for m in range(M):
        w = explicit_step(w)
    w0 = w[1:-1,1:-1]
    v  = w0.copy() 
    v.resize(nx*ny)
    return v

def implicit_operator(v0):
    w = v0.copy()
    for m in range(M):
        w = as1.conjgrad(previous_step,w,w)
    return w


Lambda = []
Lambda_created = False
def create_Lambda():
    global Lambda, Lambda_created
    z = np.zeros(nx*ny)
    print(1,end="")
    for k in range(nx*ny):
        z[k], z[k-1] = 1.0, 0.0
        val = explicit_operator(z)[k]
        Lambda.append(1/np.sqrt(val))
        print(".",end="")
    print(nx*ny,end="\n")
    Lambda = np.array(Lambda)
    Lambda_created = True

def correlation_operator(u):
    if not Lambda_created: create_Lambda()
    v = Lambda * u
    w = explicit_operator(v)
    x = Lambda * w
    return x

def test_L(mu=0,sig=1):
    u = np.random.normal(mu,sig,size = nx*ny)
    v = np.random.normal(mu,sig,size = nx*ny)
    return u.dot(explicit_operator(v)) - v.dot(explicit_operator(u))

def test_C(mu=0,sig=1):
    u = np.random.normal(mu,sig,size = nx*ny)
    v = np.random.normal(mu,sig,size = nx*ny)
    return u.dot(correlation_operator(v)) - v.dot(correlation_operator(u))

def test_op(oper,mu=0,sig=1,mask=None,\
            mask_type='ridge center',scheme='symmetric'):
    u = np.random.normal(mu,sig,size = nx*ny)
    v = np.random.normal(mu,sig,size = nx*ny)
    return u.dot(oper(v,mask=mask,mask_type=mask_type,scheme=scheme)) -\
        v.dot(oper(u,mask=mask,mask_type=mask_type,scheme=scheme))

def power_iteration(oper, dim:int, num_iterations = 10):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(dim)

    for _ in range(num_iterations):
        # calculate the matrix-by-vector product Ab
        b_k1 = oper(b_k)

        # calculate the norm
        b_k1_norm = alg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k1_norm

def colormap(v,mask=None,nx=nx,ny=ny,nb_iter=M):
    w = v.copy()
    w.resize((nx, ny))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('X / dx')  # Add x-axis label
    ax.set_ylabel('Y / dy')  # Add y-axis label
    if nx>6:ax.set_xticks(np.arange(0, nx, nx//6))
    if ny>6:ax.set_yticks(np.arange(0, ny, ny//6))
    im = ax.imshow(w.T[::-1], cmap=plt.get_cmap('hot'), vmin=v.min(), vmax=v.max())

    # Create a masked array where the mask is True
    if not mask is None:
        masked_w = np.ma.masked_where(mask(nx,ny)[1:-1,1:-1], w)
        cmap = plt.get_cmap('hot')
        cmap.set_bad(color='cyan')
        # Plot the data with the colormap and masked areas
        im = ax.imshow(masked_w.T[::-1], cmap=cmap, vmin=Tcool, vmax=Thot)
        ax.set_title('{:.1f} iter'.format(nb_iter * dt))
        # Create a colorbar
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        cbar_ax.set_xlabel('$T$ / K', labelpad=20)
        fig.colorbar(im, cax=cbar_ax)
        plt.show()
    else:
        ax.set_title('{:.1f} iter'.format(nb_iter * dt))
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        cbar_ax.set_xlabel('$T$ / K', labelpad=20)
        fig.colorbar(im, cax=cbar_ax)
        plt.show()
    
######################################
## Curvilinear coordinates (sphere) ##
######################################
n1, n2 = nx, ny

R = 2

theta_secu = np.deg2rad(10)
theta_init, theta_fin = theta_secu, np.pi - theta_secu

phi_redu = np.deg2rad(180)

dphi, dtheta = (2*np.pi-phi_redu)/n1, (np.pi - 2*theta_secu)/n2



def e1(i,j): return (R*np.sin(j*dtheta + theta_secu)*dphi)
def e2(i,j): return (R*dtheta)

def k11(i,j): return k0 #k0*i/n1
def k22(i,j): return k0 #k0*j/n2
def k12(i,j): return 0. #k0*i*j/(n1*n2)

#def k11(i,j): return k0*i/n1
#def k22(i,j): return k0*j/n2
#def k12(i,j): return k0*i*j/(n1*n2)

def a11(i,j): return e2(i,j)/e1(i,j)*k11(i,j)
def a22(i,j): return e1(i,j)/e2(i,j)*k22(i,j)
def a12(i,j): return k12(i,j)

A11 = np.array([[a11(i,j) for j in range(n2+1)] for i in range(n1+1)])
A22 = np.array([[a22(i,j) for j in range(n2+1)] for i in range(n1+1)])
A12 = np.array([[a12(i,j) for j in range(n2+1)] for i in range(n1+1)])

E1  = np.array([[e1(i+0.5,j+0.5) for j in range(n2)] for i in range(n1)])
E2  = np.array([[e2(i+0.5,j+0.5) for j in range(n2)] for i in range(n1)])
E1E2 = E1 * E2

def mask1(n1,n2):
    Mask = np.zeros((n1,n2),dtype=bool)
    N = n2
    # Triangle
    for j in range(N//4):
        Mask[N//4 + j, N//2 + j : 3*N//4] = True
    # Square
    for j in range(N//4):
        Mask[N//2 + j, 3*N//8 : 5*N//8] = True
    return fill_edges(Mask, is_mask=True)

def Mx(n1,n2,mask=mask1):
    Mask = mask(n1,n2)
    Inter= np.logical_or(Mask[1:,:], Mask[:-1,:])
    return 0.5 * np.logical_not(Inter)

def My(n1,n2,mask=mask1):
    Mask = mask(n1,n2)
    Inter= np.logical_or(Mask[:,1:], Mask[:,:-1])
    return 0.5 * np.logical_not(Inter)

def Mc(n1,n2,mask=mask1):
    Mask = mask(n1,n2)
    Inter= np.logical_or(np.logical_or(Mask[1:,1:], Mask[1:,:-1]),\
            np.logical_or(Mask[:-1,1:], Mask[:-1,:-1]))
    return np.logical_not(Inter)

def Mp(n1,n2,mask=mask1): # mask push
    Mask = mask(n1,n2)
    #Inter= Mask[1:,1:]
    Inter= np.logical_and(np.logical_and(Mask[1:,1:], Mask[1:,:-1]),\
            np.logical_and(Mask[:-1,1:], Mask[:-1,:-1]))
    return np.logical_not(Inter)

def Mdiru(n1,n2,mask=mask1):
    Mask = mask(n1,n2)
    Inter= np.logical_or(Mask[:-1,:-1], Mask[1:,1:])
    return 0.25 * np.logical_not(Inter) 

def Mdirv(n1,n2,mask=mask1):
    Mask = mask(n1,n2)
    Inter= np.logical_or(Mask[1:,:-1], Mask[:-1,1:])
    return 0.25 * np.logical_not(Inter) 

def previous_step_curv(v0, mask=None, mask_type='ridge center', scheme='symmetric'):
    W0 = v0.copy()
    W0.resize((n1,n2))
    W  = np.zeros((n1+2,n2+2),dtype=chosen_type)
    W[1:-1,1:-1] = W0
    W  = fill_edges(W)
    global mx,my,mc,mdiru,mdirv

    # Rectangle centers
    if mask==None:
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
        else:
            #mc    = Mc(mask)
            diW   = (W[1:,1:] + W[1:,:-1] - W[:-1,1:] - W[:-1,:-1])
            djW   = (W[1:,1:] + W[:-1,1:] - W[1:,:-1] - W[:-1,:-1])
            # Vertex nods
            Qi    = - mc * 0.5 * (A11 * diW + A12 * djW)
            Qj    = - mc * 0.5 * (A12 * diW + A22 * djW)
            # Back to centers
            divQ  = (Qi[1:,1:] + Qi[1:,:-1] - Qi[:-1,1:] - Qi[:-1,:-1])/(2)
            divQ += (Qj[1:,1:] - Qj[1:,:-1] + Qj[:-1,1:] - Qj[:-1,:-1])/(2)
    
    W[1:-1,1:-1] += dt * divQ / E1E2
    
    W0 = W[1:-1,1:-1]
    v  = W0.copy() 
    v.resize(nx*ny)
    return v

def implicit_operator_curv(v0, mask=None, mask_type='ridge center', scheme='symmetric'):
    w = v0.copy()
    global mx,my,mc,mdiru,mdirv
    if mask != None:
        mx    = Mx(mask)
        my    = My(mask)
        mc    = Mc(mask)
        mdiru = Mdiru(mask)
        mdirv = Mdirv(mask)
    for m in range(M):
        previous_step_op = lambda v0:previous_step_curv(v0, mask=mask,\
                                                        mask_type=mask_type,scheme=scheme)
        w = as1.conjgrad(previous_step_op,w,w)
    return w

def W_operator(v0):
    W0 = v0.copy()
    W0.resize((n1,n2))
    W0 = (E1 * E2) * W0
    W0.resize(n1*n2)
    return W0

def WLinv_operator(v0,Linv_op=implicit_operator_curv,mask=None,\
                   mask_type='ridge center', scheme='symmetric'):
    return W_operator(Linv_op(v0,mask,mask_type,scheme))
