#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:58:28 2024

@author: khaddari
"""
import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import lil_matrix, csr_matrix, eye, diags
from scipy.sparse.linalg import spsolve, eigs, norm
from diffusion7 import colormap
import diffusion7 as df7
from diffusion13 import func2array
from diffusion14 import spherical
import assimilation1 as as1

# Parameters for the oscillating sea floor
L0 = 100  # Sea length
H0 = 100  # Average depth
y0 = 50  # Amplitude of oscillation
lambda0 = 50  # Wavelength of oscillation

# Grid parameters
N1 = 70  # Number of horizontal points
N3 = 70   # Number of vertical layers

# Horizontal positions
x = np.linspace(0, L0, N1)

# Sea floor depth function
def depth(x): return H0 - y0 * np.sin(2 * np.pi / lambda0 * x)
def dz_depth(x): return - y0 * (2 * np.pi / lambda0) * np.cos(2 * np.pi / lambda0 * x)

h = depth(x)

# intervals in x-, z- directions
dx, dz = L0/N1, H0/N3

## Constants
D = 10
dt = 1.0
S = 25
k0 = D**2/((2*S)*dt) # kappa_max

## Boundary Condition
BC = 'periodic'

## Diffusivity
diffusivity='isotropic_and_homogeneous'
#diffusivity='anisotropic_and_inhomogeneous'

## Gradient expression
finite_diff_scheme = 'symmetric'
finite_diff_scheme = 'adaptable'

## Initial conditions
# Amplitudes
Tcool, Thot = 0, 1

# Initial conditions - circle of radius r centred at (cx,cy) (mm)

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
        r2 = r**2
        if p2 < r2:
            return Thot
        else: return Tcool
    else:
        z = x * ang1 + y * ang2
        return np.cos(z) + 1j * np.sin(z)
    
def U0(r,cx,cy,case=CIcase):
    u0 = Tcool * np.ones((N1, N3+1), dtype=chosen_type)
    for i in range(N1):
        for k in range(N3+1):
            u0[i,k] = wi((i+0.5)*dx,(k+0.5)*dz,r,cx,cy)
    u0.resize(N1*(N3+1))
    return u0


######################################
## Curvilinear coordinates (sphere) ##
######################################
N = N1 * (N3 + 1)

## Sphere
R = 2 # radius

theta_secu = np.deg2rad(10) 
theta_init, theta_fin = theta_secu, np.pi - theta_secu

phi_redu = np.deg2rad(0)
dphi, dtheta = (2*np.pi-phi_redu)/N1, (np.pi - 2*theta_secu)/N3

## Geometry
gamma1,gamma3 = 1.0/N1,1.0/N3

def hdp(i): return depth(i*L0*gamma1)
def di_hdp(i): return L0*gamma1*dz_depth(i*L0*gamma1)

def e1(i,k): return (L0/N1) * np.sqrt( 1. + (k/N3)**2 * dz_depth(i*L0/N1)**2 )
def e3(i,k): return depth(i*L0/N1)/N3

# Covariant vectors - Upper index
def b_u1_1(i,k): return 1/(gamma1 * L0)
def b_u1_3(i,k): return 0.0

def b_u3_1(i,k): return -(k * di_hdp(i))/(hdp(i))
def b_u3_3(i,k): return 1/(gamma3 * hdp(i))

# Contravariant vectors - Lower index
def b_l1_1(i,k): return (gamma1 * L0)
def b_l1_3(i,k): return gamma3 * k * di_hdp(i)

def b_l3_1(i,k): return 0.0
def b_l3_3(i,k): return (gamma3 * hdp(i))

# Original diffusivity
if diffusivity=='isotropic_and_homogeneous':
    def k11(i,k): return k0 #k0*i/N1
    def k33(i,k): return k0 #k0*k/N3
    def k13(i,k): return 0. #k0*i*k/(N1*N3)
if diffusivity=='anisotropic_and_inhomogeneous':
    def k11(i,k): return 4*k0*(i-N1//2)**2/N1**2
    def k33(i,k): return 4*k0*(k-N3//2)**2/N3**2
    def k13(i,k): return 8*k0*(i-N1//2)**2*(k-N3//2)**2/(N1*N3)**2
    
# Upper index kappa
def k_u11(i,k): return b_u1_1(i,k) * k11(i,k) * b_l1_1(i,k) +\
                       b_u1_1(i,k) * k13(i,k) * b_l1_3(i,k) +\
                       b_u1_3(i,k) * k13(i,k) * b_l1_1(i,k) +\
                       b_u1_3(i,k) * k33(i,k) * b_l1_3(i,k) 
def k_u13(i,k): return b_u1_1(i,k) * k11(i,k) * b_l3_1(i,k) +\
                       b_u1_1(i,k) * k13(i,k) * b_l3_3(i,k) +\
                       b_u1_3(i,k) * k13(i,k) * b_l3_1(i,k) +\
                       b_u1_3(i,k) * k33(i,k) * b_l3_3(i,k) 
def k_u33(i,k): return b_u3_1(i,k) * k11(i,k) * b_l3_1(i,k) +\
                       b_u3_1(i,k) * k13(i,k) * b_l3_3(i,k) +\
                       b_u3_3(i,k) * k13(i,k) * b_l3_1(i,k) +\
                       b_u3_3(i,k) * k33(i,k) * b_l3_3(i,k) 
                       
# The Metric tensor
def g11(i,k): return (gamma1 * L0)**2 + (gamma3 * k * di_hdp(i))**2
def g13(i,k): return gamma3**2 * k * hdp(i) * di_hdp(i)
def g33(i,k): return (gamma3 * hdp(i))**2

def det_g(i,k): return g11(i,k) * g33(i,k) - g13(i,k)**2
def abs_g(i,k): return np.sqrt(abs(det_g(i,k)))

def g_u11(i,k): return g33(i,k) / det_g(i,k)
def g_u13(i,k): return - g13(i,k) / det_g(i,k)
def g_u33(i,k): return g11(i,k) / det_g(i,k)

# Curvilinear diffusivity
def a11(i,k): return abs_g(i,k) * (k_u11(i,k) * g_u11(i,k) + k_u13(i,k) * g_u13(i,k))
def a13(i,k): return abs_g(i,k) * (k_u11(i,k) * g_u13(i,k) + k_u13(i,k) * g_u33(i,k))
def a33(i,k): return abs_g(i,k) * (k_u13(i,k) * g_u13(i,k) + k_u33(i,k) * g_u33(i,k))

A11 = np.array([[a11(i+0.5,k+0.5) for k in range(N3)] for i in range(N1)])
A33 = np.array([[a33(i+0.5,k+0.5) for k in range(N3)] for i in range(N1)])
A13 = np.array([[a13(i+0.5,k+0.5) for k in range(N3)] for i in range(N1)])

A11r = np.array([[a11(i+0.5,k) for k in range(N3+1)] for i in range(N1)])
A33r = np.array([[a33(i,k+0.5) for k in range(N3)] for i in range(N1+1)])

E1  = np.array([[e1(i,k) for k in range(N3+1)] for i in range(1,N1+1)])
E3  = np.array([[e3(i,k) for k in range(N3+1)] for i in range(1,N1+1)])
E1E3 = np.array([[abs_g(i,k) for k in range(N3+1)] for i in range(1,N1+1)])

## Construct K's coefficients
Aa11 = np.zeros((N1+2,N3+2)); Aa11[1:-1,1:-1] = A11; Aa11[0,:] = Aa11[-2,:]; Aa11[-1,:] = Aa11[1,:]
Aa33 = np.zeros((N1+2,N3+2)); Aa33[1:-1,1:-1] = A33; Aa33[0,:] = Aa33[-2,:]; Aa33[-1,:] = Aa33[1,:]
Aa13 = np.zeros((N1+2,N3+2)); Aa13[1:-1,1:-1] = A13; Aa13[0,:] = Aa13[-2,:]; Aa13[-1,:] = Aa13[1,:]

Aa11r = np.zeros((N1+2,N3+1)); Aa11r[1:-1,:] = A11r;
Aa11r[0,:] = Aa11r[-2,:]; Aa11r[-1,:] = Aa11r[1,:]
Aa33r = np.zeros((N1+1,N3+2)); Aa33r[:,1:-1] = A33r

if finite_diff_scheme == 'symmetric':
    Kcc = Aa11[1:,1:] + 2 * Aa13[1:,1:] + Aa33[1:,1:] +\
        Aa11[:-1,1:] - 2 * Aa13[:-1,1:] + Aa33[:-1,1:] +\
        Aa11[1:,:-1] - 2 * Aa13[1:,:-1] + Aa33[1:,:-1] +\
        Aa11[:-1,:-1] + 2 * Aa13[:-1,:-1] + Aa33[:-1,:-1] 
    Krr = - Aa11[1:,1:] + Aa33[1:,1:] - Aa11[1:,:-1] + Aa33[1:,:-1]
    Kll = - Aa11[:-1,1:] + Aa33[:-1,1:] - Aa11[:-1,:-1] + Aa33[:-1,:-1]
    Kuu = Aa11[1:,1:] - Aa33[1:,1:] + Aa11[:-1,1:] - Aa33[:-1,1:]
    Kdd = Aa11[1:,:-1] - Aa33[1:,:-1] + Aa11[:-1,:-1] - Aa33[:-1,:-1]
    Kru = - Aa11[1:,1:] - Aa33[1:,1:] - 2 * Aa13[1:,1:]
    Klu = - Aa11[:-1,1:] - Aa33[:-1,1:] + 2 * Aa13[:-1,1:]
    Krd = - Aa11[1:,:-1] - Aa33[1:,:-1] + 2 * Aa13[1:,:-1]
    Kld = - Aa11[:-1,:-1] - Aa33[:-1,:-1] - 2 * Aa13[:-1,:-1]
    Kcc*=2;Krr*=2;Kll*=2;Kuu*=2;Kdd*=2;Kru*=2;Klu*=2;Krd*=2;Kld*=2
else:
    Kcc = 8 * (Aa11r[:-1,:] + Aa11r[1:,:] + Aa33r[:,:-1] + Aa33r[:,1:]) +\
        2 * (Aa13[:-1,:-1] - Aa13[:-1,1:]) +\
        2 * (Aa13[1:,1:] - Aa13[1:,:-1]) 
    Krr = - 8 * Aa11r[1:,:]
    Kll = - 8 * Aa11r[:-1,:]
    Kuu = - 8 * Aa33r[:,1:]
    Kdd = - 8 * Aa33r[:,:-1]
    Kru = - 2 * Aa13[1:,1:]
    Klu = 2 * Aa13[:-1,1:]
    Krd = 2 * Aa13[1:,:-1]
    Kld = - 2 * Aa13[:-1,:-1]

## Construct M's coefficients
#Gmet = np.zeros((N1+1,N3+2)); Gmet[1:-1,1:-1] = E1e3
#Gmoy = Gmet[1:,1:]+Gmet[1:,:-1]+Gmet[:-1,1:]+Gmet[:-1,:-1]
Gmet = E1E3; Gmet[1:-1,1:-1] *= 4;
Gmet[1:-1,0] *= 2; Gmet[1:-1,-1] *= 2; Gmet[0,1:-1] *= 2; Gmet[0,1:-1] *= 2 

## Construct K
K = lil_matrix((N,N))
for i in range(N1):
    for k in range(N3+1):
        j = k + i * (N3 + 1) # true index is (i+1,k)
        # Logicals
        ru = (i == N1 - 1 or k == N3); lu = (i == 0 or k == N3)
        rd = (i == N1 - 1 or k == 0); ld = (i == 0 or k == 0)
        rr = (i == N1 - 1); ll = (i == 0)
        uu = (k == N3); dd = (k == 0)
        cc = (0 < i < N1 - 1 and 0 < k < N3)
        
        K[j,j] = Kcc[i+1,k]
        if not uu: K[j,j+1] = Kuu[i+1,k]
        if not dd: K[j,j-1] = Kdd[i+1,k]
        if not rr: K[j,j+N3+1] = Krr[i+1,k]
        else: K[j,j+N3+1 - N] = Krr[i+1,k]
        if not ll: K[j,j-N3-1] = Kll[i+1,k]
        else: K[j,j-N3-1 + N] = Kll[i+1,k]
        if not rr and not ru: K[j,j+N3+2] = Kru[i+1,k]
        elif rr and (j!=N3): K[j,j+N3+2 - N] = Kru[i+1,k]
        if not ll and not ld: K[j,j-N3-2] = Kld[i+1,k]
        elif ll and (j!=0): K[j,j-N3-2 + N] = Kld[i+1,k]
        if not rr and not rd: K[j,j+N3] = Krd[i+1,k]
        elif rr and (j!=0): K[j,j+N3 - N] = Krd[i+1,k]
        if not ll and not lu: K[j,j-N3] = Klu[i+1,k]
        elif ll and (j!=N3): K[j,j-N3 + N] = Klu[i+1,k]
        
K = K.tocsr()
        
## Construct M
Gmet *= 2
M = diags(Gmet.flatten())
Mdir = Gmet.flatten()
Minv = 1/Mdir

## Construct A=M+K
A = M + K

## Energy scheme
def energy_iteration(u):
    v = M.dot(u)
    b = spsolve(A,v)
    #b = as1.conkgrad(A, v, v, nature=0)
    return b

def energy(u,up):
    return (M @ u) @ up - 0.5 * (M @ u) @ u - 0.5 * (K @ u) @ u

def energy_scheme(u,nb_iter=S):
    E_list = []
    v = np.copy(u)
    for s in range(nb_iter):
        w = energy_iteration(v)
        e = energy(w, v)
        E_list.append(e)
        v = w
    return v,E_list


## Algorithm
def algo(nb_iter=S,x_cursor=0.5,z_cursor=0.5,u0=None):
    r, cx, cy = 0.2*min(L0,H0), x_cursor*L0, z_cursor*H0
    if type(u0)!=type(np.zeros(2)): u0 = U0(r,cx,cy)
    vv,E_list = energy_scheme(u0,nb_iter=nb_iter)
    print(vv.dot(vv))
    colormap(vv[::-1],nx=N1,ny=N3+1,nb_iter=nb_iter) 
    plt.show()
    s_list = list(range(nb_iter))
    #plt.plot(s_list,E_list)
    #plt.show()
    return vv, E_list
    
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

# Dirac
def Dirac(i,k):
    v = np.zeros(N)
    v[k + (N3+1) * i] = 1 / (abs_g(i+0.5,k+0.5))
    return v





def plot_with_dots(vv):
    # Create temperature distribution T_ik
    # Here, we simulate a simple model where temperature decreases with depth
    # and has some variation in x-direction.
    T = np.zeros((N1, N3+1))
    for i in range(N1):
        for k in range(N3+1):
            # Simulate some temperature profile (for demonstration)
            # Temperature decreases linearly with depth and has sinusoidal variation with x
            T[i, k] = vv[k + (N3+1) * i]
    Tmin, Tmax = T.min(), T.max()
    
    # Vertical layers, scaled by the depth function
    z_layers = np.zeros((N1, N3+1))
    for i in range(N1):
        z_layers[i] = np.linspace(0, h[i], N3+1)

    # Plotting
    plt.figure(figsize=(13, 8))

    # Plot each column with its associated depth and temperature
    for i in range(N1):
        plt.scatter(
            #np.full(N3+1, x[i]), z_layers[i], c=T[i], cmap='coolwarm', s=10
            np.full(N3+1, x[i]), z_layers[i], c=T[i], cmap='coolwarm', s=10, vmin=Tmin, vmax=Tmax
        )
    
    # Plot the sea floor
    plt.plot(x, h, color='black', label='Sea Floor', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Horizontal Distance ($z_1$)')
    plt.ylabel('Depth ($z_3$)')
    plt.title('Diffused State Perturbation Cross Section of the Ocean')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Perturbation Quantity')
    plt.grid()
    plt.show()
