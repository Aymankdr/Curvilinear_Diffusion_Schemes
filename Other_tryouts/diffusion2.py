#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:35:15 2024

@author: khaddari
"""
import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import quad

## setting

"""          DEBUT QUADRATURE            """
class Quadrature:
    def __init__(self,poids):
        # les poids sont des couples (w_i,x_i) où -1<x_i<1 et sum(w_i) = 2
        self.poids = poids
    def integrate(self,func,a,b,nb=50): # segment
        s   = 0
        x_m = a
        h   = (b-a)/nb
        k   = h/2
        q   = len(self.poids)
        for n in range(nb):
            x_p = x_m + h
            x_c = x_m + k
            for i in range(q):
                s += k*self.poids[i][0]*func(k*self.poids[i][1] + x_c)
            x_m = x_p
        return s

r1s3 = 0.57735026918962576450914878050195745564760175127012656
r3s5 = 0.77459666924148337703585307995647992216658434105831767

droite = Quadrature([(2.0,1)])
gauss2 = Quadrature([(1.0,-r1s3),(1.0,r1s3)])
gauss3 = Quadrature([(5/9,-r3s5),(8/9,0),(5/9,r3s5)])

"""            FIN QUADRATURE             """

## Domaine
borneINF, borneSUP = 0, 1
mid = (borneINF + borneSUP)/2

## Constantes
kappa0 = 4.0 # mm2.s-1 (acier)
kappa_variation_factor = 0
init_shape_factor = 5
init_cond_choice = 1
ecart = 1e-2

## Maillage
dt  = 3e-5
N   = 100
dx  = (borneSUP - borneINF)/N
cfl = dt/dx**2
dt = dx**2 / (2 * kappa0) #kappa max

xmo = np.linspace(borneINF-dx/2,borneSUP+dx/2,N+2)
xm  = xmo[1:-1]



## Conditions aux limites (Dirichlet)

mu = 0.5
eps = 2*mu/(mu + 2*(1 - mu))

def wd(t):
    return 0

def wg(t):
    return 0

## Conditions aux limites (von Neumann)

## Conditions aux limites mixtes

## Condition initiale

def wi(x):
    if init_cond_choice == 0:
        return 1/(1+init_shape_factor*(x-mid)**2)
    else:
        if mid-ecart<=x<=mid+ecart: return 1
        else: return 0

w_ci  = np.array([wi(borneINF + i*dx + 0.5*dx) for i in range(N)])

## Coefficient de diffusion

def kappa(x):
    return kappa0*(1+4*kappa_variation_factor*x*(1-x))

alpha = dt / dx **2 * np.array([ kappa ( i * dx ) for i in range ( N + 1) ])


# Partie implicite
dti = 2*dt
alphai = dti / dx **2 * np.array([ kappa ( i * dx ) for i in range ( N + 1) ])

ctr   = []
ctr.append(1 + eps*alphai[0] + alphai[1])
for i in range(1,N-1):
    ctr.append(1 + alphai[i] + alphai[i+1])
ctr.append(1 + alphai[N-2] + eps*alphai[N-1])

aux = - alphai[1:-1]

## Schema explicite
def explicite(T,CL='Dirichlet'):
    M = int(T/dt)
    w_ci  = np.array([wi(borneINF + i*dx - 0.5*dx) for i in range(N+2)])
    w0 = w_ci
    w  = np.zeros(N+2)
    alpha = dt / dx **2 * np.array([ kappa ( i * dx ) for i in range ( N + 1) ])
    for n in range(M):
        # Nods
        w [1: -1] = (1 - alpha [1:] - alpha [: -1]) * w0 [1: -1]
        w [1: -1] += alpha [1:]* w0 [2:] + alpha [: -1]* w0 [: -2]
        # Edges
        w [0] = (1 - eps ) * w [1]
        w [ -1] = (1 - eps ) * w [ -2]
        # reiterate
        w0 = np.copy(w)
           
    return w0
"""
def explicite(T,CL='Dirichlet'):
    M = int(T/dt)
    w_1 = list(w_ci)
    w_2 = []
    if CL == 'Dirichlet':
        for n in range(M):
            for i in range(1,N+1):
                print(len(w_1)," ",len(w_2))
                s = w_1[i] + (alpha[i]*w_1[i+1] - (alpha[i] + alpha[i-1])*w_1[i] + alpha[i-1]*w_1[i-1])
                w_2.append(s)
            w_1 = [wg(n*dt + dt)] + w_2 + [wd(n*dt + dt)]
            w_2 = []
        return w_1
    elif CL == 'Neumann':
        return w_2
    else:return w_ci
"""

## Schema implicite
# Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type; d is second term.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]

    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

def implicite(T):
    M = int(T/dti)
    w_1 = np.copy(w_ci)
    for n in range(M):
        w_1 = TDMAsolver(aux, ctr, aux, w_1)
    return w_1

## Solution exacte fourier pour kappa == kappa0

def we(x,T):
    if T > 0:
        ff = lambda y : np.exp(-(x-y)**2/(4*kappa0*T)) * wi(y)
        return gauss3.integrate(ff,mid-ecart,mid+ecart)/np.sqrt(4*np.pi*kappa0*T)
    else:
        return wi(x)

def compare_fi(T):
    qq = implicite(T)
    rr = [we(x,T) for x in xm]
    plt.plot(xm,rr,color='green',label='fourier')
    plt.plot(xm,qq,color='blue',label='implicit')
    plt.show()
    
def compare_fe(T):
    qq = explicite(T)[1:-1]
    rr = [we(x,T) for x in xm]
    plt.plot(xm,rr,color='green',label='fourier')
    plt.plot(xm,qq,color='crimson',label='explicit')
    plt.show()
    
def cmprall(T):
    pp = implicite(T)
    qq = explicite(T)[1:-1]
    rr = [we(x,T) for x in xm]
    plt.plot(xm,rr,color='green',label='fourier')
    plt.plot(xm,pp,'--',color='blue',label='implicit')
    plt.plot(xm,qq,'--',color='crimson',label='explicit')
    plt.show()

def compare_ff(t1,t2):
    qq = [we(x,t1) for x in xm]
    rr = [we(x,t2) for x in xm]
    plt.plot(xm,qq,color='blue',label='t1')
    plt.plot(xm,rr,color='green',label='t2')
    plt.show()
