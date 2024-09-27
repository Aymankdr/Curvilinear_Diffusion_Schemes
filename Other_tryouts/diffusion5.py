#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:35:32 2024

@author: khaddari
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import quad


## Domaine
borneINF, borneSUP = 0, 1
mid = (borneINF + borneSUP)/2

## Constantes
D = 0.05
M = 50
dt = 1.0
kappa0 = D**2/((2*M)*dt) # kappa_max


kappa_variation_factor = 0
init_shape_factor = 5
init_cond_choice = 1
ecart = 1e-2

## Maillage
dx  = 2*np.sqrt(kappa0*dt) # >= np.sqrt(2*kappa0*dt)
N   = int((borneSUP - borneINF)/dx)+1
dx  = (borneSUP - borneINF)/N
cfl = dt/dx**2

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

def w_ci(j):
    wl = np.zeros(N)
    wl[j] = 1.0
    return wl

## Coefficient de diffusion

def kappa(x):
    return kappa0*(1-4*kappa_variation_factor*x*(1-x))

def kappai(x):
    return D**2/((2*M-3)*dt)

alpha = dt / dx **2 * np.array([ kappa ( i * dx ) for i in range ( N + 1) ])


## Schema explicite
def explicite(w_i,CL='Robin'):
    w0 = w_i
    w  = np.zeros(N)
    for n in range(M):
        # Central diagonal
        w  = (1 - alpha [1:] - alpha [: -1]) * w0 
        # Lower diagonal

        w[1:]  += alpha[1:-1]* w0[:-1]
        # Upper diagonal
        w[:-1] += alpha[1:-1]* w0[1:]

        # Edges
        w [0 ] += (1 - eps ) * alpha[0 ] * w0[0 ]
        w [-1] += (1 - eps ) * alpha[-1] * w0[-1]
        # reiterate
        w0 = np.copy(w)
           
    return w0

# Partie implicite
dti = 2*dt
alphai = dti / dx **2 * np.array([ kappai( i * dx ) for i in range ( N + 1) ])

ctr   = []
ctr.append(1 + eps*alphai[0] + alphai[1])
for i in range(1,N-1):
    ctr.append(1 + alphai[i] + alphai[i+1])
ctr.append(1 + alphai[N-2] + eps*alphai[N-1])

aux = - alphai[1:-1]

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

def implicite(w_i):
    w_1 = np.copy(w_i)
    for n in range(M):
        w_1 = TDMAsolver(aux, ctr, aux, w_1)
    return w_1

## OPERATEURS
def test_autoadj(operator=explicite):
    v=np.random.normal(size=N) 
    u=np.random.normal(size=N)
    return operator(u).dot(v) - operator(v).dot(u)

def assemble(idx=None,operator=explicite):
    if idx==None:
        L   = []
        cal = []
        C   = []
        for j in range(N):
            s = operator(w_ci(j))#[1:-1]
            L.append(s)
            cal.append(1/np.sqrt(s[j]))
            C.append(s/s[j])
            #plt.plot(xm,s)
        cal2= np.diag(cal)
        L   = np.array(L)
        C2  = cal2.dot(L).dot(cal2)
        return np.array(L),np.array(cal),C2
    else:
        s = operator(w_ci(idx))
        plt.plot(xm,s)
        
def graphs(i):
    II = assemble(operator=implicite)[2]
    EE = assemble(operator=explicite)[2]
    AA = abs(EE - II)
    plt.plot(xm,EE[i],color='blue')
    plt.plot(xm,II[i],color='green')
    plt.plot(xm,AA[i],'--',color='crimson')
    print("The average gap: ",sum(AA[i])/N)