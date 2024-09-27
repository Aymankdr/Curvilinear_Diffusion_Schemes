import numpy as np
import matplotlib.pyplot as plt

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

## Maillage
dt  = 3e-5
N   = 99
dx  = (borneSUP - borneINF)/(N+1)
cfl = dt/dx**2

xm  = np.linspace(borneINF,borneSUP,N+2)

## Constantes
kappa0 = 1
ecart = 1e-2

## Conditions aux limites (Dirichlet)

def wd(t):
    return 0

def wg(t):
    return 0

## Conditions aux limites (von Neumann)

## Conditions aux limites mixtes

## Condition initiale

def wi(x):
    if mid-ecart<=x<=mid+ecart: return 1
    else: return 0

w_ci       = np.array([wi(borneINF + i*dx) for i in range(N+2)])

## Coefficient de diffusion

def kappa(x):
    return kappa0*(1+0.1*x*(1-x))

kappa_list = np.array([kappa(borneINF + (i+0.5)*dx) for i in range(N+1)])

cfl_max = 0.5/min(kappa_list)

## Schema explicite

def explicite(T,CL='Dirichlet'):
    M = int(T/dt)
    w_1 = list(w_ci)
    w_2 = []
    if CL == 'Dirichlet':
        for n in range(M):
            for i in range(1,N+1):
                s = w_1[i] + dt/dx**2 * (kappa_list[i]*w_1[i+1] - (kappa_list[i] + kappa_list[i-1])*w_1[i] + kappa_list[i-1]*w_1[i-1])
                w_2.append(s)
            w_1 = [wg(n*dt + dt)] + w_2 + [wd(n*dt + dt)]
            w_2 = []
        return w_1
    elif CL == 'Neumann':
        return w_2
    else:return w_ci
    
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
    
 ## Solution exacte fourier pour kappa == kappa0

def we(x,T):
    if T > 0:
        ff = lambda y : np.exp(-(x-y)**2/(4*kappa0*T)) * wi(y)
        return gauss3.integrate(ff,borneINF,borneSUP)/np.sqrt(4*np.pi*kappa0*T)
    else:
        return wi(x)
     
def compare_fe(T):
    qq = explicite(T)
    rr = [we(x,T) for x in xm]
    plt.plot(xm,qq,color='blue',label='explicit')
    plt.plot(xm,rr,color='green',label='fourier')
    plt.show()

    