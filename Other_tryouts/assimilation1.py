#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:20:55 2024

@author: khaddari
"""
import numpy as np

def conjgrad(A, b, x, nature='isomorphism'):
    """
    A function to solve [A]{x} = {b} linear equation system with the 
    conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A : matrix 
        A real autoadjoint positive definite isomorphism.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    """  
    if nature=='isomorphism': r = b - A(x)
    else: r = b - A.dot(x)
    p = r
    rsold = np.dot(np.transpose(r), r)
    
    #s = 0
    
    for i in range(len(b)):
        if nature=='isomorphism': Ap = A(p)
        else: Ap = A.dot(p)
        alpha = rsold / np.dot(p, Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < 1e-8:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return x

def blue(xb,y,B,R,H): 
    # assuming B,R,H are matrices 
    # H is a jacobian, then it is possible to transpose it
    v = y - H.dot(xb)
    A = lambda v : R.dot(v) + H.dot(B).dot(v.dot(H))
    z = conjgrad(A, v, v)
    return xb + B.dot(z.dot(H))

def autoadj(A, b, nature='homomorphism', dimE = None, dimF = None, nb_sample=50):
    """
    Monte Carlo Method 
    """
    n = len(b)
    if dimF != None:
        n = dimF
    if dimE == None: 
        m = n
    else:
        m = dimE
    s = np.zeros((m,n))
    for l in range(nb_sample):
        v  = np.random.normal(0,1,m)
        w  = A(v)
        #A_ = np.outer(v,w)
        s += np.outer(v,w)
    s = s/nb_sample
    return s
