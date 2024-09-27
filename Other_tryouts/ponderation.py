#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:02:47 2024

@author: khaddari
"""

import numpy as np
import numpy.linalg as alg

r = np.sqrt(2)/2

b = np.zeros(6)
A = np.zeros((6,6))
b[0] = 1
A[0,0],A[0,1] = 1-r,-r
A[1,2],A[1,3],A[1,4],A[1,5] = -r,-r,1-r,1-r
A[2,0],A[2,1] = 1,1
A[3,2],A[3,3],A[3,4],A[3,5] = 1,1,1,1
A[4,2],A[4,3],A[4,4],A[4,5] = -r,r,1-r,r-1
A[5,0],A[5,1],A[5,2],A[5,3],A[5,4],A[5,5] = (1-r)**2,0.5,0.5,0.5,(1-r)**2,(1-r)**2

def coefs(h):
    b = np.zeros(6)
    A = np.zeros((6,6))
    b[0] = 1/h
    A[0,0],A[0,1] = 1-r,-r
    A[1,2],A[1,3],A[1,4],A[1,5] = -r,-r,1-r,1-r
    A[2,0],A[2,1] = 1,1
    A[3,2],A[3,3],A[3,4],A[3,5] = 1,1,1,1
    A[4,2],A[4,3],A[4,4],A[4,5] = -r,r,1-r,r-1
    A[5,0],A[5,1],A[5,2],A[5,3],A[5,4],A[5,5] = (1-r)**2,0.5,0.5,0.5,(1-r)**2,(1-r)**2
    return alg.solve(A,b)