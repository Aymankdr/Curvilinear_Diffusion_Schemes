#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:58:30 2024

@author: khaddari
"""

from geometry import e1, e2
from data import D

## Daley's tensor

def d11(i,j): return D 
def d22(i,j): return D
def d12(i,j): return 0. 


## Curvilinear Daley's tensor
def a11(i,j): return e2(i,j)/e1(i,j)*d11(i,j)
def a22(i,j): return e1(i,j)/e2(i,j)*d22(i,j)
def a12(i,j): return d12(i,j)


