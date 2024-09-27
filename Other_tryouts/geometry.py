#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:56:46 2024

@author: khaddari
"""

import numpy as np
from data import n1, n2, theta_secu, R

######################################
## Curvilinear coordinates (sphere) ##
######################################

theta_init, theta_fin = theta_secu, np.pi - theta_secu


dphi, dtheta = (2*np.pi)/n1, (np.pi - 2*theta_secu)/n2


def e1(i,j): return (R*np.sin(j*dtheta + theta_secu)*dphi)
def e2(i,j): return (R*dtheta)