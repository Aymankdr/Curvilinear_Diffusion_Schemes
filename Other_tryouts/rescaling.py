#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 17:52:31 2024

@author: khaddari
"""

import numpy as np
from scipy.ndimage import zoom

def rescale_mask(mask, Nx2, Ny2):
    """
    Rescale a binary mask to new dimensions using bilinear interpolation.
    
    Parameters:
    mask (2D numpy array): Original binary mask of size (Nx1, Ny1).
    Nx2 (int): Desired number of rows in the rescaled mask.
    Ny2 (int): Desired number of columns in the rescaled mask.
    
    Returns:
    2D numpy array: Rescaled binary mask of size (Nx2, Ny2).
    """
    # Calculate the zoom factors for each dimension
    zoom_factor_x = Nx2 / mask.shape[0]
    zoom_factor_y = Ny2 / mask.shape[1]
    
    # Use scipy's zoom function with bilinear interpolation (order=1)
    rescaled_mask = zoom(mask, (zoom_factor_x, zoom_factor_y), order=1)
    
    # Since the mask should be binary, threshold the interpolated values
    rescaled_mask = (rescaled_mask > 0.5).astype(np.uint8)
    
    return rescaled_mask

# Example usage:
Nx1, Ny1 = 10, 10  # Original mask dimensions
Lx, Ly = 100, 100  # Rectangle dimensions
Nx2, Ny2 = 40, 40  # New desired mask dimensions

# Create an example mask
mask = np.random.randint(0, 2, size=(Nx1, Ny1))

# Rescale the mask
rescaled_mask = rescale_mask(mask, Nx2, Ny2)

print("Original Mask:")
print(mask)
print("Rescaled Mask:")
print(rescaled_mask)
