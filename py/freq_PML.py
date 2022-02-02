'''
Created on May 22, 2021
@author: Min Basnet

@Description:
Calculates PML layer in a row for finite difference grid.

@Literature:

[1]
Chen, J. B., & Cao, J. (2016). Modeling of frequency-domain elastic-wave equation with an average-derivative optimal method. Geophysics, 81(6), T339-T356.

'''

import math
import numpy as np

class freq_PML(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''



    def pml_PSV(self, npml0, npml1, ngrid, npad, a0_pml, omega_pml, pml_xi, pml_xi_half):
        '''
        Compute PML variables
        '''
        
        for i_grid in range(0, npml0):
            pml_xi[i_grid+npad] = 1.0 - (0.0+1.0j)*a0_pml*math.cos(0.5*math.pi*(i_grid+1)/npml0)/omega_pml
            pml_xi_half[i_grid+npad] = 1.0 - (0.0+1.0j)*a0_pml*math.cos(0.5*math.pi*(i_grid+1.5)/npml0)/omega_pml
        
        for i_grid in range(0, npml1):
            pml_xi[ngrid-i_grid-npad-1] = 1.0 - (0.0+1.0j)*a0_pml*math.cos(0.5*math.pi*(i_grid+1)/npml0)/omega_pml
            pml_xi_half[ngrid-i_grid-npad-1] = 1.0 - (0.0+1.0j)*a0_pml*math.cos(0.5*math.pi*(i_grid+0.5)/npml0)/omega_pml
            
