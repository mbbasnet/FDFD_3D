'''
Created on May 22, 2021
@author: Min Basnet

@Description:
Frequency domain finite difference method for PSV  wave propagation model.

@Literature:

[1]
Chen, J. B., & Cao, J. (2016). Modeling of frequency-domain elastic-wave equation with an average-derivative optimal method. Geophysics, 81(6), T339-T356.

'''

import math
import numpy as np
from scipy.sparse import csr_matrix 
from scipy.sparse.lil import lil_matrix

class freq_3D(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''



    def stencil_3D(self, ix, iy, iz, grid_dx, grid_dy, grid_dz, pml_x, pml_x_half, pml_y, pml_y_half, pml_z, pml_z_half,\
        med_rho, med_lam, med_mu, med_eta, omega, opt_alpha1, opt_alpha2, opt_beta1, opt_beta2, opt_gamma1, opt_gamma2, opt_c, opt_d):
        
        '''
        Calculates the  Stiffness parameters for Local Stencils \
        based on Average-derivative optimal method \
        for a nine point stencil centered at [ix][iz]
        '''

        # Taking varaiables from Literature [1]
        # optimization factors

        c = opt_c
        d = opt_d
        e = 0.25*(1 - c - 4.0*d)
        f = 0.25*(1 - c - 4.0*d)
        
        alpha1 = opt_alpha1
        alpha2 = opt_alpha2
        alpha3 = (1.0 - 4.0 * alpha1 - 4.0 * alpha2)
        
        beta1 = opt_beta1
        beta2 = opt_beta2
        beta3 = (1.0 - 4.0 * beta1 - 4.0 * beta2)
        
        gamma1 = opt_gamma1
        gamma2 = opt_gamma2
        gamma3 = (1.0 - 4.0 * gamma1 - 4.0 * gamma2)
        
        fxx = 1.0 / ( grid_dx * grid_dx * pml_x[ix] )
        fyy = 1.0 / ( grid_dy * grid_dy * pml_y[iy] )
        fzz = 1.0 / ( grid_dz * grid_dz * pml_z[iz] )
        
        omg = med_rho[ix][iy][iz] * omega * omega
        
        Ubar_p1 = fxx * 0.5* (med_eta[ix][iy][iz] + med_eta[ix+1][iy][iz])/pml_x_half[ix]
        Ubar_m1 = fxx * 0.5* (med_eta[ix][iy][iz] + med_eta[ix-1][iy][iz])/pml_x_half[ix-1]
        Ubar = -(Ubar_p1 + Ubar_m1)
        
        Uhat_p1 = fyy * 0.5* (med_mu[ix][iy][iz] + med_mu[ix][iy+1][iz])/pml_y_half[iy]
        Uhat_m1 = fyy * 0.5* (med_mu[ix][iy][iz] + med_mu[ix][iy-1][iz])/pml_y_half[iy-1]
        Uhat = -(Uhat_p1 + Uhat_m1)
        
        Utilde_p1 = fzz * 0.5* (med_mu[ix][iy][iz] + med_mu[ix][iy][iz+1])/pml_z_half[iz]
        Utilde_m1 = fzz * 0.5* (med_mu[ix][iy][iz] + med_mu[ix][iy][iz-1])/pml_z_half[iz-1]
        Utilde = -(Utilde_p1 + Utilde_m1)
        
        
        
              
        UU000 = alpha2 * Ubar_m1 + beta2 * Uhat_m1 + gamma2 * Utilde_m1 + omg * f
        UU001 = alpha1 * Ubar_m1 + beta1 * Uhat_m1 + gamma2 * Utilde    + omg * e
        UU002 = alpha2 * Ubar_m1 + beta2 * Uhat_m1 + gamma2 * Utilde_p1 + omg * f
        
        UU010 = alpha1 * Ubar_m1 + beta2 * Uhat + gamma1 * Utilde_m1 + omg * e
        UU011 = alpha3 * Ubar_m1 + beta1 * Uhat + gamma1 * Utilde    + omg * d
        UU012 = alpha1 * Ubar_m1 + beta2 * Uhat + gamma1 * Utilde_p1 + omg * e
        
        UU020 = alpha2 * Ubar_m1 + beta2 * Uhat_p1 + gamma2 * Utilde_m1 + omg * f
        UU021 = alpha1 * Ubar_m1 + beta1 * Uhat_p1 + gamma2 * Utilde    + omg * e
        UU022 = alpha2 * Ubar_m1 + beta2 * Uhat_p1 + gamma2 * Utilde_p1 + omg * f
        
        UU100 = alpha2 * Ubar + beta1 * Uhat_m1 + gamma1 * Utilde_m1 + omg * e
        UU101 = alpha1 * Ubar + beta3 * Uhat_m1 + gamma1 * Utilde    + omg * d
        UU102 = alpha2 * Ubar + beta1 * Uhat_m1 + gamma1 * Utilde_p1 + omg * e
        
        UU110 = alpha1 * Ubar + beta1 * Uhat + gamma2 * Utilde_m1 + omg * d
        UU111 = alpha3 * Ubar + beta3 * Uhat + gamma3 * Utilde    + omg * c
        UU112 = alpha1 * Ubar + beta1 * Uhat + gamma3 * Utilde_p1 + omg * d
        
        UU120 = alpha2 * Ubar + beta1 * Uhat_p1 + gamma1 * Utilde_m1 + omg * e
        UU121 = alpha1 * Ubar + beta3 * Uhat_p1 + gamma1 * Utilde    + omg * d
        UU122 = alpha2 * Ubar + beta1 * Uhat_p1 + gamma1 * Utilde_p1 + omg * e
        
        UU200 = alpha2 * Ubar_p1 + beta2 * Uhat_m1 + gamma2 * Utilde_m1 + omg * f
        UU201 = alpha1 * Ubar_p1 + beta1 * Uhat_m1 + gamma2 * Utilde    + omg * e
        UU202 = alpha2 * Ubar_p1 + beta2 * Uhat_m1 + gamma2 * Utilde_p1 + omg * f
        
        UU210 = alpha1 * Ubar_p1 + beta2 * Uhat + gamma1 * Utilde_m1 + omg * e
        UU211 = alpha3 * Ubar_p1 + beta1 * Uhat + gamma1 * Utilde    + omg * d
        UU212 = alpha1 * Ubar_p1 + beta2 * Uhat + gamma1 * Utilde_p1 + omg * e
        
        UU220 = alpha2 * Ubar_p1 + beta2 * Uhat_p1 + gamma2 * Utilde_m1 + omg * f
        UU221 = alpha1 * Ubar_p1 + beta1 * Uhat_p1 + gamma2 * Utilde    + omg * e
        UU222 = alpha2 * Ubar_p1 + beta2 * Uhat_p1 + gamma2 * Utilde_p1 + omg * f
        
        sten27_Kuu = np.array([UU000, UU001, UU002, UU010, UU011, UU012, UU020, UU021, UU022, \
                      UU100, UU101, UU102, UU110, UU111, UU112, UU120, UU121, UU122,\
                      UU200, UU201, UU202, UU210, UU211, UU212, UU220, UU221, UU222])
        
        
        lam_uv_p1 = (1.0/(4.0*grid_dx*grid_dy*pml_x[ix]))*(med_lam[ix+1][iy][iz]/pml_y[iy])
        lam_uv_m1 = (1.0/(4.0*grid_dx*grid_dy*pml_x[ix]))*(med_lam[ix-1][iy][iz]/pml_y[iy])
        
        mu_uv_p1 = (1.0/(4.0*grid_dx*grid_dy*pml_y[iy]))*(med_mu[ix][iy+1][iz]/pml_x[ix])
        mu_uv_m1 = (1.0/(4.0*grid_dx*grid_dy*pml_y[iy]))*(med_mu[ix][iy-1][iz]/pml_x[ix])
        
        UV00 = lam_uv_m1 + mu_uv_m1
        UV01 = -lam_uv_m1 - mu_uv_p1
        UV10 = -lam_uv_p1 - mu_uv_m1
        UV11 = lam_uv_p1 + mu_uv_p1
        
        sten27_Kuv = np.array([UV00, UV01, UV10, UV11])
        
        
        lam_uw_p1 = (1.0/(4.0*grid_dx*grid_dz*pml_x[ix]))*(med_lam[ix+1][iy][iz]/pml_z[iz])
        lam_uw_m1 = (1.0/(4.0*grid_dx*grid_dz*pml_x[ix]))*(med_lam[ix-1][iy][iz]/pml_z[iz])
        
        mu_uw_p1 = (1.0/(4.0*grid_dx*grid_dz*pml_z[iy]))*(med_mu[ix][iy][iz+1]/pml_x[ix])
        mu_uw_m1 = (1.0/(4.0*grid_dx*grid_dz*pml_z[iy]))*(med_mu[ix][iy][iz-1]/pml_x[ix])
        
        UW00 = lam_uw_m1 + mu_uw_m1
        UW01 = -lam_uw_m1 - mu_uw_p1
        UW10 = -lam_uw_p1 - mu_uw_m1
        UW11 = lam_uw_p1 + mu_uw_p1
        
        sten27_Kuw = np.array([UW00, UW01, UW10, UW11])
        
        
        Vbar_p1 = fxx * 0.5* (med_mu[ix][iy][iz] + med_mu[ix+1][iy][iz])/pml_x_half[ix]
        Vbar_m1 = fxx * 0.5* (med_mu[ix][iy][iz] + med_mu[ix-1][iy][iz])/pml_x_half[ix-1]
        Vbar = -(Vbar_p1 + Vbar_m1)
        
        Vhat_p1 = fyy * 0.5* (med_eta[ix][iy][iz] + med_eta[ix][iy+1][iz])/pml_y_half[iy]
        Vhat_m1 = fyy * 0.5* (med_eta[ix][iy][iz] + med_eta[ix][iy-1][iz])/pml_y_half[iy-1]
        Vhat = -(Vhat_p1 + Vhat_m1)
        
        Vtilde_p1 = fzz * 0.5* (med_mu[ix][iy][iz] + med_mu[ix][iy][iz+1])/pml_z_half[iz]
        Vtilde_m1 = fzz * 0.5* (med_mu[ix][iy][iz] + med_mu[ix][iy][iz-1])/pml_z_half[iz-1]
        Vtilde = -(Vtilde_p1 + Vtilde_m1)
        
        
        VV000 = alpha2 * Vbar_m1 + beta2 * Vhat_m1 + gamma2 * Vtilde_m1 + omg * f
        VV001 = alpha1 * Vbar_m1 + beta1 * Vhat_m1 + gamma2 * Vtilde    + omg * e
        VV002 = alpha2 * Vbar_m1 + beta2 * Vhat_m1 + gamma2 * Vtilde_p1 + omg * f
        
        VV010 = alpha1 * Vbar_m1 + beta2 * Vhat + gamma1 * Vtilde_m1 + omg * e
        VV011 = alpha3 * Vbar_m1 + beta1 * Vhat + gamma1 * Vtilde    + omg * d
        VV012 = alpha1 * Vbar_m1 + beta2 * Vhat + gamma1 * Vtilde_p1 + omg * e
        
        VV020 = alpha2 * Vbar_m1 + beta2 * Vhat_p1 + gamma2 * Vtilde_m1 + omg * f
        VV021 = alpha1 * Vbar_m1 + beta1 * Vhat_p1 + gamma2 * Vtilde    + omg * e
        VV022 = alpha2 * Vbar_m1 + beta2 * Vhat_p1 + gamma2 * Vtilde_p1 + omg * f
        
        VV100 = alpha2 * Vbar + beta1 * Vhat_m1 + gamma1 * Vtilde_m1 + omg * e
        VV101 = alpha1 * Vbar + beta3 * Vhat_m1 + gamma1 * Vtilde    + omg * d
        VV102 = alpha2 * Vbar + beta1 * Vhat_m1 + gamma1 * Vtilde_p1 + omg * e
        
        VV110 = alpha1 * Vbar + beta1 * Vhat + gamma2 * Vtilde_m1 + omg * d
        VV111 = alpha3 * Vbar + beta3 * Vhat + gamma3 * Vtilde    + omg * c
        VV112 = alpha1 * Vbar + beta1 * Vhat + gamma3 * Vtilde_p1 + omg * d
        
        VV120 = alpha2 * Vbar + beta1 * Vhat_p1 + gamma1 * Vtilde_m1 + omg * e
        VV121 = alpha1 * Vbar + beta3 * Vhat_p1 + gamma1 * Vtilde    + omg * d
        VV122 = alpha2 * Vbar + beta1 * Vhat_p1 + gamma1 * Vtilde_p1 + omg * e
        
        VV200 = alpha2 * Vbar_p1 + beta2 * Vhat_m1 + gamma2 * Vtilde_m1 + omg * f
        VV201 = alpha1 * Vbar_p1 + beta1 * Vhat_m1 + gamma2 * Vtilde    + omg * e
        VV202 = alpha2 * Vbar_p1 + beta2 * Vhat_m1 + gamma2 * Vtilde_p1 + omg * f
        
        VV210 = alpha1 * Vbar_p1 + beta2 * Vhat + gamma1 * Vtilde_m1 + omg * e
        VV211 = alpha3 * Vbar_p1 + beta1 * Vhat + gamma1 * Vtilde    + omg * d
        VV212 = alpha1 * Vbar_p1 + beta2 * Vhat + gamma1 * Vtilde_p1 + omg * e
        
        VV220 = alpha2 * Vbar_p1 + beta2 * Vhat_p1 + gamma2 * Vtilde_m1 + omg * f
        VV221 = alpha1 * Vbar_p1 + beta1 * Vhat_p1 + gamma2 * Vtilde    + omg * e
        VV222 = alpha2 * Vbar_p1 + beta2 * Vhat_p1 + gamma2 * Vtilde_p1 + omg * f
        
        sten27_Kvv = np.array([VV000, VV001, VV002, VV010, VV011, VV012, VV020, VV021, VV022, \
                      VV100, VV101, VV102, VV110, VV111, VV112, VV120, VV121, VV122,\
                      VV200, VV201, VV202, VV210, VV211, VV212, VV220, VV221, VV222])
        
        
        lam_vu_p1 = (1.0/(4.0*grid_dx*grid_dy*pml_y[iy]))*(med_lam[ix][iy+1][iz]/pml_x[ix])
        lam_vu_m1 = (1.0/(4.0*grid_dx*grid_dy*pml_y[iy]))*(med_lam[ix][iy-1][iz]/pml_x[ix])
        
        mu_vu_p1 = (1.0/(4.0*grid_dx*grid_dy*pml_x[ix]))*(med_mu[ix+1][iy][iz]/pml_y[iy])
        mu_vu_m1 = (1.0/(4.0*grid_dx*grid_dy*pml_x[ix]))*(med_mu[ix-1][iy][iz]/pml_y[iy])
        
        VU00 = lam_vu_m1 + mu_vu_m1
        VU01 = -lam_vu_m1 - mu_vu_p1
        VU10 = -lam_vu_p1 - mu_vu_m1
        VU11 = lam_vu_p1 + mu_vu_p1
        
        sten27_Kvu = np.array([VU00, VU01, VU10, VU11])
        
        lam_vw_p1 = (1.0/(4.0*grid_dx*grid_dz*pml_y[iy]))*(med_lam[ix][iy+1][iz]/pml_z[iz])
        lam_vw_m1 = (1.0/(4.0*grid_dx*grid_dz*pml_y[iy]))*(med_lam[ix][iy+1][iz]/pml_z[iz])
        
        mu_vw_p1 = (1.0/(4.0*grid_dx*grid_dz*pml_z[iy]))*(med_mu[ix][iy][iz+1]/pml_y[iy])
        mu_vw_m1 = (1.0/(4.0*grid_dx*grid_dz*pml_z[iy]))*(med_mu[ix][iy][iz-1]/pml_y[iy])
       
        
        VW00 = lam_vw_m1 + mu_vw_m1
        VW01 = -lam_vw_m1 - mu_vw_p1
        VW10 = -lam_vw_p1 - mu_vw_m1
        VW11 = lam_vw_p1 + mu_vw_p1
        
        sten27_Kvw = np.array([VW00, VW01, VW10, VW11])
        
       
        Wbar_p1 = fxx * 0.5* (med_mu[ix][iy][iz] + med_mu[ix+1][iy][iz])/pml_x_half[ix]
        Wbar_m1 = fxx * 0.5* (med_mu[ix][iy][iz] + med_mu[ix-1][iy][iz])/pml_x_half[ix-1]
        Wbar = -(Wbar_p1 + Wbar_m1)
        
        What_p1 = fyy * 0.5* (med_mu[ix][iy][iz] + med_mu[ix][iy+1][iz])/pml_y_half[iy]
        What_m1 = fyy * 0.5* (med_mu[ix][iy][iz] + med_mu[ix][iy-1][iz])/pml_y_half[iy-1]
        What = -(What_p1 + What_m1)
        
        Wtilde_p1 = fzz * 0.5* (med_eta[ix][iy][iz] + med_eta[ix][iy][iz+1])/pml_z_half[iz]
        Wtilde_m1 = fzz * 0.5* (med_eta[ix][iy][iz] + med_eta[ix][iy][iz-1])/pml_z_half[iz-1]
        Wtilde = -(Wtilde_p1 + Wtilde_m1)
       
        WW000 = alpha2 * Wbar_m1 + beta2 * What_m1 + gamma2 * Wtilde_m1 + omg * f
        WW001 = alpha1 * Wbar_m1 + beta1 * What_m1 + gamma2 * Wtilde    + omg * e
        WW002 = alpha2 * Wbar_m1 + beta2 * What_m1 + gamma2 * Wtilde_p1 + omg * f
        
        WW010 = alpha1 * Wbar_m1 + beta2 * What + gamma1 * Wtilde_m1 + omg * e
        WW011 = alpha3 * Wbar_m1 + beta1 * What + gamma1 * Wtilde    + omg * d
        WW012 = alpha1 * Wbar_m1 + beta2 * What + gamma1 * Wtilde_p1 + omg * e
        
        WW020 = alpha2 * Wbar_m1 + beta2 * What_p1 + gamma2 * Wtilde_m1 + omg * f
        WW021 = alpha1 * Wbar_m1 + beta1 * What_p1 + gamma2 * Wtilde    + omg * e
        WW022 = alpha2 * Wbar_m1 + beta2 * What_p1 + gamma2 * Wtilde_p1 + omg * f
        
        WW100 = alpha2 * Wbar + beta1 * What_m1 + gamma1 * Wtilde_m1 + omg * e
        WW101 = alpha1 * Wbar + beta3 * What_m1 + gamma1 * Wtilde    + omg * d
        WW102 = alpha2 * Wbar + beta1 * What_m1 + gamma1 * Wtilde_p1 + omg * e
        
        WW110 = alpha1 * Wbar + beta1 * What + gamma2 * Wtilde_m1 + omg * d
        WW111 = alpha3 * Wbar + beta3 * What + gamma3 * Wtilde    + omg * c
        WW112 = alpha1 * Wbar + beta1 * What + gamma3 * Wtilde_p1 + omg * d
        
        WW120 = alpha2 * Wbar + beta1 * What_p1 + gamma1 * Wtilde_m1 + omg * e
        WW121 = alpha1 * Wbar + beta3 * What_p1 + gamma1 * Wtilde    + omg * d
        WW122 = alpha2 * Wbar + beta1 * What_p1 + gamma1 * Wtilde_p1 + omg * e
        
        WW200 = alpha2 * Wbar_p1 + beta2 * What_m1 + gamma2 * Wtilde_m1 + omg * f
        WW201 = alpha1 * Wbar_p1 + beta1 * What_m1 + gamma2 * Wtilde    + omg * e
        WW202 = alpha2 * Wbar_p1 + beta2 * What_m1 + gamma2 * Wtilde_p1 + omg * f
        
        WW210 = alpha1 * Wbar_p1 + beta2 * What + gamma1 * Wtilde_m1 + omg * e
        WW211 = alpha3 * Wbar_p1 + beta1 * What + gamma1 * Wtilde    + omg * d
        WW212 = alpha1 * Wbar_p1 + beta2 * What + gamma1 * Wtilde_p1 + omg * e
        
        WW220 = alpha2 * Wbar_p1 + beta2 * What_p1 + gamma2 * Wtilde_m1 + omg * f
        WW221 = alpha1 * Wbar_p1 + beta1 * What_p1 + gamma2 * Wtilde    + omg * e
        WW222 = alpha2 * Wbar_p1 + beta2 * What_p1 + gamma2 * Wtilde_p1 + omg * f
        
        sten27_Kww = np.array([WW000, WW001, WW002, WW010, WW011, WW012, WW020, WW021, WW022, \
                      WW100, WW101, WW102, WW110, WW111, WW112, WW120, WW121, WW122,\
                      WW200, WW201, WW202, WW210, WW211, WW212, WW220, WW221, WW222])
        

        lam_wu_p1 = (1.0/(4.0*grid_dx*grid_dz*pml_z[iy]))*(med_lam[ix][iy][iz+1]/pml_x[ix])
        lam_wu_m1 = (1.0/(4.0*grid_dx*grid_dz*pml_z[iy]))*(med_lam[ix][iy][iz-1]/pml_x[ix])
        
        mu_wu_p1 = (1.0/(4.0*grid_dx*grid_dz*pml_x[ix]))*(med_mu[ix+1][iy][iz]/pml_z[iz])
        mu_wu_m1 = (1.0/(4.0*grid_dx*grid_dz*pml_x[ix]))*(med_mu[ix-1][iy][iz]/pml_z[iz])
        
        WU00 = lam_wu_m1 + mu_wu_m1
        WU01 = -lam_wu_m1 - mu_wu_p1
        WU10 = -lam_wu_p1 - mu_wu_m1
        WU11 = lam_wu_p1 + mu_wu_p1
        
        sten27_Kwu = np.array([WU00, WU01, WU10, WU11])
        
        lam_wv_p1 = (1.0/(4.0*grid_dz*grid_dy*pml_z[iz]))*(med_lam[ix][iy][iz+1]/pml_y[iy])
        lam_wv_m1 = (1.0/(4.0*grid_dz*grid_dy*pml_z[iz]))*(med_lam[ix][iy][iz-1]/pml_y[iy])
        
        mu_wv_p1 = (1.0/(4.0*grid_dz*grid_dy*pml_y[iy]))*(med_mu[ix][iy+1][iz]/pml_z[iz])
        mu_wv_m1 = (1.0/(4.0*grid_dz*grid_dy*pml_y[iy]))*(med_mu[ix][iy-1][iz]/pml_z[iz])

        WV00 = lam_wv_m1 + mu_wv_m1
        WV01 = -lam_wv_m1 - mu_wv_p1
        WV10 = -lam_wv_p1 - mu_wv_m1
        WV11 = lam_wv_p1 + mu_wv_p1
        
        sten27_Kwv = np.array([WV00, WV01, WV10, WV11])
        
        return sten27_Kuu, sten27_Kuv, sten27_Kuw, sten27_Kvu, sten27_Kvv, sten27_Kvw, sten27_Kwu, sten27_Kwv, sten27_Kww
        
        
    def update_global_K(self,ix, iy, iz, K, sten27_Kuu, sten27_Kuv, sten27_Kuw, sten27_Kvu, \
        sten27_Kvv, sten27_Kvw, sten27_Kwu, sten27_Kwv, sten27_Kww, grid_nx, grid_ny, grid_nz):
        '''
        Update the global stiffness matrix from each 9 point stencils
        '''
        #
        # K is scipy sparse matrix
        # 
        '''
        [UU000, UU001, UU002, UU010, UU011, UU012, UU020, UU021, UU022, \
        UU100, UU101, UU102, UU110, UU111, UU112, UU120, UU121, UU122,\
        UU200, UU201, UU202, UU210, UU211, UU212, UU220, UU221, UU222]
        '''
        
        nnode = grid_nx*grid_ny*grid_nz # number of nodes
        # global mapping for different regions regions
        # Coordinate for the nodes
        # xy plane z = 1 
        p111 = (grid_nx*grid_ny)*iz + grid_nx*iy + ix # 111 (central node)
        p011 = p111-1; p211 = p111 + 1 # along x axes
        
        p101 = p111 - grid_nx; p001 = p101 -1; p201 = p101 + 1
        p121 = p111 + grid_nx; p021 = p121 -1; p221 = p121 + 1
        
        
        # xy plane z = 0
        p110 = p111-(grid_nx*grid_ny); p010 = p110-1; p210 = p110 +1
        p100 = p110-grid_nx; p000 = p100-1; p200 = p100+1
        p120 = p110+grid_nx; p020 = p120-1; p220 = p120+1
        
        #xy plane z = 2
        p112 = p111+(grid_nx*grid_ny); p012 = p112-1; p212 = p112 +1
        p102 = p112-grid_nx; p002 = p102-1; p202 = p102+1
        p122 = p112+grid_nx; p022 = p122-1; p222 = p122+1
        
        
        #print("The stencils:",  p000, p001, p002, p010, p011, p012, p020, p021, p022, \
        #                        p100, p101, p102, p110, p111, p112, p120, p121, p122, \
        #                        p200, p201, p202, p210, p211, p212, p220, p221, p222)
        # Populating the global matrix to the scipy sparse matrix
        #
        # Population in uu region
        pRow = p111 + 0
        #
        K[pRow, p000] += sten27_Kuu[0]
        K[pRow, p001] += sten27_Kuu[1]
        K[pRow, p002] += sten27_Kuu[2]
        #
        K[pRow, p010] += sten27_Kuu[3]
        K[pRow, p011] += sten27_Kuu[4]
        K[pRow, p012] += sten27_Kuu[5]
        #
        K[pRow, p020] += sten27_Kuu[6]
        K[pRow, p021] += sten27_Kuu[7]
        K[pRow, p022] += sten27_Kuu[8]
        #
        K[pRow, p100] += sten27_Kuu[9]
        K[pRow, p101] += sten27_Kuu[10]
        K[pRow, p102] += sten27_Kuu[11]
        #
        K[pRow, p110] += sten27_Kuu[12]
        K[pRow, p111] += sten27_Kuu[13]
        K[pRow, p112] += sten27_Kuu[14]
        #
        K[pRow, p120] += sten27_Kuu[15]
        K[pRow, p121] += sten27_Kuu[16]
        K[pRow, p122] += sten27_Kuu[17]
        #
        K[pRow, p200] += sten27_Kuu[18]
        K[pRow, p201] += sten27_Kuu[19]
        K[pRow, p202] += sten27_Kuu[20]
        #
        K[pRow, p210] += sten27_Kuu[21]
        K[pRow, p211] += sten27_Kuu[22]
        K[pRow, p212] += sten27_Kuu[23]
        #
        K[pRow, p220] += sten27_Kuu[24]
        K[pRow, p221] += sten27_Kuu[25]
        K[pRow, p222] += sten27_Kuu[26]
        
        # population in uv region
        K[pRow, p001+nnode] += sten27_Kuv[0]
        K[pRow, p021+nnode] += sten27_Kuv[1]
        K[pRow, p201+nnode] += sten27_Kuv[2]
        K[pRow, p221+nnode] += sten27_Kuv[3]
        
        # population in uw region
        K[pRow, p010+2*nnode] += sten27_Kuw[0]
        K[pRow, p012+2*nnode] += sten27_Kuw[1]
        K[pRow, p210+2*nnode] += sten27_Kuw[2]
        K[pRow, p212+2*nnode] += sten27_Kuw[3]
        
        # Population in vv region
        pRow = p111 + nnode
        #
        K[pRow, p000+nnode] += sten27_Kvv[0]
        K[pRow, p001+nnode] += sten27_Kvv[1]
        K[pRow, p002+nnode] += sten27_Kvv[2]
        #
        K[pRow, p010+nnode] += sten27_Kvv[3]
        K[pRow, p011+nnode] += sten27_Kvv[4]
        K[pRow, p012+nnode] += sten27_Kvv[5]
        #
        K[pRow, p020+nnode] += sten27_Kvv[6]
        K[pRow, p021+nnode] += sten27_Kvv[7]
        K[pRow, p022+nnode] += sten27_Kvv[8]
        #
        K[pRow, p100+nnode] += sten27_Kvv[9]
        K[pRow, p101+nnode] += sten27_Kvv[10]
        K[pRow, p102+nnode] += sten27_Kvv[11]
        #
        K[pRow, p110+nnode] += sten27_Kvv[12]
        K[pRow, p111+nnode] += sten27_Kvv[13]
        K[pRow, p112+nnode] += sten27_Kvv[14]
        #
        K[pRow, p120+nnode] += sten27_Kvv[15]
        K[pRow, p121+nnode] += sten27_Kvv[16]
        K[pRow, p122+nnode] += sten27_Kvv[17]
        #
        K[pRow, p200+nnode] += sten27_Kvv[18]
        K[pRow, p201+nnode] += sten27_Kvv[19]
        K[pRow, p202+nnode] += sten27_Kvv[20]
        #
        K[pRow, p210+nnode] += sten27_Kvv[21]
        K[pRow, p211+nnode] += sten27_Kvv[22]
        K[pRow, p212+nnode] += sten27_Kvv[23]
        #
        K[pRow, p220+nnode] += sten27_Kvv[24]
        K[pRow, p221+nnode] += sten27_Kvv[25]
        K[pRow, p222+nnode] += sten27_Kvv[26]
        
         # population in vu region
        K[pRow, p001] += sten27_Kvu[0]
        K[pRow, p021] += sten27_Kvu[1]
        K[pRow, p201] += sten27_Kvu[2]
        K[pRow, p221] += sten27_Kvu[3]
        
         # population in vw region
        K[pRow, p100+2*nnode] += sten27_Kvw[0]
        K[pRow, p102+2*nnode] += sten27_Kvw[1]
        K[pRow, p120+2*nnode] += sten27_Kvw[2]
        K[pRow, p122+2*nnode] += sten27_Kvw[3]
        
        # Population in ww region
        pRow = p111 + 2*nnode
        #
        K[pRow, p000+2*nnode] += sten27_Kww[0]
        K[pRow, p001+2*nnode] += sten27_Kww[1]
        K[pRow, p002+2*nnode] += sten27_Kww[2]
        #
        K[pRow, p010+2*nnode] += sten27_Kww[3]
        K[pRow, p011+2*nnode] += sten27_Kww[4]
        K[pRow, p012+2*nnode] += sten27_Kww[5]
        #
        K[pRow, p020+2*nnode] += sten27_Kww[6]
        K[pRow, p021+2*nnode] += sten27_Kww[7]
        K[pRow, p022+2*nnode] += sten27_Kww[8]
        #
        K[pRow, p100+2*nnode] += sten27_Kww[9]
        K[pRow, p101+2*nnode] += sten27_Kww[10]
        K[pRow, p102+2*nnode] += sten27_Kww[11]
        #
        K[pRow, p110+2*nnode] += sten27_Kww[12]
        K[pRow, p111+2*nnode] += sten27_Kww[13]
        K[pRow, p112+2*nnode] += sten27_Kww[14]
        #
        K[pRow, p120+2*nnode] += sten27_Kww[15]
        K[pRow, p121+2*nnode] += sten27_Kww[16]
        K[pRow, p122+2*nnode] += sten27_Kww[17]
        #
        K[pRow, p200+2*nnode] += sten27_Kww[18]
        K[pRow, p201+2*nnode] += sten27_Kww[19]
        K[pRow, p202+2*nnode] += sten27_Kww[20]
        #
        K[pRow, p210+2*nnode] += sten27_Kww[21]
        K[pRow, p211+2*nnode] += sten27_Kww[22]
        K[pRow, p212+2*nnode] += sten27_Kww[23]
        #
        K[pRow, p220+2*nnode] += sten27_Kww[24]
        K[pRow, p221+2*nnode] += sten27_Kww[25]
        K[pRow, p222+2*nnode] += sten27_Kww[26]
        
         # population in wu region
        K[pRow, p010] += sten27_Kwu[0]
        K[pRow, p012] += sten27_Kwu[1]
        K[pRow, p210] += sten27_Kwu[2]
        K[pRow, p212] += sten27_Kwu[3]
        
         # population in wv region
        K[pRow, p100+nnode] += sten27_Kwv[0]
        K[pRow, p102+nnode] += sten27_Kwv[1]
        K[pRow, p120+nnode] += sten27_Kwv[2]
        K[pRow, p122+nnode] += sten27_Kwv[3]
