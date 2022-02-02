
#from seismic_def import e_lami
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.lil import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, linalg as sla
from scipy.sparse import csr_matrix
import math
import time

from seismic_def import *
from freq_3D import freq_3D
from freq_PML import freq_PML

#------------------------
# Model initialization
#-------------------------

#---------------
# Geometry
len_x = 1.1* 10.0
len_y = 1.1* 10.0
len_z = 1.1* 10.0

grid_nx = 21
grid_ny = 21
grid_nz = 21

grid_dx = len_x/(grid_nx-1)
grid_dy = len_y/(grid_ny-1)
grid_dz = len_z/(grid_nz-1)

# PML layers
npml = 3 # in all directions
npad = 1 # zero row layers in outer boundaries

# wave frequency
omega = 15*2*math.pi # Hz

#----------------------
# Material Modoel

# Scalar wave velocities and Density
Cp = 3500.0 # m/s
Cs = 1429.0 # m/s
rho_sc = 2000 # kg/m3
'''
E= 33837e6 #Pa
nu = 0.2
rho_sc = 2500 #kg/m2
'''
zeta = 0.0
# --------------------------------------
#--------------------------------------

# ----------------------------------------
# MATERIAL ARRAYS COMPUTATION
# -----------------------------------------

# Lamis' Constants
lam_sc, mu_sc = v_lami(Cp, Cs, rho_sc)
#lam_sc, mu_sc = e_lami(E, nu)


# preparing  the starting material arrays
med_lam = np.full((grid_nx, grid_ny, grid_nz), lam_sc*(1.0 + zeta* 1.j))
med_mu = np.full((grid_nx, grid_ny, grid_nz), mu_sc*(1.0 + zeta* 1.j))
med_rho = np.full((grid_nx, grid_ny, grid_nz), rho_sc*(1.0 + zeta* 1.j))

# lamda + 2 mu
med_eta = med_lam + 2.0*med_mu

#--------------------------------
# PML
# -------------------------------

# initializing PML factor arrays
pml_x = np.ones((grid_nx,), dtype=np.complex)
pml_y = np.ones((grid_ny,), dtype=np.complex)
pml_z = np.ones((grid_nz,), dtype=np.complex)

pml_x_half = np.ones((grid_nx,), dtype=np.complex)
pml_y_half = np.ones((grid_ny,), dtype=np.complex)
pml_z_half = np.ones((grid_nz,), dtype=np.complex)

# Computation of PML grids
pml = freq_PML()
a0_PMLx = -Cp * math.log(1.e-4)/(grid_dx*npml)
a0_PMLy = -Cp * math.log(1.e-4)/(grid_dy*npml)
a0_PMLz = -Cp * math.log(1.e-4)/(grid_dz*npml)
print("a0: ", a0_PMLx, a0_PMLz)
# Compute PML in x and z directions
pml.pml_PSV(npml, npml, grid_nx, npad, 900/npml, omega, pml_x, pml_x_half)
pml.pml_PSV(npml, npml, grid_ny, npad, 900/npml, omega, pml_y, pml_y_half)
pml.pml_PSV(npml, npml, grid_nz, npad, 900/npml, omega, pml_z, pml_z_half)
print('xi: ', pml_x)
print('xi_half: ', pml_x_half)
#exit()

#--------------------------------
# STENCIL OPTIMIZTION PARAMETERS
# -------------------------------
opt_alpha1 = 0 #0.5828
opt_alpha2 = 0 #0.5828
opt_beta1 = 0 #0.5828
opt_beta2 = 0 #0.5828
opt_gamma1 = 0 #0.5828
opt_gamma2 = 0 #0.5828
opt_c = 0.6308
opt_d = 0.0923

# --------------------------------------
# INITIALIZATION OF COMPUTATIONAL ARRAYS
# ---------------------------------------

# Initialization of Stiffness matrix
ndof = grid_nx*grid_ny*grid_nz*3
K = lil_matrix( (ndof,ndof), dtype=np.complex )
F = lil_matrix((ndof,1), dtype = np.complex)

# Initialization of Stencil arrays
# 9 point stencils
sten27_Kuu = np.zeros((27,), dtype=np.complex)
sten27_Kuv = np.zeros((4,), dtype=np.complex)
sten27_Kuw = np.zeros((4,), dtype=np.complex)

sten27_Kvv = np.zeros((27,), dtype=np.complex)
sten27_Kvu = np.zeros((4,), dtype=np.complex)
sten27_Kvw = np.zeros((4,), dtype=np.complex)

sten27_Kww = np.zeros((27,), dtype=np.complex)
sten27_Kwu = np.zeros((4,), dtype=np.complex)
sten27_Kwv = np.zeros((4,), dtype=np.complex)



# --------------------------------------
# FREQUENCY DOMAIN PSV COMPUTATION
# ---------------------------------------
wave3 = freq_3D()

t1 = time.time()

#-------------------------------------------------------
# Matrix population
for ix in range(1,grid_nx-1):
    print("grid X: ", ix)
    for iy in range(1,grid_ny-1):
        for iz in range(1,grid_nz-1):
            # update for each stencil
            sten27_Kuu, sten27_Kuv, sten27_Kuw, sten27_Kvu, sten27_Kvv, sten27_Kvw, sten27_Kwu, sten27_Kwv, sten27_Kww = \
                wave3.stencil_3D(ix, iy, iz, grid_dx, grid_dy, grid_dz, pml_x, pml_x_half, pml_y, pml_y_half, pml_z, pml_z_half,\
            med_rho, med_lam, med_mu, med_eta, omega, opt_alpha1, opt_alpha2, opt_beta1, opt_beta2, opt_gamma1, opt_gamma2, opt_c, opt_d)

            #print(sten27_Kuu)
            
            # Update global stiffness Matrix in sparse matrix form
            wave3.update_global_K(ix, iy, iz, K, sten27_Kuu, sten27_Kuv, sten27_Kuw, sten27_Kvu, \
            sten27_Kvv, sten27_Kvw, sten27_Kwu, sten27_Kwv, sten27_Kww, grid_nx, grid_ny, grid_nz)



#--------------------------------------------------------------------
# MATRIX Conditioning: removing zero rows and columns from matrix
t2 = time.time()

print("K size: ", K.size)
print("F size: ", F.size)
#F[grid_nx*grid_nz+grid_nx*grid_nz//2+grid_nx//2] = 1.0
# removing zero rows and columns from the system

print("removing zero rows and columns.")
F = F.tocsc()[K.getnnz(1)>0]

K = K.tocsc()[:,K.getnnz(1)>0]

K = K.tocsr()[K.getnnz(1)>0,:]





# Creating source in b matrix (unit force at the center)
F[F.todense().size//6] = 1.0*rho_sc*omega*omega #(grid_dx*grid_dz)

print("K size: ", K.size)
print("F size: ", F.size)
print("removing zero rows and columns.[DONE]")

#--------------------------------
# Sparse solver
t3 = time.time()
print("Solving the system:")
U = spsolve(K,F)

print("Solving the system. [DONE]")
nx = grid_nx-2
ny = grid_ny-2
nz = grid_nz-2
#print(U)
print(U.size, nx*ny*nz)
U1 = U[0:(nx*ny*nz)]
U2 = U[(nx*ny*nz):(2*(nx*ny*nz))]
U3 = U[(2*(nx*ny*nz)):]
U1 = np.reshape(U1,(nz,ny,nx))
U2 = np.reshape(U2,(nz,ny,nx))
U3 = np.reshape(U3,(nz,ny,nx))

t4 = time.time()

print("Grid size: ", grid_nz)
print("Time to populate: ", t2-t1)
print("Time to process matrix: ", t3-t2)
print("Time to solve: ", t4-t3)

#-----------------------------------------------
# Plotting of results
print('Plotting initial materials')
plt.figure(1)
'''
plt.suptitle("Frequency domain PSV results for: nx = "+np.str(grid_nx)+", nz = "+np.str(grid_nz)+", dx = "+'%.2f' % grid_dx+", dz = "+'%.2f' % grid_dz\
    +"\nMedium: Cp = "+'%.2f' % Cp+", Cs = "+'%.2f' % Cs+", density = "+'%.2f' %rho_sc\
        +"\nCircular frequency = "+'%.2f' % omega+", number of PML layers = "+np.str(npml)\
            +"\nUnit force normalized with dx*dz applied in vertical direction at the center")
'''
plt.subplot(121)
plt.imshow(np.real(U1[nz//2][:][:]), cmap=cm.jet, animated=True,  interpolation='nearest')
plt.colorbar()
plt.title('(a) u-component (m)')
plt.xlabel('X [no. of grids]')
plt.ylabel('Z [no. of grids]')
plt.axhline(npml+npad, color='black', label='PML') # PML boundary 
plt.axvline(npml+npad, color='black', label='PML') # PML boundary
plt.axhline(nz-npml-1, color='black', label='PML') # PML boundary 
plt.axvline(nx-npml-1, color='black', label='PML') # PML boundary
#plt.grid()
plt.subplot(122)
plt.imshow(np.real(U2[nz//2][:][:]), cmap=cm.jet, animated=True,  interpolation='nearest') #
plt.colorbar()
plt.title('(b) w-component (m)')
plt.xlabel('X [no. of grids]')
plt.ylabel('Z [no. of grids]')
plt.axhline(npml, color='black', label='PML') # PML boundary 
plt.axvline(npml, color='black', label='PML') # PML boundary
plt.axhline(nz-npml-1, color='black', label='PML') # PML boundary 
plt.axvline(nx-npml-1, color='black', label='PML') # PML boundary
#plt.grid()
plt.show()