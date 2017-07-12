#!/usr/bin/python

import sys
import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt

#---------------------------------------
def load_data():
  ''' Dimension matrices using hard coded values from Chronus '''

  # dimension of the problem
  # Note: n_mo and n_el are the same in this example, so n_mo is always used
  n_mo, n_spinor = 3, 6

  # nuclear repulsion energy
  enuc = 1.5875316362

  # core hamiltonian
  h_ao = np.array([[ -1.49241090e+00,-9.56945828e-01,-9.56945828e-01],
                   [ -9.56945828e-01,-1.49241090e+00,-9.56945824e-01], 
                   [ -9.56945828e-01,-9.56945824e-01,-1.49241090e+00]],dtype=complex)

  # two-electron integrals
  ao_2e = np.zeros([n_mo, n_mo, n_mo, n_mo],dtype=complex)
  ao_2e[ 0, 0, 0, 0]=  0.7746059
  ao_2e[ 0, 0, 0, 1]=  0.3093090
  ao_2e[ 0, 0, 0, 2]=  0.3093090
  ao_2e[ 0, 0, 1, 0]=  0.3093090
  ao_2e[ 0, 0, 1, 1]=  0.4780414
  ao_2e[ 0, 0, 1, 2]=  0.2484902
  ao_2e[ 0, 0, 2, 0]=  0.3093090
  ao_2e[ 0, 0, 2, 1]=  0.2484902
  ao_2e[ 0, 0, 2, 2]=  0.4780414
  ao_2e[ 0, 1, 0, 0]=  0.3093090
  ao_2e[ 0, 1, 0, 1]=  0.1578658
  ao_2e[ 0, 1, 0, 2]=  0.1423294
  ao_2e[ 0, 1, 1, 0]=  0.1578658
  ao_2e[ 0, 1, 1, 1]=  0.3093090
  ao_2e[ 0, 1, 1, 2]=  0.1423294
  ao_2e[ 0, 1, 2, 0]=  0.1423294
  ao_2e[ 0, 1, 2, 1]=  0.1423294
  ao_2e[ 0, 1, 2, 2]=  0.2484902
  ao_2e[ 0, 2, 0, 0]=  0.3093090
  ao_2e[ 0, 2, 0, 1]=  0.1423294
  ao_2e[ 0, 2, 0, 2]=  0.1578658
  ao_2e[ 0, 2, 1, 0]=  0.1423294
  ao_2e[ 0, 2, 1, 1]=  0.2484902
  ao_2e[ 0, 2, 1, 2]=  0.1423294
  ao_2e[ 0, 2, 2, 0]=  0.1578658
  ao_2e[ 0, 2, 2, 1]=  0.1423294
  ao_2e[ 0, 2, 2, 2]=  0.3093090
  ao_2e[ 1, 0, 0, 0]=  0.3093090
  ao_2e[ 1, 0, 0, 1]=  0.1578658
  ao_2e[ 1, 0, 0, 2]=  0.1423294
  ao_2e[ 1, 0, 1, 0]=  0.1578658
  ao_2e[ 1, 0, 1, 1]=  0.3093090
  ao_2e[ 1, 0, 1, 2]=  0.1423294
  ao_2e[ 1, 0, 2, 0]=  0.1423294
  ao_2e[ 1, 0, 2, 1]=  0.1423294
  ao_2e[ 1, 0, 2, 2]=  0.2484902
  ao_2e[ 1, 1, 0, 0]=  0.4780414
  ao_2e[ 1, 1, 0, 1]=  0.3093090
  ao_2e[ 1, 1, 0, 2]=  0.2484902
  ao_2e[ 1, 1, 1, 0]=  0.3093090
  ao_2e[ 1, 1, 1, 1]=  0.7746059
  ao_2e[ 1, 1, 1, 2]=  0.3093090
  ao_2e[ 1, 1, 2, 0]=  0.2484902
  ao_2e[ 1, 1, 2, 1]=  0.3093090
  ao_2e[ 1, 1, 2, 2]=  0.4780414
  ao_2e[ 1, 2, 0, 0]=  0.2484902
  ao_2e[ 1, 2, 0, 1]=  0.1423294
  ao_2e[ 1, 2, 0, 2]=  0.1423294
  ao_2e[ 1, 2, 1, 0]=  0.1423294
  ao_2e[ 1, 2, 1, 1]=  0.3093090
  ao_2e[ 1, 2, 1, 2]=  0.1578658
  ao_2e[ 1, 2, 2, 0]=  0.1423294
  ao_2e[ 1, 2, 2, 1]=  0.1578658
  ao_2e[ 1, 2, 2, 2]=  0.3093090
  ao_2e[ 2, 0, 0, 0]=  0.3093090
  ao_2e[ 2, 0, 0, 1]=  0.1423294
  ao_2e[ 2, 0, 0, 2]=  0.1578658
  ao_2e[ 2, 0, 1, 0]=  0.1423294
  ao_2e[ 2, 0, 1, 1]=  0.2484902
  ao_2e[ 2, 0, 1, 2]=  0.1423294
  ao_2e[ 2, 0, 2, 0]=  0.1578658
  ao_2e[ 2, 0, 2, 1]=  0.1423294
  ao_2e[ 2, 0, 2, 2]=  0.3093090
  ao_2e[ 2, 1, 0, 0]=  0.2484902
  ao_2e[ 2, 1, 0, 1]=  0.1423294
  ao_2e[ 2, 1, 0, 2]=  0.1423294
  ao_2e[ 2, 1, 1, 0]=  0.1423294
  ao_2e[ 2, 1, 1, 1]=  0.3093090
  ao_2e[ 2, 1, 1, 2]=  0.1578658
  ao_2e[ 2, 1, 2, 0]=  0.1423294
  ao_2e[ 2, 1, 2, 1]=  0.1578658
  ao_2e[ 2, 1, 2, 2]=  0.3093090
  ao_2e[ 2, 2, 0, 0]=  0.4780414
  ao_2e[ 2, 2, 0, 1]=  0.2484902
  ao_2e[ 2, 2, 0, 2]=  0.3093090
  ao_2e[ 2, 2, 1, 0]=  0.2484902
  ao_2e[ 2, 2, 1, 1]=  0.4780414
  ao_2e[ 2, 2, 1, 2]=  0.3093090
  ao_2e[ 2, 2, 2, 0]=  0.3093090
  ao_2e[ 2, 2, 2, 1]=  0.3093090
  ao_2e[ 2, 2, 2, 2]=  0.7746059

  # initial orbitals
  orb_a = np.array([[ 4.60055224e-01, 9.96503108e-01,-5.35359679e-01],
                    [ 3.01612373e-01, 2.39043912e-08, 1.18334662e+00],
                    [ 4.60055227e-01,-9.96503129e-01,-5.35359635e-01]],dtype=complex)
  orb_b = np.array([[ 2.80666701e-01,-6.47678159e-01,-9.96503108e-01],
                    [ 6.38859002e-01, 1.04073936e+00,-1.05203884e-08],
                    [ 2.80666692e-01,-6.47678129e-01, 9.96503129e-01]],dtype=complex)

  # transformation matrix and its redimensioned form to work with full matrices
  x_small = np.array([[  1.17562979e+00,-2.33638436e-01,-2.33638436e-01],
                      [ -2.33638436e-01, 1.17562979e+00,-2.33638434e-01],
                      [ -2.33638436e-01,-2.33638434e-01, 1.17562979e+00]],dtype=complex)
  x_s = np.zeros([n_spinor, n_spinor],dtype=complex)
  x_s[:n_mo,:n_mo]         = x_small
  x_s[n_mo:n_spinor,n_mo:n_spinor] = x_small
  x = reverse_spin_block(x_s,n_mo,n_spinor)

  return n_mo, n_spinor, enuc, h_ao, ao_2e, orb_a, orb_b, x_small
#---------------------------------------

#---------------------------------------
def check_UHF_E(h_ao,ao_2e,orb_a,orb_b,n_mo):

  # Make sure everything matches with Gaussian
  p_a = np.zeros([n_mo,n_mo],dtype=complex)
  p_b = np.zeros([n_mo,n_mo],dtype=complex)
  for u in range(n_mo):
    for v in range(n_mo):
      for i in range(2):
        p_a[u,v] += orb_a[u,i]*np.conjugate(orb_a[v,i])
      for j in range(1):
        p_b[u,v] += orb_b[u,j]*np.conjugate(orb_b[v,j])
  p_t = p_a + p_b
  f_a = np.zeros([n_mo,n_mo],dtype=complex)
  f_b = np.zeros([n_mo,n_mo],dtype=complex)
  for u in range(n_mo):
    for v in range(n_mo):
      f_a[u,v] = h_ao[u,v]
      f_b[u,v] = h_ao[u,v]
      for s in range(n_mo):
        for y in range(n_mo):
          f_a[u,v] += p_t[y,s]*ao_2e[u,v,s,y] - p_a[y,s]*ao_2e[u,y,s,v]
          f_b[u,v] += p_t[y,s]*ao_2e[u,v,s,y] - p_b[y,s]*ao_2e[u,y,s,v]

  E = 0
  for u in range(n_mo):
    for v in range(n_mo):
      E += p_t[v,u]*h_ao[u,v] 
  E = E
# print "UHF 1E Energy    = ", E.real
  E = 0
  for u in range(n_mo):
    for v in range(n_mo):
      E += p_t[v,u]*h_ao[u,v] + p_a[v,u]*f_a[u,v] + p_b[v,u]*f_b[u,v]
  E = E/2 + enuc
# print "UHF Total Energy = ", E.real

  return p_a, p_b, p_t, f_a, f_b
#---------------------------------------

#---------------------------------------
def build_G_matrix(ao_2e,p,n_spinor,n_mo):
  ''' Build the perturbation tensor (G) in the AO basis '''

  # scatter the AO density matrix
  p_scal, p_mx, p_my, p_mz = scatter(p,n_mo)

  # form G matrices and gather back together
  G_scal = np.zeros([n_mo,n_mo],dtype=complex)
  G_mx   = np.zeros([n_mo,n_mo],dtype=complex)
  G_my   = np.zeros([n_mo,n_mo],dtype=complex)
  G_mz   = np.zeros([n_mo,n_mo],dtype=complex)
  for i in range(n_mo):
    for j in range(n_mo):
      for k in range(n_mo):
        for l in range(n_mo):
          G_scal[i,j] += p_scal[l,k] * (2*ao_2e[i,j,k,l] - ao_2e[i,l,k,j])
          G_mx[i,j]   -= p_mx[l,k] * ao_2e[i,l,k,j]
          G_my[i,j]   -= p_my[l,k] * ao_2e[i,l,k,j]
          G_mz[i,j]   -= p_mz[l,k] * ao_2e[i,l,k,j]
  G = gather(G_scal,G_mx,G_my,G_mz,n_mo,n_spinor)

  return G
#---------------------------------------

#---------------------------------------
def calc_GHF_energy(h,G,p):
  ''' Calculate the GHF energy '''

  # calculate the trace
  E = 0.
  for u in range(n_spinor):
    for v in range(n_spinor):
      E += p[v,u]*(h[u,v] + G[u,v]/2)

  return E.real + enuc
#---------------------------------------

#---------------------------------------
def reverse_spin_block(p_s,m,n):
  ''' Switch a spin-blocked matrix to be like Chronus' full
      dimensioned matrices '''

  # grab the individual spin components and put them all together
  p_aa = p_s[:m,:m]
  p_ab = p_s[:m,m:n]
  p_ba = p_s[m:n,:m]
  p_bb = p_s[m:n,m:n]
  p = np.zeros([n,n],dtype=complex)
  for i in range(m):
    for j in range(m):
      p[2*i,2*j]     = p_aa[i,j]  
      p[2*i+1,2*j]   = p_ba[i,j] 
      p[2*i,2*j+1]   = p_ab[i,j]  
      p[2*i+1,2*j+1] = p_bb[i,j] 

  return p
#---------------------------------------

#---------------------------------------
def trapezoid(start,end,n):
  ''' Builds a Trapezoid grid from start to end with n points '''

  grid   = np.zeros([n])
  weight = np.zeros([n])
  space = (end-start)/n
  start = start + space /2

  for i in range(n):
    grid[i]   = start + i*space
    weight[i] = 1./n

  return grid, weight
#---------------------------------------

#---------------------------------------
def gaussLeg(start,end,n):
  ''' Builds a Gauss-Legendre grid from start to end with n points '''

  grid   = np.zeros([n])
  weight = np.zeros([n])

  abscissa, weight = np.polynomial.legendre.leggauss(n)
  for u in range(n): 
    grid[u]   = end*abscissa[u]/2 + end/2
    weight[u] = (end/2)*weight[u]*np.sin(grid[u])

  return grid, weight
#---------------------------------------

#---------------------------------------
def Wigner(grdb,spin,ncis,ngrdb):
  ''' Sets up the Wigner d-matrix elements.
    Only works for doublets for this debugging case.  '''

  # NOTE: I hard-coded the dmt indices for the doublet case
  dmt = np.zeros([ngrdb,ncis,ncis],dtype=complex)
  for i in range(ngrdb):
    angb = grdb[i]
    if spin == 0.5: 
      dmt[i,0,0] =  np.cos(angb/2)
#     dmt[i,0,1] = -np.sin(angb/2)
#     dmt[i,1,0] =  np.sin(angb/2)
#     dmt[i,1,1] =  np.cos(angb/2)
    else:
      print "not a recognized spin\n"
      exit() 
 
  return dmt
#---------------------------------------

def form_rot_density(angb,p,nov,n_spinor,n_mo):
  ''' Form the rotation matrix R_g and overlap matrix N_g and
      the rotated density p_g '''

  R = np.zeros([n_spinor, n_spinor],dtype=complex)
  N = np.zeros([n_mo, n_mo],dtype=complex)

  # Form the rotation matrix in the OAO basis
  for i in range(0,n_spinor,2):
    R[i,i]     =  np.cos(angb/2)
    R[i,i+1]   =  np.sin(angb/2)
    R[i+1,i]   = -np.sin(angb/2)
    R[i+1,i+1] =  np.cos(angb/2)

  # Transform to the NO basis
  R = np.conjugate(np.transpose(nov)).dot(R).dot(nov)

  # Form the overlap matrix directly in the NO basis
  N = LA.inv(p[:n_mo,:n_spinor].dot(R).dot(p[:n_spinor,:n_mo]))

  # calculate determinant of Ng  
  detN = LA.det(N.dot(p[:n_mo,:n_mo]))

  # form rotated density in the NO basis
  p_g = R.dot(p[:n_spinor,:n_mo]).dot(N).dot(p[:n_mo,:n_spinor])

  return R, N, detN, p_g
#---------------------------------------

#---------------------------------------
def Fock_build(h,p,ao_2e,nov,x,n_spinor,n_mo):
  ''' Build a rotated Fock matrix in the NO basis '''

  # transform densities from NO to OAO basis
  p_temp = nov.dot(p).dot(np.transpose(np.conjugate(nov)))

  # transform to AO basis
  p_scal, p_mx, p_my, p_mz = scatter(p_temp,n_mo)
  p_scal = x.dot(p_scal).dot(np.transpose(x))
  p_mx = x.dot(p_mx).dot(np.transpose(x))
  p_my = x.dot(p_my).dot(np.transpose(x))
  p_mz = x.dot(p_mz).dot(np.transpose(x))

  # form the different blocks of G
  G_scal = np.zeros([n_mo,n_mo],dtype=complex)
  G_mx   = np.zeros([n_mo,n_mo],dtype=complex)
  G_my   = np.zeros([n_mo,n_mo],dtype=complex)
  G_mz   = np.zeros([n_mo,n_mo],dtype=complex)
  for i in range(n_mo):
    for j in range(n_mo):
      for k in range(n_mo):
        for l in range(n_mo):
          G_scal[i,j] += p_scal[l,k] * (2*ao_2e[i,j,k,l] - ao_2e[i,l,k,j])
          G_mx[i,j]   -= p_mx[l,k] * ao_2e[i,l,k,j]
          G_my[i,j]   -= p_my[l,k] * ao_2e[i,l,k,j]
          G_mz[i,j]   -= p_mz[l,k] * ao_2e[i,l,k,j]
  # transform to orthonormal basis
  G_scal = np.transpose(x).dot(G_scal).dot(x)
  G_mx   = np.transpose(x).dot(G_mx).dot(x)
  G_my   = np.transpose(x).dot(G_my).dot(x)
  G_mz   = np.transpose(x).dot(G_mz).dot(x)
  # transform to NO basis
  G = gather(G_scal,G_mx,G_my,G_mz,n_mo,n_spinor)
  G = np.transpose(np.conjugate(nov)).dot(G).dot(nov)

  # add one-electron part
  h_no = np.conjugate(np.transpose(nov)).dot(h).dot(nov)
  f    = h_no + G

  return f, h_no
#---------------------------------------

#---------------------------------------
def collect_grid_weights(ngrdb,grdb,wgtb,dmt):
  ''' Collect all grid weights together into one array '''
 
  x = np.zeros([ngrdb],dtype=complex)

  # Loop over all grid weights 
  # NOTE: I hard-coded the dmt indices for the doublet case
  for i in range(ngrdb): 
    x[i]  = wgtb[i] * dmt[i,0,0]

  return x
#---------------------------------------

#---------------------------------------
def form_effective_fock(R,N,p,p_g,h,f_g,n_mo,n_spinor):
  ''' Build the effective PHF Fock matrix in the NO basis '''
 
  # Note that X_g is \mathcal{Y}_g in Carlos' paper
  I   = np.identity(n_spinor,dtype=complex)
  F_g = np.zeros([n_spinor,n_spinor],dtype=complex)
  X_g = np.zeros([n_spinor,n_spinor],dtype=complex)

  # form the Xg matrix
  X_g[:n_spinor,:n_mo] += R.dot(p[:n_spinor,:n_mo]).dot(N)
  X_g[:n_mo,:n_spinor] += N.dot(p[:n_mo,:n_spinor]).dot(R)

  # calculate the energy at this grid point
  E_g = 0.
  for i in range(n_spinor):
    for j in range(n_spinor):
      E_g += (h[i,j] + f_g[i,j])*p_g[j,i]/2

  # add contributions to form Fg
  F_g = X_g * E_g
  F_g[:n_mo,:n_spinor] += N.dot(p[:n_mo,:n_spinor]).dot(f_g).dot(I-p_g).dot(R)
  F_g[:n_spinor,:n_mo] += (I-p_g).dot(f_g).dot(R).dot(p[:n_spinor,:n_mo]).dot(N)

  return X_g, F_g, E_g
#---------------------------------------

#---------------------------------------
def modify_PHF_Fock(F_no,f_ortho,nov,n_mo,n_spinor):
  ''' Replace the off-diagonal blocks of the effective Fock matrix in the NO
      basis with those of the GHF matrix. Return the effective Fock matrix in
      the OAO basis '''

  # zero out off-diagonal spin blocks
  F_ortho = nov.dot(F_no).dot(np.transpose(np.conjugate(nov)))
  for i in range(n_mo):
    for j in range(n_mo):
      F_ortho[2*i+1,2*j] = 0.
      F_ortho[2*i,2*j+1] = 0.
  F_no = np.transpose(np.conjugate(nov)).dot(F_ortho).dot(nov)

  # transform the GHF Fock matrix to the NO basis 
  f_no = np.transpose(np.conjugate(nov)).dot(f_ortho).dot(nov)

  # modify PHF fock matrix w/ HF fock matrix
  F_no[:n_mo,:n_mo] = f_no[:n_mo,:n_mo]
  F_no[n_mo:n_spinor,n_mo:n_spinor] = f_no[n_mo:n_spinor,n_mo:n_spinor]

  # transform back to the OAO basis
  F_ortho = nov.dot(F_no).dot(np.transpose(np.conjugate(nov)))
  F_oa = np.zeros([n_mo,n_mo],dtype=complex)
  F_ob = np.zeros([n_mo,n_mo],dtype=complex)
  for i in range(n_mo):
    for j in range(n_mo):
      F_oa[i,j] = F_ortho[2*i,2*j]
      F_ob[i,j] = F_ortho[2*i+1,2*j+1]

  return F_oa, F_ob
#---------------------------------------

#---------------------------------------
def scatter(p,n_mo):
  ''' Scatter a full dimension matrix into its individual spin components '''

  p_aa     = np.zeros([n_mo, n_mo],dtype=complex) 
  p_ab     = np.zeros([n_mo, n_mo],dtype=complex) 
  p_ba     = np.zeros([n_mo, n_mo],dtype=complex) 
  p_bb     = np.zeros([n_mo, n_mo],dtype=complex) 

  # grab each spin block
  for i in range(n_mo):
    for j in range(n_mo):
      p_aa[i,j] = p[2*i,2*j] 
      p_ab[i,j] = p[2*i,2*j+1] 
      p_ba[i,j] = p[2*i+1,2*j] 
      p_bb[i,j] = p[2*i+1,2*j+1] 

  # form the spin components
  p_scalar = p_aa + p_bb
  p_mx     = p_ab + p_ba
  p_my     = 1j * (p_ab - p_ba)
  p_mz     = p_aa - p_bb

  return p_scalar, p_mx, p_my, p_mz
#---------------------------------------

#---------------------------------------
def gather(p_scal,p_mx,p_my,p_mz,n_mo,n_spinor):
  ''' Gather all spin components to form the full dimension matrix '''

  p    = np.zeros([n_spinor, n_spinor],dtype=complex) 

  # turn into spin blocks
  p_aa = 0.5 * (p_scal + p_mz)
  p_bb = 0.5 * (p_scal - p_mz)
  p_ba = 0.5 * (p_mx + 1j*p_my)
  p_ab = 0.5 * (p_mx - 1j*p_my)

  # place spin blocks into the right place
  for i in range(n_mo):
    for j in range(n_mo):
      p[2*i,2*j]     = p_aa[i,j] 
      p[2*i,2*j+1]   = p_ab[i,j]
      p[2*i+1,2*j]   = p_ba[i,j]
      p[2*i+1,2*j+1] = p_bb[i,j]

  return p
#---------------------------------------

#---------------------------------------
def calculate_S2(S):
  ''' Evaluate <S**2> for this density. 
      Note: this does a lot of redundant work and is not how it's 
            structured in Chronus '''

  # Loop over all grid points
  SSq = 0.
  for g in range(ngrdb):

    # form rotated matrices and other quantities in NO basis
    R, N, detN, p_g = form_rot_density(grdb[g],p_no,nov,n_spinor,n_mo)
    p_temp = nov.dot(p_g).dot(np.transpose(np.conjugate(nov)))
    p_scal, p_mx, p_my, p_mz = scatter(p_temp,n_mo)

    SSqg, Trx, Try, Trz = 0., 0., 0., 0.
    scr  = 0.5 * p_mz.dot(p_mz)
    scr2 = 0.5 * p_my.dot(p_my)
    scr3 = 0.5 * p_mx.dot(p_mx)
    scr4 = p_scal - 0.5 * p_scal.dot(p_scal)
    # Compute traces 
    for i in range(n_mo):
      Trz += p_mz[i,i] 
      Try += p_my[i,i] 
      Trx += p_mx[i,i] 
      SSqg += scr[i,i] + scr2[i,i] + scr3[i,i] + 3.*scr4[i,i]
    SSqg += Trz**2 + Try**2 + Trx**2
    SSqg *= 0.25

    weight = xmk[g]/detN
    SSq += weight * SSqg

  SSq /= S

  print "\tPHF <S**2> = %.10f\n" % float(SSq.real)
#---------------------------------------

#---------------------------------------
if __name__ == '__main__':

  # load data from Chronus and expand some quantites to be 2*NBasis
  n_mo, n_spinor, enuc, h_ao, ao_2e, orb_a, orb_b, x = load_data()
  ortho_a = LA.inv(x).dot(orb_a)
  ortho_b = LA.inv(x).dot(orb_b)
  p_a, p_b, p_t, f_a, f_b = check_UHF_E(h_ao,ao_2e,orb_a,orb_b,n_mo) 
  p_oa   = np.zeros([n_mo,n_mo],dtype=complex)
  p_ob   = np.zeros([n_mo,n_mo],dtype=complex)
  h_ao_f = np.zeros([n_spinor,n_spinor],dtype=complex)
  x_f    = np.zeros([n_spinor,n_spinor],dtype=complex)
  for i in range(n_mo):
    for j in range(n_mo):
      x_f[2*i,2*j]        = x[i,j] 
      x_f[2*i+1,2*j+1]    = x[i,j]
      h_ao_f[2*i,2*j]     = h_ao[i,j] 
      h_ao_f[2*i+1,2*j+1] = h_ao[i,j]

  # setup the grid for integration 
  spin, ncis   = 0.5, 1
  ngrdb = 12
  grdb, wgtb = gaussLeg(0,np.pi,ngrdb)

  # form Wigner small-d matrix
  dmt = Wigner(grdb,spin,ncis,ngrdb)

  # collect grid weights into a single array
  xmk = collect_grid_weights(ngrdb,grdb,wgtb,dmt)

  # SCF optimization
  zero = np.zeros([n_mo,n_mo],dtype=complex)
  EOld, count, conv = 1., 0, False
  h_ortho = np.transpose(x_f).dot(h_ao_f).dot(x_f)
  while (count < 100 and not conv):

    p_oa = np.zeros([n_mo,n_mo],dtype=complex)
    p_ob = np.zeros([n_mo,n_mo],dtype=complex)
    # form OAO UHF quantities
    for u in range(n_mo):
      for v in range(n_mo):
        for i in range(2):
          p_oa[u,v] += ortho_a[u,i]*np.conjugate(ortho_a[v,i])
        for j in range(1):
          p_ob[u,v] += ortho_b[u,j]*np.conjugate(ortho_b[v,j])
    p_ortho = gather(p_oa+p_ob,zero,zero,p_oa-p_ob,n_mo,n_spinor)
    orb_a = x.dot(ortho_a)
    orb_b = x.dot(ortho_b)
    p_a, p_b, p_t, f_a, f_b = check_UHF_E(h_ao,ao_2e,orb_a,orb_b,n_mo) 
    f_oa = np.transpose(x).dot(f_a).dot(x)
    f_ob = np.transpose(x).dot(f_b).dot(x)
    f_ortho = gather(f_oa+f_ob,zero,zero,f_oa-f_ob,n_mo,n_spinor)

    # transform density to NO basis and reorder to have occ first
    w_a, nov_a = LA.eigh(p_oa)
    w_b, nov_b = LA.eigh(p_ob)
    idxa, idxb = w_a.argsort()[::-1], w_b.argsort()[::-1]   
    w_a, nov_a = w_a[idxa], nov_a[:,idxa]
    w_b, nov_b = w_b[idxb], nov_b[:,idxb]
    p_no_a = np.conjugate(np.transpose(nov_a)).dot(p_oa).dot(nov_a)
    p_no_b = np.conjugate(np.transpose(nov_b)).dot(p_ob).dot(nov_b)
    nov = np.zeros([n_spinor,n_spinor],dtype=complex)
    for i in range(n_mo):
      for j in range(n_mo):
        nov[2*i,2*j]     = nov_a[i,j] 
        nov[2*i+1,2*j+1] = nov_b[i,j]
    p_no = np.conjugate(np.transpose(nov)).dot(p_ortho).dot(nov)

    # loop over grid points and add contributions
    E, S = 0., 0.
    F_no = np.zeros([n_spinor,n_spinor],dtype=complex)
    X_no = np.zeros([n_spinor,n_spinor],dtype=complex)
    for g in range(ngrdb):

      # form rotated matrices and other quantities in NO basis
      R, N, detN, p_g = form_rot_density(grdb[g],p_no,nov,n_spinor,n_mo)
      f_g, h_no = Fock_build(h_ortho,p_g,ao_2e,nov,x,n_spinor,n_mo)
      X_g, F_g, E_g = form_effective_fock(R,N,p_no,p_g,h_no,f_g,n_mo,n_spinor)

      # add contributions to various quantities   
      weight = xmk[g]/detN   
      S     += weight
      E     += weight * E_g
      X_no  += weight * X_g
      F_no  += weight * F_g

    # print the total PHF energy
    print "SCFIt = %2d" % (count + 1)
    print "PHF Energy = ", (E/S).real + enuc
 
    # form fully integrated effective Fock matrix and 
    # transform to the NO basis
    F_no = F_no/S - (E/S)*(X_no/S)
    F_oa, F_ob = modify_PHF_Fock(F_no,f_ortho,nov,n_mo,n_spinor)

    # diagonalize Fock matrix and calculate <S^2>
    w_a, ortho_a = LA.eigh(F_oa)
    w_b, ortho_b = LA.eigh(F_ob)
    calculate_S2(S)

    # check for convergence
    if abs(E-EOld) < 1.e-6:
      conv = True
    else:
      count += 1
      EOld = E

