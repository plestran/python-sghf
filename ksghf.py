#!/usr/bin/python

from __future__ import division
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
  h_ao_full = np.zeros([n_spinor,n_spinor],dtype=complex)
  h_ao = np.array([[ -1.49241090e+00,-9.56945828e-01,-9.56945828e-01],
                   [ -9.56945828e-01,-1.49241090e+00,-9.56945824e-01], 
                   [ -9.56945828e-01,-9.56945824e-01,-1.49241090e+00]],dtype=complex)
  h_ao_full[:n_mo,:n_mo]                 = h_ao
  h_ao_full[n_mo:n_spinor,n_mo:n_spinor] = h_ao

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

  # MO coefficients
  orb_real = np.array([[  3.24039668e-01,-5.60702157e-01,-5.69914129e-01,-4.69420911e-01, 9.77388234e-02, -1.60936014e-01],
                       [  1.89213110e-01, 1.94808612e-01, 9.10402545e-02,-2.08692970e-01, 5.62610789e-01,  3.30265281e-01],
                       [ -4.10983784e-01, 2.60344409e-01, 4.61353051e-02,-4.80551061e-01, 3.42485603e-01, -2.88245153e-01],
                       [  3.28918437e-01, 9.68676114e-02, 2.25867257e-01,-2.34496774e-01,-9.78207783e-02,  1.49504157e-01],
                       [ -3.90530437e-01,-4.01667170e-02,-2.70815525e-01,-6.65778471e-02,-2.86876810e-01,  3.32615653e-01],
                       [  1.90172655e-01, 1.88185106e-01,-1.32512464e-01, 1.18598044e-01,-1.89326387e-01, -2.64381628e-01]])
  orb_imag = np.array([[  0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,  0.00000000e+00],
                       [  3.98016969e-01, 3.40547551e-01,-2.12856495e-01, 1.42195271e-01,-2.59589121e-01, -2.03707701e-01],
                       [ -4.19663554e-01,-2.17845613e-01,-2.19864990e-01, 1.68555446e-01,-9.51114985e-02,  1.43184195e-01],
                       [  9.55672981e-02,-3.97584043e-01, 8.77787912e-02, 5.66614796e-01, 4.90723291e-01, -8.79272678e-02],
                       [ -2.08417679e-01, 2.46526790e-01,-2.95395767e-01, 1.56891064e-01, 2.15851620e-01, -5.59006890e-01],
                       [  7.94731350e-02, 3.88838367e-01,-5.78145480e-01, 1.93332668e-01, 2.55096384e-01,  4.43664756e-01]])
  orb = np.empty([n_spinor, n_spinor],dtype=complex)
  for i in range(n_spinor):
    for j in range(n_spinor):
      orb[i,j] = orb_real[i,j] + 1j*orb_imag[i,j]

  # transformation matrix and its redimensioned form to work with full matrices
  x_small = np.array([[  1.17562979e+00,-2.33638436e-01,-2.33638436e-01],
                      [ -2.33638436e-01, 1.17562979e+00,-2.33638434e-01],
                      [ -2.33638436e-01,-2.33638434e-01, 1.17562979e+00]],dtype=complex)
  x_s = np.zeros([n_spinor, n_spinor],dtype=complex)
  x_s[:n_mo,:n_mo]         = x_small
  x_s[n_mo:n_spinor,n_mo:n_spinor] = x_small

  # orthonormal MO coefficients
  ortho_orb = orb
  orb       = x_s.dot(ortho_orb)

  return n_mo, n_spinor, enuc, h_ao_full, h_ao, ao_2e, orb, x_small, x_s, ortho_orb
#---------------------------------------

#---------------------------------------
def build_G_matrix(h_ao,ao_2e,p,n_spinor,n_mo):
  ''' Build the perturbation tensor (G) in the AO basis '''

  # scatter the AO density matrix
  p_aa = p[:n_mo,:n_mo]
  p_ab = p[:n_mo,n_mo:n_spinor]
  p_ba = p[n_mo:n_spinor,:n_mo]
  p_bb = p[n_mo:n_spinor,n_mo:n_spinor]
  p_t  = p_aa + p_bb

  # form G matrices and gather back together
  G_aa = np.zeros([n_mo,n_mo],dtype=complex)
  G_ab = np.zeros([n_mo,n_mo],dtype=complex)
  G_ba = np.zeros([n_mo,n_mo],dtype=complex)
  G_bb = np.zeros([n_mo,n_mo],dtype=complex)
  for i in range(n_mo):
    for j in range(n_mo):
      for k in range(n_mo):
        for l in range(n_mo):
          G_aa[i,j] += p_t[l,k]*ao_2e[i,j,k,l] - p_aa[l,k]*ao_2e[i,l,k,j]
          G_ab[i,j] -= p_ab[l,k] * ao_2e[i,l,k,j]
          G_ba[i,j] -= p_ba[l,k] * ao_2e[i,l,k,j]
          G_bb[i,j] += p_t[l,k]*ao_2e[i,j,k,l] - p_bb[l,k]*ao_2e[i,l,k,j]

  # form the full G matrix
  G = np.zeros([n_spinor,n_spinor],dtype=complex)
  G[:n_mo,:n_mo]                 = G_aa
  G[:n_mo,n_mo:n_spinor]         = G_ab
  G[n_mo:n_spinor,:n_mo]         = G_ba
  G[n_mo:n_spinor,n_mo:n_spinor] = G_bb

  f_aa = h_ao + G_aa
  f_ab = G_ab
  f_ba = G_ba
  f_bb = h_ao + G_bb

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
def Wigner(grid_index,grdb,spin,ncis,ngrdb):
  ''' Sets up the Wigner d-matrix elements.
    Only works for doublets for this debugging case.  '''

  dmt = np.zeros([ngrdb,ncis,ncis],dtype=complex)
  sz  = np.zeros([ncis],dtype=complex)
  for i in range(ngrdb):
    angb = grdb[i]
    if spin == 0.5: 
      dmt[i,0,0] =  np.cos(angb/2)
      dmt[i,0,1] = -np.sin(angb/2)
      dmt[i,1,0] =  np.sin(angb/2)
      dmt[i,1,1] =  np.cos(angb/2)
    else:
      print "not a recognized spin\n"
      exit() 
 
  for i in range(ncis): sz[i] = spin - i

  return dmt, sz
#---------------------------------------

def form_rot_density(anga,angb,angg,p,pstr,orb,nov,n_spinor,n_mo,I,J):
  ''' Form the rotation matrix R_g and overlap matrix N_g and
      the rotated density p_g '''

  Ra   = np.zeros([n_spinor, n_spinor],dtype=complex)
  Rb   = np.zeros([n_spinor, n_spinor],dtype=complex)
  Rg   = np.zeros([n_spinor, n_spinor],dtype=complex)
  N    = np.zeros([n_mo, n_mo],dtype=complex)
  eye  = np.identity(n_mo)
  
  # Form the rotation matrix in the OAO basis
  Ra[:n_mo,:n_mo]                 =  np.exp(-1j*anga/2 ) * eye
  Ra[n_mo:n_spinor,n_mo:n_spinor] =  np.exp( 1j*anga/2 ) * eye
  Rb[:n_mo,:n_mo]                 =  np.cos( angb/2 ) * eye
  Rb[:n_mo,n_mo:n_spinor]         = -np.sin( angb/2 ) * eye
  Rb[n_mo:n_spinor,:n_mo]         =  np.sin( angb/2 ) * eye
  Rb[n_mo:n_spinor,n_mo:n_spinor] =  np.cos( angb/2 ) * eye
  Rg[:n_mo,:n_mo]                 =  np.exp(-1j*angg/2 ) * eye
  Rg[n_mo:n_spinor,n_mo:n_spinor] =  np.exp( 1j*angg/2 ) * eye
  R = Ra.dot(Rb).dot(Rg)

  # Transform to the NO basis
  R    = np.conj(nov).T.dot(R).dot(nov)
  c    = np.conj(nov).T.dot(orb)[:n_mo,:n_mo]
  cstr = np.conj(nov).T.dot(np.conj(orb))[:n_mo,:n_mo]

  # Form Ng, overlap (detN), and pg
  if I == 0 and J == 0:
    N    = LA.inv(p[:n_mo,:n_spinor].dot(R).dot(p[:n_spinor,:n_mo]))
    detN = LA.det(c) * LA.det(N) * np.conj(LA.det(c))
    p_g  = R.dot(p[:n_spinor,:n_mo]).dot(N).dot(p[:n_mo,:n_spinor])
  elif I == 1 and J == 0:
    N    = LA.inv(pstr[:n_mo,:n_spinor].dot(R).dot(p[:n_spinor,:n_mo]))
    detN = LA.det(cstr) * LA.det(N) * np.conj(LA.det(c))
    p_g  = R.dot(p[:n_spinor,:n_mo]).dot(N).dot(pstr[:n_mo,:n_spinor])
  elif I == 0 and J == 1:
    N    = LA.inv(p[:n_mo,:n_spinor].dot(R).dot(pstr[:n_spinor,:n_mo]))
    detN = LA.det(c) * LA.det(N) * np.conj(LA.det(cstr))
    p_g  = R.dot(pstr[:n_spinor,:n_mo]).dot(N).dot(p[:n_mo,:n_spinor])
  elif I == 1 and J == 1:
    N    = LA.inv(pstr[:n_mo,:n_spinor].dot(R).dot(pstr[:n_spinor,:n_mo]))
    detN = LA.det(cstr) * LA.det(N) * np.conj(LA.det(cstr))
    p_g  = R.dot(pstr[:n_spinor,:n_mo]).dot(N).dot(pstr[:n_mo,:n_spinor])

  return R, N, detN, p_g
#---------------------------------------

#---------------------------------------
def Fock_build(h_ao,h_ortho,p,ao_2e,nov,x,n_spinor,n_mo):
  ''' Build a rotated Fock matrix in the NO basis '''

  # transform density from NO to OAO basis
  p_temp = nov.dot(p).dot(np.transpose(np.conjugate(nov)))

  # transform each spin block to AO basis
  p_aa = x.dot(p_temp[:n_mo,:n_mo]).dot(x.T)
  p_ab = x.dot(p_temp[:n_mo,n_mo:n_spinor]).dot(x.T)
  p_ba = x.dot(p_temp[n_mo:n_spinor,:n_mo]).dot(x.T)
  p_bb = x.dot(p_temp[n_mo:n_spinor,n_mo:n_spinor]).dot(x.T)
  p_t  = p_aa + p_bb

  # form the different blocks of G
  G_aa = np.zeros([n_mo,n_mo],dtype=complex)
  G_ab = np.zeros([n_mo,n_mo],dtype=complex)
  G_ba = np.zeros([n_mo,n_mo],dtype=complex)
  G_bb = np.zeros([n_mo,n_mo],dtype=complex)
  for i in range(n_mo):
    for j in range(n_mo):
      for k in range(n_mo):
        for l in range(n_mo):
          G_aa[i,j] += p_t[l,k]*ao_2e[i,j,k,l] - p_aa[l,k]*ao_2e[i,l,k,j]
          G_ab[i,j] -= p_ab[l,k] * ao_2e[i,l,k,j]
          G_ba[i,j] -= p_ba[l,k] * ao_2e[i,l,k,j]
          G_bb[i,j] += p_t[l,k]*ao_2e[i,j,k,l] - p_bb[l,k]*ao_2e[i,l,k,j]

  # form fg in the AO basis
  f_aa = h_ao + G_aa
  f_ab = G_ab
  f_ba = G_ba
  f_bb = h_ao + G_bb

  # transform to orthonormal basis
  f_aa = x.T.dot(f_aa).dot(x)
  f_ab = x.T.dot(f_ab).dot(x)
  f_ba = x.T.dot(f_ba).dot(x)
  f_bb = x.T.dot(f_bb).dot(x)

  # transform to NO basis
  F = np.zeros([n_spinor,n_spinor],dtype=complex)
  F[:n_mo,:n_mo]                 = f_aa
  F[:n_mo,n_mo:n_spinor]         = f_ab
  F[n_mo:n_spinor,:n_mo]         = f_ba
  F[n_mo:n_spinor,n_mo:n_spinor] = f_bb
  F = np.conj(nov.T).dot(F).dot(nov)

  # add one-electron part
  h_no = np.conj(nov.T).dot(h_ortho).dot(nov)

  return F, h_no
#---------------------------------------

#---------------------------------------
def collect_grid_weights(ngrdt,grid_index,grda,wgta,grdb,wgtb,grdg,wgtg,dmt,sz):
  ''' Collect all grid weights together into one array '''
 
  x = np.zeros([ngrdt,ncis,ncis],dtype=complex)

  # Loop over all grid weights and spin projections
  for i in range(ngrdt): 
    ianga = int(grid_index[i,0])
    iangb = int(grid_index[i,1])
    iangg = int(grid_index[i,2])
    wgt   = wgta[ianga] * wgtb[iangb] * wgtg[iangg]
    anga, angb, angg = grda[ianga], grdb[iangb], grdg[iangg]
    for m in range(ncis):
      for k in range(ncis):
        x[i,m,k] = ( wgt * dmt[iangb,m,k] * 
                     np.exp(1j * anga * sz[m]) * 
                     np.exp(1j * angg * sz[k]) )

  return x
#---------------------------------------

#---------------------------------------
def form_effective_fock(R,N,p,pstr,p_g,h,f_g,nov,n_mo,n_s,I,J):
  ''' Build the effective PHF Fock matrix in the NO basis '''
 
  # Note that X_g is \mathcal{Y}_g in Carlos' paper
  eye = np.identity(n_s,dtype=complex)
  F_g = np.zeros([n_s,n_s],dtype=complex)
  X_g = np.zeros([n_s,n_s],dtype=complex)
  U   = nov.T.dot(nov)

  # form the Xg matrix
  if I == 0 and J == 0:
    X_g[:n_s,:n_mo] += R.dot(p[:n_s,:n_mo]).dot(N)
    X_g[:n_mo,:n_s] += N.dot(p[:n_mo,:n_s]).dot(R)
  elif I == 1 and J == 0:
    X_g[:n_mo,:n_s] += (R.dot(p[:n_s,:n_mo]).dot(N)).T
    X_g = np.conj(U.T).dot(X_g).dot(U)
    X_g[:n_mo,:n_s] += N.dot(pstr[:n_mo,:n_s]).dot(R)
  elif I == 0 and J == 1:
    X_g[:n_s,:n_mo] += (N.dot(p[:n_mo,:n_s]).dot(R)).T
    X_g = np.conj(U.T).dot(X_g).dot(U)
    X_g[:n_s,:n_mo] += R.dot(pstr[:n_s,:n_mo]).dot(N)
  elif I == 1 and J == 1:
    X_g[:n_mo,:n_s] += (R.dot(pstr[:n_s,:n_mo]).dot(N)).T
    X_g[:n_s,:n_mo] += (N.dot(pstr[:n_mo,:n_s]).dot(R)).T
    X_g = np.conj(U.T).dot(X_g).dot(U)

  # calculate the energy at this grid point
  E_g = 0.
  for i in range(n_s):
    for j in range(n_s):
      E_g += (h[i,j] + f_g[i,j])*p_g[j,i]/2

  # add contributions to form Fg
  if I == 0 and J == 0:
    F_g[:n_mo,:n_s] += N.dot(p[:n_mo,:n_s]).dot(f_g).dot(eye-p_g).dot(R)
    F_g[:n_s,:n_mo] += (eye-p_g).dot(f_g).dot(R).dot(p[:n_s,:n_mo]).dot(N)
  elif I == 1 and J == 0:
    F_g[:n_mo,:n_s] += ((eye-p_g).dot(f_g).dot(R).dot(p[:n_s,:n_mo]).dot(N)).T
    F_g = np.conj(U.T).dot(F_g).dot(U)
    F_g[:n_mo,:n_s] += N.dot(pstr[:n_mo,:n_s]).dot(f_g).dot(eye-p_g).dot(R)
  elif I == 0 and J == 1:
    F_g[:n_s,:n_mo] += (N.dot(p[:n_mo,:n_s]).dot(f_g).dot(eye-p_g).dot(R)).T
    F_g = np.conj(U.T).dot(F_g).dot(U)
    F_g[:n_s,:n_mo] += (eye-p_g).dot(f_g).dot(R).dot(pstr[:n_s,:n_mo]).dot(N)
  elif I == 1 and J == 1:
    F_g[:n_s,:n_mo] += (N.dot(pstr[:n_mo,:n_s]).dot(f_g).dot(eye-p_g).dot(R)).T
    F_g[:n_mo,:n_s] += ((eye-p_g).dot(f_g).dot(R).dot(pstr[:n_s,:n_mo]).dot(N)).T
    F_g = np.conj(U.T).dot(F_g).dot(U)

  # Add the scaled X_g matrix
  F_g += X_g * E_g

  return X_g, F_g, E_g
#---------------------------------------

#---------------------------------------
def contract_w_ci(Xci,Fci,H,S,nci,n_spinor,enuc):
  ''' Solve the CI problem and contract the Fci and Xci matrices with the
      resulting eigenvectors/eigenvalue to form the final effective Fock 
      matrix in the NO basis '''

# print "Ham = \n", H.real
# print "Ham = \n", H.imag

  # diagonalize the overlap matrix to check if it is positive definite
  w_S, vec_S = LA.eig(-S)
  thresh = 1.e-10
  for i in range(nci):
    if (abs(w_S[i]) > thresh):
      vec_S[:,i] /= np.sqrt(-w_S[i])
    else:
      vec_S[:,i] *= 0

  # orthonormalize the hamiltonian and diagonalize it
  Hn = np.transpose(np.conjugate(vec_S)).dot(H).dot(vec_S)
  w_H, vec_H = LA.eigh(Hn)

  # transform back from the orthonormal basis 
  ci = vec_S.dot(vec_H)
  ci, E = ci[:,0], w_H[0]
  print "SCFIt = %2d" % (count + 1)
  print "\tPHF Total Energy = %.10f" % (E + enuc)

  # contract Xci and Fci w/ CI vector
  Xint = np.zeros([n_spinor,n_spinor],dtype=complex)
  Fint = np.zeros([n_spinor,n_spinor],dtype=complex)
  for Im in range(nci):
    for Jk in range(nci):
      Xint += np.conjugate(ci[Im]) * Xci[Im,Jk] * ci[Jk]
      Fint += np.conjugate(ci[Im]) * Fci[Im,Jk] * ci[Jk]

  # Add contribution from CI energy
  Fint -= E * Xint

  return Fint, E, ci
#---------------------------------------

#---------------------------------------
def modify_PHF_Fock(F_no,f_ortho,nov,n_mo,n_spinor):
  ''' Replace the off-diagonal blocks of the effective Fock matrix in the NO
      basis with those of the GHF matrix. Return the effective Fock matrix in
      the OAO basis '''

  # transform the GHF Fock matrix to the NO basis 
  f_no = np.transpose(np.conjugate(nov)).dot(f_ortho).dot(nov)

  # modify PHF fock matrix w/ HF fock matrix
  F_no[:n_mo,:n_mo]         = f_no[:n_mo,:n_mo]
  F_no[n_mo:n_spinor,n_mo:n_spinor] = f_no[n_mo:n_spinor,n_mo:n_spinor]

  # transform back to the OAO basis
  F_ortho = nov.dot(F_no).dot(np.conj(nov).T)

  return F_ortho
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
def scatter2(p,n_mo):
  ''' Scatter a full dimension matrix into its individual spin components '''

  p_aa     = np.zeros([n_mo, n_mo],dtype=complex) 
  p_ab     = np.zeros([n_mo, n_mo],dtype=complex) 
  p_ba     = np.zeros([n_mo, n_mo],dtype=complex) 
  p_bb     = np.zeros([n_mo, n_mo],dtype=complex) 

  # grab each spin block
  p_aa = p[:n_mo,:n_mo]
  p_ab = p[n_mo:2*n_mo,:n_mo]
  p_ba = p[:n_mo,n_mo:2*n_mo]
  p_bb = p[n_mo:2*n_mo,n_mo:2*n_mo]

  # form the spin components
  p_scal = p_aa + p_bb
  p_mx   = p_ab + p_ba
  p_my   = 1j * (p_ab - p_ba)
  p_mz   = p_aa - p_bb

  return p_scal, p_mx, p_my, p_mz
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
def calculate_S2(ci,ncik,ncis):
  ''' Evaluate <S**2> for this density. 
      Note: this does a lot of redundant work and is not how it's 
            structured in Chronus 
            Plus it's broken right now...    
  '''

  # Loop over all grid points
  SSq = 0.
  for g in range(ngrdt):
    anga = grda[int(grid_index[g,0])]
    angb = grdb[int(grid_index[g,1])]
    angg = grdg[int(grid_index[g,2])]

    # loop over types of complex conjugation operators
    for I in range(ncik):
      for J in range(ncik):

        # form rotated matrices and other quantities in NO basis
        R, N, detN, p_g = form_rot_density(anga,angb,angg,p_no,pstr_no,
                                           ortho_orb,nov,n_spinor,n_mo,I,J)
        p_temp = nov.dot(p_g).dot(np.transpose(np.conjugate(nov)))
        p_scal, p_mx, p_my, p_mz = scatter(p_temp,n_mo)

        # start forming <S^2> from densities
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

        # contract with the CI vector
        for m in range(ncis):
          for k in range(ncis):
            Im, Jk = ncik*I+m, ncik*J+k
            weight = xmk[g,m,k]/detN
            SSq += weight* np.conjugate(ci[Im]) * SSqg * ci[Jk]

  print "\tPHF <S**2> = %.10f\n" % float(SSq.real*0.25)

#---------------------------------------

#---------------------------------------
if __name__ == '__main__':

  # load data from Chronus
  n_mo, n_spinor, enuc, h_ao_full, h_ao,\
  ao_2e, orb, x_small, x, ortho_orb = load_data()

  # Build the initial GHF density
  p = np.zeros([n_spinor,n_spinor],dtype=complex)
  for u in range(n_spinor):
    for v in range(n_spinor):
      for i in range(n_mo):
        p[u,v] += orb[u,i]*np.conj(orb[v,i])
  p_scal, p_mx, p_my, p_mz = scatter2(p,n_mo)

  # setup the grid for integration 
  spin  = 0.5
  ncis, ncik  = int(2*spin + 1), 1
  nci   = ncik * ncis
  ngrda, ngrdb, ngrdg = 6, 6, 6
  ngrdt = ngrda * ngrdb * ngrdg
  grda, wgta = trapezoid(0,2*np.pi,ngrda) 
  grdb, wgtb = gaussLeg(0,np.pi,ngrdb)
  grdg, wgtg = trapezoid(0,2*np.pi,ngrdg) 

  # setup index of grid points to make things easier later on
  grid_index = np.zeros([ngrdt,3])
  count = 0
  for a in range(ngrda):
    for b in range(ngrdb):
      for g in range(ngrdg):
        grid_index[count,0] = a
        grid_index[count,1] = b
        grid_index[count,2] = g
        count += 1

  # form Wigner small-d matrix
  dmt, sz = Wigner(grid_index,grdb,spin,ncis,ngrdb)

  # collect grid weights into a single array
  xmk = collect_grid_weights(ngrdt,grid_index,grda,wgta,grdb,
                             wgtb,grdg,wgtg,dmt,sz)

  # SCF optimization
  EOld, count, conv = 1., 0, False
  h_ortho = np.transpose(x).dot(h_ao_full).dot(x)
  while (count < 500 and not conv):

    # form OAO GHF quantities
    p       = np.zeros([n_spinor,n_spinor],dtype=complex)
    p_ortho = np.zeros([n_spinor,n_spinor],dtype=complex)
    for u in range(n_spinor):
      for v in range(n_spinor):
        for i in range(n_mo):
          p_ortho[u,v] += ortho_orb[u,i]*np.conj(ortho_orb[v,i])
          p[u,v] += orb[u,i]*np.conj(orb[v,i])
    G = build_G_matrix(h_ao,ao_2e,p,n_spinor,n_mo)
    G_ortho = np.transpose(x).dot(G).dot(x)
    f_ortho = h_ortho + G_ortho
    E_GHF = calc_GHF_energy(h_ao_full,G,p)

    # transform density to NO basis
    no_w, nov = LA.eigh(-p_ortho) 
    p_no = np.conj(nov.T).dot(p_ortho).dot(nov)
    pstr_no = np.conj(nov.T).dot(np.conj(p_ortho)).dot(nov)

    # loop over grid points and add contributions
    S   = np.zeros([nci,nci],dtype=complex)
    H   = np.zeros([nci,nci],dtype=complex)
    Xci = np.zeros([nci,nci,n_spinor,n_spinor],dtype=complex)
    Fci = np.zeros([nci,nci,n_spinor,n_spinor],dtype=complex)
    for g in range(ngrdt):
      anga = grda[int(grid_index[g,0])]
      angb = grdb[int(grid_index[g,1])]
      angg = grdg[int(grid_index[g,2])]

      # loop over types of complex conjugation operators
      for I in range(ncik):
        for J in range(ncik):

          # form rotated matrices and other quantities in NO basis
          R, N, detN, p_g = form_rot_density(anga,angb,angg,p_no,pstr_no,
                                             ortho_orb,nov,n_spinor,n_mo,I,J)
          f_g, h_no = Fock_build(h_ao,h_ortho,p_g,ao_2e,nov,x_small,n_spinor,
                                 n_mo)
          X_g, F_g, E_g = form_effective_fock(R,N,p_no,pstr_no,p_g,h_no,
                                              f_g,nov,n_mo,n_spinor,I,J)

          # loop over spin projections and add contributions to matrices
          for m in range(ncis):
            for k in range(ncis):
              Im, Jk = ncik*I+m, ncik*J+k
              weight = xmk[g,m,k]/detN
              S[Im,Jk]   += weight
              H[Im,Jk]   += weight * E_g
              Xci[Im,Jk] += weight * X_g
              Fci[Im,Jk] += weight * F_g

    # form fully integrated effective Fock matrix
    F_no, E, ci = contract_w_ci(Xci,Fci,H,S,nci,n_spinor,enuc)
    F_ortho = modify_PHF_Fock(F_no,f_ortho,nov,n_mo,n_spinor)

    # diagonalize Fock matrix and calculate <S^2>
    w, ortho_orb = LA.eigh(F_ortho)
    orb = x.dot(ortho_orb)
#   calculate_S2(ci,ncik,ncis)

    # check for convergence
    if abs(E-EOld) < 1.e-8:
      conv = True
    else:
      count += 1
      EOld = E

