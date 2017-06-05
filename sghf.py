#!/usr/bin/python

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
  h_ao = np.array([[-0.149241E+01, 0.000000E+00,-0.956946E+00, 0.000000E+00,-0.956946E+00, 0.000000E+00],
                   [ 0.000000E+00,-0.149241E+01, 0.000000E+00,-0.956946E+00, 0.000000E+00,-0.956946E+00], 
                   [-0.956946E+00, 0.000000E+00,-0.149241E+01, 0.000000E+00,-0.956946E+00, 0.000000E+00],
                   [ 0.000000E+00,-0.956946E+00, 0.000000E+00,-0.149241E+01, 0.000000E+00,-0.956946E+00],
                   [-0.956946E+00, 0.000000E+00,-0.956946E+00, 0.000000E+00,-0.149241E+01, 0.000000E+00],
                   [ 0.000000E+00,-0.956946E+00, 0.000000E+00,-0.956946E+00, 0.000000E+00,-0.149241E+01]],dtype=complex)

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
  orb_real = np.array([[ 2.48476349e-10, 1.18656265e-09,-1.98609186e-09, 3.23081841e-09,-3.06201921e-09 ,  7.62784486e-10],
                       [ 2.13272155e-01,-4.51741533e-01, 1.83068424e-09,-8.36752418e-01,-7.35913859e-01 ,  9.92057617e-10],
                       [ 5.76732360e-10,-1.74551046e-10, 1.18331166e-09,-1.03460067e-09, 1.28949728e-10 , -9.87910233e-10],
                       [ 3.25308169e-01,-1.98461324e-01, 7.04634111e-01, 3.78556446e-01, 4.57977610e-01 ,  7.04634111e-01],
                       [ 1.09854462e-10, 7.22834445e-10, 6.32598935e-10,-2.60915355e-09, 3.18715937e-09 ,  7.43118253e-12],
                       [ 3.25308170e-01,-1.98461324e-01,-7.04634112e-01, 3.78556444e-01, 4.57977607e-01 , -7.04634112e-01]])
  orb_imag = np.array([[ 2.13272155e-01, 4.51741533e-01, 8.66135955e-10,-8.36752418e-01, 7.35913859e-01 , -1.41251598e-09],
                       [-2.04226200e-10, 5.65166461e-11,-1.99026910e-09,-2.76730099e-09,-1.85413231e-09 , -9.69862773e-10],
                       [ 3.25308169e-01, 1.98461325e-01, 7.04634111e-01, 3.78556447e-01,-4.57977610e-01 , -7.04634110e-01],
                       [ 6.19696018e-10, 6.35823468e-10, 1.96710684e-09, 1.86948533e-09, 3.18085220e-09 ,  1.40805112e-09],
                       [ 3.25308169e-01, 1.98461324e-01,-7.04634112e-01, 3.78556444e-01,-4.57977608e-01 ,  7.04634112e-01],
                       [-1.03047355e-10,-1.71718413e-10, 5.74484710e-12, 2.22681713e-10,-3.29061150e-10 , -1.08682141e-10]])
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
  x = reverse_spin_block(x_s,n_mo,n_spinor)

  # orthonormal MO coefficients
  ortho_orb = LA.inv(x).dot(orb)

  return n_mo, n_spinor, enuc, h_ao, ao_2e, orb, x_small, x, ortho_orb
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

def form_rot_density(anga,angb,angg,p,nov,n_spinor,n_mo):
  ''' Form the rotation matrix R_g and overlap matrix N_g and
      the rotated density p_g '''

  R = np.zeros([n_spinor, n_spinor],dtype=complex)
  N = np.zeros([n_mo, n_mo],dtype=complex)

  # Form the rotation matrix in the OAO basis
  for i in range(0,n_spinor,2):
    R[i,i]     =  np.exp( 1j*(anga+angg)/2) * np.cos(angb/2)
    R[i,i+1]   =  np.exp( 1j*(anga-angg)/2) * np.sin(angb/2)
    R[i+1,i]   = -np.exp(-1j*(anga-angg)/2) * np.sin(angb/2)
    R[i+1,i+1] =  np.exp(-1j*(anga+angg)/2) * np.cos(angb/2)

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
  G_mx = np.transpose(x).dot(G_mx).dot(x)
  G_my = np.transpose(x).dot(G_my).dot(x)
  G_mz = np.transpose(x).dot(G_mz).dot(x)
  # transform to NO basis
  G = gather(G_scal,G_mx,G_my,G_mz,n_mo,n_spinor)
  G = np.transpose(np.conjugate(nov)).dot(G).dot(nov)

  # add one-electron part
  h_no = np.conjugate(np.transpose(nov)).dot(h).dot(nov)
  F    = h_no + G

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
def contract_w_ci(Xci,Fci,H,S,ncis,n_spinor,enuc,E_GHF):
  ''' Solve the CI problem and contract the Fci and Xci matrices with the
      resulting eigenvectors/eigenvalue to form the final effective Fock 
      matrix in the NO basis '''

  # diagonalize the overlap matrix to check if it is positivie definite
  w_S, vec_S = LA.eig(-S)
  thresh = 1.e-10
  for i in range(ncis):
    if (abs(w_S[i]) > thresh):
      vec_S[:,i] /= np.sqrt(-w_S[i])
    else:
      vec_S[:,i] *= 0
  vec_S *= -1j

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
  for m in range(ncis):
    for k in range(ncis):
      Xint += np.conjugate(ci[m]) * Xci[m,k] * ci[k]
      Fint += np.conjugate(ci[m]) * Fci[m,k] * ci[k]

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
  F_ortho = nov.dot(F_no).dot(np.transpose(np.conjugate(nov)))

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
def calculate_S2(ci):
  ''' Evaluate <S**2> for this density. 
      Note: this does a lot of redundant work and is not how it's 
            structured in Chronus '''

  # Loop over all grid points
  SSq = 0.
  for g in range(ngrdt):
    anga = grda[int(grid_index[g,0])]
    angb = grdb[int(grid_index[g,1])]
    angg = grdg[int(grid_index[g,2])]

    # form rotated matrices and other quantities in NO basis
    R, N, detN, p_g = form_rot_density(anga,angb,angg,p_no,nov,n_spinor,n_mo)
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

    # contract with the CI vector
    for m in range(ncis):
      for k in range(ncis):
        weight = xmk[g,m,k]/detN
        SSq += weight* np.conjugate(ci[m]) * SSqg * ci[k]

  print "\tPHF <S**2> = %.10f\n" % float(SSq.real*0.25)

#---------------------------------------

#---------------------------------------
if __name__ == '__main__':

  # load data from Chronus
  n_mo, n_spinor, enuc, h_ao, ao_2e, orb, x_small, x, ortho_orb = load_data()

  # Build the initial GHF density
  p = np.zeros([n_spinor,n_spinor],dtype=complex)
  for u in range(n_spinor):
    for v in range(n_spinor):
      for i in range(n_mo):
        p[u,v] += orb[u,i]*np.conj(orb[v,i])

  # setup the grid for integration 
  spin  = 0.5
  ncis  = int(2*spin + 1)
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
  xmk = collect_grid_weights(ngrdt,grid_index,grda,wgta,grdb,wgtb,grdg,wgtg,dmt,sz)

  # SCF optimization
  EOld, count, conv = 1., 0, False
  h_ortho = np.transpose(x).dot(h_ao).dot(x)
  while (count < 100 and not conv):

    # form OAO GHF quantities
    p       = np.zeros([n_spinor,n_spinor],dtype=complex)
    p_ortho = np.zeros([n_spinor,n_spinor],dtype=complex)
    for u in range(n_spinor):
      for v in range(n_spinor):
        for i in range(n_mo):
          p_ortho[u,v] += ortho_orb[u,i]*np.conj(ortho_orb[v,i])
          p[u,v] += orb[u,i]*np.conj(orb[v,i])
    G = build_G_matrix(ao_2e,p,n_spinor,n_mo)
    G_ortho = np.transpose(x).dot(G).dot(x)
    f_ortho = h_ortho + G_ortho
    E_GHF = calc_GHF_energy(h_ao,G,p)

    # transform density to NO basis
    # Note: negative sign ensures that occupied orbitals are first
    no_w, nov = LA.eigh(-p_ortho) 
    p_no = np.conjugate(np.transpose(nov)).dot(p_ortho).dot(nov)

    # loop over grid points and add contributions
    S   = np.zeros([ncis,ncis],dtype=complex)
    H   = np.zeros([ncis,ncis],dtype=complex)
    Xci = np.zeros([ncis,ncis,n_spinor,n_spinor],dtype=complex)
    Fci = np.zeros([ncis,ncis,n_spinor,n_spinor],dtype=complex)
    for g in range(ngrdt):
      anga = grda[int(grid_index[g,0])]
      angb = grdb[int(grid_index[g,1])]
      angg = grdg[int(grid_index[g,2])]

      # form rotated matrices and other quantities in NO basis
      R, N, detN, p_g = form_rot_density(anga,angb,angg,p_no,nov,n_spinor,n_mo)
      f_g, h_no = Fock_build(h_ortho,p_g,ao_2e,nov,x_small,n_spinor,n_mo)
      X_g, F_g, E_g = form_effective_fock(R,N,p_no,p_g,h_no,f_g,n_mo,n_spinor)

      # add contributions to S, H, Xci, and Fci
      for m in range(ncis):
        for k in range(ncis):
          weight = xmk[g,m,k]/detN
          S[m,k]   += weight
          H[m,k]   += weight * E_g
          Xci[m,k] += weight * X_g
          Fci[m,k] += weight * F_g

    # form fully integrated effective Fock matrix
    F_no, E, ci = contract_w_ci(Xci,Fci,H,S,ncis,n_spinor,enuc,E_GHF)
    F_ortho = modify_PHF_Fock(F_no,f_ortho,nov,n_mo,n_spinor)

    # diagonalize Fock matrix and calculate <S^2>
    w, ortho_orb = LA.eigh(F_ortho)
    orb = x.dot(ortho_orb)
    calculate_S2(ci)

    # check for convergence
    if abs(E-EOld) < 1.e-6:
      conv = True
    else:
      count += 1
      EOld = E

