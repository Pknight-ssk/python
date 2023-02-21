# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:52:07 2022

@author: Admin
"""

"""
mainExactDiag.py
---------------------------------------------------------------------
Script file for initializing exact diagonalization using the 'eigsh' routine
for a 1D quantum system.

by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 06/2020
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from timeit import default_timer as timer

from totLinEntropy import totLinEntropy    # compute total linear entropy
from doApplyHam import doApplyHam

# Simulation parameters
model = 'ising'  # select 'XX' model or 'ising' model
Nsites = 7  # number of lattice sites  
# for XX/XY model with odd Nsites, there's degenerency in ground state, so the (IRLM) Lanczos method may not be accurate?
# for even Nsites there's no degenerency in ground state (?)
usePBC = False  # use periodic or open boundaries
numval = 12  # number of eigenstates to compute

# Define Hamiltonian (quantum XX model)
d = 2  # local dimension
sX = np.array([[0, 1.0], [1.0, 0]])
sY = np.array([[0, -1.0j], [1.0j, 0]])
sZ = np.array([[1.0, 0], [0, -1.0]])
sI = np.array([[1.0, 0], [0, 1.0]])
if model == 'XX':
  hloc = (np.kron(sX, sX) + np.kron(sY, sY)).reshape(2, 2, 2, 2)
  EnExact = -4 / np.sin(np.pi / Nsites)  # Note: only for PBC
elif model == 'ising':
  hloc = (-np.kron(sX, sX) + 0.5 * np.kron(sZ, sI) + 0.5 * np.kron(sI, sZ)
          ).reshape(2, 2, 2, 2)
  EnExact = -2 / np.sin(np.pi / (2 * Nsites))  # Note: only for PBC


# cast the Hamiltonian 'H' as a linear operator
def doApplyHamClosed(psiIn):
  return doApplyHam(psiIn, hloc, Nsites, usePBC)


H = LinearOperator((2**Nsites, 2**Nsites), matvec=doApplyHamClosed)

# do the exact diag
# start_time = timer()
Energy, psi = eigsh(H, k=numval, which='SA')

# tle = totLinEntropy(psi)/Nsites  # modified totLinearEntropy: without size effect

# diag_time = timer() - start_time

# check with exact energy
# EnErr = Energy[0] - EnExact  # should equal to zero

# print('NumSites: %d, Time: %1.2f, Energy: %e, EnErr: %e, totLinearEntropy: %1.2f' %
#       (Nsites, diag_time, Energy[0], EnErr, tle))


# useful_data = psi[np.abs(psi) > 1e-3]
print(psi)
print(Energy)
# with open('log.txt','a',newline='\n') as f:
#     f.write(str(useful_data) + "\n")







