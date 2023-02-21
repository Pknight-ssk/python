# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:06:34 2022

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
import matplotlib.pyplot as plt

from scipy.sparse.linalg import LinearOperator, eigsh

from totLinEntropy import totLinEntropy    # compute total linear entropy
from doApplyHam import doApplyHam

def exactDiag(Nsites, model = 'XX'):
# Simulation parameters
  # select 'XX' model or 'ising' model
  # number of lattice sites
  usePBC = False  # use periodic or open boundaries
  numval = 1  # number of eigenstates to compute

# Define Hamiltonian (quantum XX model)
  # d = 2  # local dimension
  sX = np.array([[0, 1.0], [1.0, 0]])
  sY = np.array([[0, -1.0j], [1.0j, 0]])
  sZ = np.array([[1.0, 0], [0, -1.0]])
  sI = np.array([[1.0, 0], [0, 1.0]])
  if model == 'XX':
      hloc = (np.real(np.kron(sX, sX) + 0.5 * np.kron(sY, sY))).reshape(2, 2, 2, 2)
  elif model == 'ising':
      hloc = (-np.kron(sX, sX) + 0.5 * np.kron(sZ, sI) + 0.5 * np.kron(sI, sZ)
             ).reshape(2, 2, 2, 2)

# cast the Hamiltonian 'H' as a linear operator
  def doApplyHamClosed(psiIn):
      return doApplyHam(psiIn, hloc, Nsites, usePBC)

  H = LinearOperator((2**Nsites, 2**Nsites), matvec=doApplyHamClosed)

# do the exact diag
  Energy, psi = eigsh(H, k=numval, which='SA')  # SA means smallest algebraic value
  tle = totLinEntropy(psi)/Nsites  # modified totLinearEntropy: without size effect
  
  return tle


tledata = list(map(exactDiag, np.arange(2, 16, 2).astype(int)))


plt.plot(np.arange(2, 16, 2), tledata)
plt.show()

# with open('log.txt','a',newline='\n') as f:
#     f.write(str(tledata) + "\n")

    






