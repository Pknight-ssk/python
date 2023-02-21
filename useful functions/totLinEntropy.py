# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:47:41 2022

@author: Pknight
"""

"""
totLinEntropy.py
--------------------------------------
with format reference to https://www.tensors.net/

to compute the total linear entropy of a state
"""


import numpy as np
from numpy import linalg as LA

# import cupy as cp  # to use GPU(CUDA) speedup and as np alternative

from numba import jit # to speedup for loops, use @numba.jit(nopython=True, parallel=True) as a wrapper before a function

@jit(nopython=True)
def totLinEntropy(psiIn: np.ndarray,
                  d = 2):
  """
  Args:
    psiIn: vector of length d**N describing the quantum state.
    d: local dimension of a site, default to be 2.
    
  Returns:
    tlentropy: the total linear entropy of a state
  """
  tlentropy = 0.
  # do normalization for psiIn
  psiIn = psiIn / LA.norm(psiIn)
  
  N = int(np.floor(np.log2(psiIn.size)))  # N: the number of lattice sites.
  
  if N != int(np.ceil(np.log2(psiIn.size))):
      raise ValueError("The size of input state should be d**N !")
  
  for i in range(N):
      tvec = psiIn.reshape(d**i, d, d**(N-1-i)).transpose(1, 0, 2)
      nvec = tvec.copy().reshape(d, d**(N-1))   # in case of a contiguous array copy, transpose make the array non-contiguous
      # nvec = nvec/LA.norm(nvec)  # equal to np.sqrt(sum(list(map(lambda x: LA.norm(x)**2, nvec[:]))))
      for j in range(d):
          for k in range(d):
              tlentropy += np.abs(np.vdot(nvec[j], nvec[k]))**2  # actually it's matrix 2-form
  tlentropy = N - tlentropy
  
  return tlentropy
              
      
    
    







































