# test for GHMM
# Paul Riechers code
# C. Hillar 2024
#

import numpy as np
from ghmm import GHMM

def mag(vector):
    """
    Returns the magnitude of the vector in Euclidean norm
    """
    return np.sqrt(vector @ vector)

def hat(vector):
    """
    Returns vector scaled to unit length in Euclidean norm
    """
    return vector / mag(vector)
    
def orthonormal_basis_orthog_to(vector):
    """
    Given an N-dimensional vector v,
      constructs and returns an (N-1)-dimensional orthonormal basis 
      orthogonal to v, via something like Gram-Schmidt 
    I.e., returns N-1 orthonormal vectors, each of dimension N, 
      and each orthogonal to the input vector
    """
    d = len(vector)  # dimension of the original vector space
    # assume old_basis is one-hot in the computational basis of \mathbb{R}^d
    old_basis = []
    for index in range(d):
        basis_vec = np.zeros(d)
        basis_vec[index] = 1.
        old_basis.append(basis_vec.copy())
    new_basis = [hat(vector), ]
    for index in range(d-1):
        new_vec = old_basis[index]
        for vec in new_basis:
            overlap = vec @ new_vec
            new_vec -= overlap*vec
        if mag(new_vec) > 0.00001:
            new_basis.append(hat(new_vec))
        else:
            # if we are here, it is probably because vector is parallel to an old basis vector
            new_vec = old_basis[index+1]
            for vec in new_basis:
                overlap = vec @ new_vec
                new_vec -= overlap*vec
            new_basis.append(hat(new_vec))
    return new_basis[1:]
    
def project_vec_to_subspace(vec, orthonormal_basis):
    # assumes orthonormal_basis is list-like, containing orthonormal basis vectors
    # returns the lower-dimensional vector, projected onto these basis elements
    return np.array([vec @ basis_vec for basis_vec in orthonormal_basis])

def project_3vec_to_2simplex(vec):
    return project_vec_to_subspace(vec, [hat(np.array([-1,1,0])), hat(np.array([-1,-1,2]))])


def make_tokens(L=1000, num_samples=3, x=0.05, alpha = 0.08, beta=None):
    """ returns num_samples time-series each of length L for test hmm """
    if beta is None:
        beta = (1-alpha)/2; y = 1-2*x

    TA = np.array([[alpha*y, beta*x, beta*x],[alpha*x, beta*y, beta*x],[alpha*x, beta*x, beta*y]])
    TB = np.array([[beta*y, alpha*x, beta*x],[beta*x, alpha*y, beta*x],[beta*x, alpha*x, beta*y]])
    TC = np.array([[beta*y, beta*x, alpha*x],[beta*x, beta*y, alpha*x],[beta*x, beta*x, alpha*y]])

    dict_labeledTMs = dict()
    dict_tokens = dict()
    dict_tokens[0] = 'A'
    dict_tokens[1] = 'B'
    dict_tokens[2] = 'C'
    dict_labeledTMs[0] = TA
    dict_labeledTMs[1] = TB
    dict_labeledTMs[2] = TC

    mess3 = GHMM(dict_labeledTMs, dict_tokens)

    xlist = []
    ylist = []
    distr_list = []  # only keeping around for rgb colors

    tokens = [[] for i in range(num_samples)]

    for i in range(num_samples):
        for ell in range(L):
            vec = mess3.current_distr
            distr_list.append(vec)
            X, Y = project_3vec_to_2simplex(vec)
            xlist.append(X); ylist.append(Y)
            mess3.sample()
        labels = mess3.yield_emissions(L)
        c = 0
        for x in labels:
            if x == 'A':
                tokens[i].append([1, 0, 0]) 
            if x == 'B':
                tokens[i].append([0, 1, 0]) 
            if x == 'C':
                tokens[i].append([0, 0, 1]) 
            c += 1

    return tokens