import numpy as np
import os
import numpy.linalg as lin
import matplotlib.pyplot as plt


# Singulärwertzerlegung
# M = UE(V)T
# U: orthogonale (m x m)
# V: orthogonale (n x n)
# E: diagonale (m x n), diagonalelemente nicht neg.

# Geometrische Interpretation der SVD
script_dir = os.path.dirname(__file__)
rel_path = "Daten/datenPunkte.csv"
abs_file_path = os.path.join(script_dir, rel_path)

Q = np.loadtxt(abs_file_path, delimiter=",")

n = len(Q[:,0])

Xm = 1/n * sum(Q[:,0])
Ym = 1/n * sum(Q[:,1])

Q[:,0] = Q[:,0]-Xm
Q[:,1] = Q[:,1]-Ym

U, Sigma, V = lin.svd(Q)
sigma = np.diag(sigma)
M = U@sigma@np.transpose(V)
