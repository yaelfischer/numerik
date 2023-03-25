import numpy as np
import os
import numpy.linalg as lin
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular

script_dir = os.path.dirname(__file__)
rel_path = "Daten/datenPunkte.csv"
abs_file_path1 = os.path.join(script_dir, rel_path)

Q = np.loadtxt(abs_file_path1, delimiter=",")
#Q = np.array([[1,0],[0,1],[-1,0],[0,-1],[0.707,0.707],[-0.707,0.707],[0.707,-0.707],[-0.707,-0.707]])     # circular data -> SW are the same
points = np.copy(Q)
n = len(Q[:,0])

# center of cloud
Xm = 1/n * sum(Q[:,0])
Ym = 1/n * sum(Q[:,1])

# move center to origin
Q[:,0] = Q[:,0]-Xm
Q[:,1] = Q[:,1]-Ym

# SVD
U, Sigma, V = lin.svd(Q)
#Sigma = np.diag(Sigma)
#Sigma = np.pad(Sigma, [(0, n - len(Sigma)), (0, 0)], mode='constant')

# create SVD fit
line = np.zeros((2,2))
line[0,0] = Xm + (-2.5) * V[0,0]
line[0,1] = Ym + (-2.5) * V[0,1]
line[1,0] = Xm + 1.8 * V[0,0]
line[1,1] = Ym + 1.8 * V[0,1]

plt.scatter(points[:,0], points[:,1])
plt.plot(line[:,0], line[:,1], 'r')

# Ausgleichsgerade
b = points[:,1]
A = np.column_stack((np.ones(n), points[:,0]))
Q1, R1 = lin.qr(A)
alpha = solve_triangular(R1,np.transpose(Q1)@b)

line_x = np.array([-0.29, 3.25])
plt.plot(line_x,alpha[0]+line_x*alpha[1])
plt.quiver([Xm,Xm],[Ym, Ym],V[:,0],V[:,1], scale=1, units='xy', angles='xy', width=0.03)
plt.legend(['Data', 'SVD', 'Ausgleichsgerade', 'Rechts-SV'])
plt.gca().set_aspect('equal')
plt.grid()
plt.savefig('plt1.png')
plt.show()

# Aufgabe 4 Gesichtserkennung
numberTrainPers = 30
numberTrainPerPers = 5
sizeOfImage = 56 * 68
Q = np.zeros((0,sizeOfImage))

for j in range(1,numberTrainPers+1):
    for i in range(1,numberTrainPerPers+1):
        rel_path = ("Daten/Gesichter/s" + str(j) + "/f" + str(i) + ".png")
        abs_file_path2 = os.path.join(script_dir, rel_path)
        image = plt.imread(abs_file_path2)
        faceVec = np.concatenate(image)
        Q = np.append(Q, [faceVec], axis=0)

U, Sigma, V = lin.svd(Q)

# test gesicht f10
rel_path = ("Daten/Gesichter/s35/f10.png")
abs_file_path2 = os.path.join(script_dir, rel_path)
image = plt.imread(abs_file_path2)
testFaceVec = np.concatenate(image)     # <- = b

# reduce number of used singular values
keepSV = 1000
U = U[:,:keepSV]
Sigma = Sigma[:keepSV]
V = V[:,:keepSV]

Q, R = lin.qr(V)
alpha = solve_triangular(R,np.transpose(Q)@testFaceVec)

# reconstrucion of face
reconst = V@alpha
test = np.array(np.split(reconst,68))

plt.imshow(test, cmap='gray')
plt.show()

