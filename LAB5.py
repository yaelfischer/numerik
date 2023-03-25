import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.linalg import solve_triangular

def QR(A):
    [m, n] = np.shape(A)

    Q = np.identity(m)
    R = A.copy()

    for k in range(n):
        x = R[k:,k]
        lam = -np.sign(R[k,k]) * lin.norm(x)

        e1 = np.zeros(len(x))
        e1[0] = 1
        e1 = np.transpose(e1)

        v = (x-lam*e1)/(lin.norm(x-lam*e1))

        H = np.identity(m) - 2 * np.outer(np.concatenate((np.zeros(k), v)),np.concatenate((np.zeros(k), v)))

        Q = Q@H.copy()
        R = H@R

    return Q[:,:n], R[:n,:]


## A2
s0 = 100
x0 = 80.3e3

data = np.loadtxt('LAB5_data.txt')

#plt.subplot(2, 1, 1)
#plt.plot(data[:, 0], data[:, 1])

A = np.zeros((len(data[:, 0]), 5))
b = data[:, 1]

for index, line in enumerate(A):

    line[0] = 1
    line[1] = data[index, 0]
    line[2] = data[index, 0]**2
    line[3] = data[index, 0]**3
    line[4] = (data[index, 0]-x0)/(s0/2)
    line[4] = 1 / (1 + line[4]**2)

btrans = np.transpose(A.copy())@b
Atrans = np.transpose(A.copy())@A

alpha1 = lin.solve(Atrans.copy(),btrans)

L = lin.cholesky(Atrans)
y = solve_triangular(L.copy(),btrans, lower=True)
alpha2 = solve_triangular(np.transpose(L.copy()),y)

Q, R = lin.qr(Atrans)
alpha3 = solve_triangular(R,np.transpose(Q)@btrans)

Q, R = lin.qr(A)
alpha4 = solve_triangular(R,np.transpose(Q)@b)

Q1, R1 = QR(A)
alpha5 = solve_triangular(R1,np.transpose(Q1)@b)

x = data[:, 0]
xs = (x-x0)/(s0/2)
f1 = alpha1[0] + x*alpha1[1] + x**2*alpha1[2] + x**3*alpha1[3] + alpha1[4]*1/(1+xs**2)
f2 = alpha2[0] + x*alpha2[1] + x**2*alpha2[2] + x**3*alpha2[3] + alpha2[4]*1/(1+xs**2)
f3 = alpha3[0] + x*alpha3[1] + x**2*alpha3[2] + x**3*alpha3[3] + alpha3[4]*1/(1+xs**2)
f4 = alpha4[0] + x*alpha4[1] + x**2*alpha4[2] + x**3*alpha4[3] + alpha4[4]*1/(1+xs**2)
f5 = alpha5[0] + x*alpha5[1] + x**2*alpha5[2] + x**3*alpha5[3] + alpha5[4]*1/(1+xs**2)

err1 = sum((data[:,1]-f1)**2)
err2 = sum((data[:,1]-f2)**2)
err3 = sum((data[:,1]-f3)**2)
err4 = sum((data[:,1]-f4)**2)
err5 = sum((data[:,1]-f5)**2)

print("\nLinsolve Fehler: " + str(err1))
print("Cholesky Fehler: " + str(err2))
print("QRtrans Fehler: " + str(err3))
print("QR Fehler: " + str(err4))
print("QRimpl Fehler: " + str(err5))

print("\nCond A: " + str(lin.cond(A)))
print("Cond Atrans: " + str(lin.cond(Atrans)))
print("Cond L: " + str(lin.cond(L)))
print("Cond Q: " + str(lin.cond(Q)))
print("Cond R: " + str(lin.cond(R)))

plt.subplot(2,1,1)
plt.plot(x,data[:,1])
plt.plot(x, f1, x, f2, x, f3, x, f4, x, f5)
plt.legend(['Data', 'lin.solve', 'solve_triangular', 'QR_trans', 'QR', 'QRimpl'], loc='upper right')

plt.subplot(2,1,2)
f6 = alpha4[4]*1/(1+xs**2)
data2 = data[:,1] - (alpha4[0] + x*alpha4[1] + x**2*alpha4[2] + x**3*alpha4[3])
plt.plot(x, data2, x, f6)
plt.legend(['Data', 'QR'])
plt.show()

