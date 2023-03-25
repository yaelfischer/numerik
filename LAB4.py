import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular

####### Aufgabe 1 ########

data = np.genfromtxt('LAB4_data.txt')
print("")
print("Aufgabe 1:  ")
print("Sample time: " + str((data[1,0]) - data[0,0]) + " s")
print("Grundfrequenz geschätzt: 0.166 Hz")
print("Dimension Systemmatrix von fn(t): 1501x(2n+1)")

n = 50
w = 1
A = np.zeros((len(data),2*n+1))
b = data[:,1].copy()
b = np.transpose(b)


#Systemmatrix erstellen

for tx, line in enumerate(A):
    line[0] = 1/2
    for k in range(1, len(line), 2):
        line[k] = np.cos(w*k*data[tx,0])
        line[k+1] = np.sin(w*k*data[tx,0])


# Solve

print("Dimension Normalgleichung: " + str(A.shape))
b = np.transpose(A)@b
A = np.transpose(A)@A
L = cholesky(A)
y = solve_triangular(L.copy(),b)
alpha = solve_triangular(np.transpose(L.copy()),y)

f_x = np.zeros(len(data[:,0]))


# Funktion rekonstruieren

for k,time in enumerate(data[:,0]):
    f_x[k] = 0.5*alpha[0]
    for i in range(1, len(alpha),2):
        f_x[k] += np.cos(w*i*time)*alpha[i]
        f_x[k] += np.sin(w*i*time)*alpha[i+1]


# Plot

plt.subplot(3,1,1)
plt.plot(data[:,0],data[:,1])
plt.plot(data[:,0],f_x)
plt.xlabel('t[s]')
plt.ylabel('U[V]')
plt.title('Data')
plt.legend(['Data', 'Calculation'])
plt.grid()


# FFT

N = len(f_x)
test = data[:,1]
X_f = scipy.fft.fft(test)[0:350]
X_f = abs(X_f) / N
X_f_calc = scipy.fft.fft(f_x)[0:350]
X_f_calc = abs(X_f_calc) / N
f = scipy.fft.fftfreq(N, 0.01)[0:350]


# error

err = sum((data[:,1] - f_x)**2)
print("Error: " + str(err))


# Plot FFT

plt.subplot(3,1,2)
plt.plot(f,20*np.log10(X_f))
plt.plot(f,20*np.log10(X_f_calc), ':')
plt.xlabel('f[Hz]')
plt.ylabel('A')
plt.title('FFT')
plt.legend(['Data', 'Calculation'])
plt.grid()


####### Aufgabe 2 ########

t2 = np.array([0.0, 0.03, 0.05, 0.08, 0.1])
u2 = np.array([5.0, 2.94, 1.73, 1.01, 0.6])
RC = 100


## f linearisiert: a + b * -((1/R) * t)

A = np.zeros((len(t2),2))

for tx, line in enumerate(A):
    line[0] = 1
    line[1] = -(1/RC)*t2[tx]

b = np.log(u2.copy())
b = np.transpose(A.copy())@b
L = np.transpose(A.copy())@A

alpha = np.linalg.solve(L.copy(),b)

Ctest = 1/alpha[1]
U0test = np.exp(alpha[0])

u2test = U0test*(np.exp(-t2/(RC*Ctest)))

print("")
print("Aufgabe 2:  ")
print("Diffgleichung: R*C*y' + y = Ue")
print("Lösung: Uc = U0*e^-t/R*C")
print("Kapazität: " + str(Ctest * 10**6) + " uF")
print("U0: " + str(U0test) + " V")
print("Systemmatrix:")
print(A)


plt.subplot(3,1,3)
plt.plot(t2,u2)
plt.plot(t2,u2test)
plt.xlabel('t[s]')
plt.ylabel('U[V]')
plt.title('Data')
plt.legend(['Data', 'Calculation'])
plt.grid()
plt.show()
