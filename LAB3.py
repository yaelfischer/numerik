import numpy as np
import numpy.linalg as lin
import numpy.random as rnd
import matplotlib.pyplot as plt


# forwardSubs: vorwärtseinsetzen
# in:
#  - Matrix LR, die das Ergebnis einer LU-Zerlegung enthält
#  - Vektor b: rechte Seite des LGS
# out: Lösung x des LGS
def fbSubsT(LR, b):
    y = []

    y.append(b[0])
    for i in range(1,len(b)):
        y.append(b[i])
        y[i] = y[i] - (LR[0, i] * y[i-1])

    # Rückwärts
    x = np.zeros_like(y)
    x[-1] = y[-1] / LR[1, -1]

    for i in range(len(b)-2, -1, -1):
        x[i] = (1 / LR[1, i]) * (y[i] - (LR[1, i]*x[i] + LR[2, i]*x[i+1]))
    return x


# LU decomposition for tridiagonal matrix
# in: a  =  [[0,      a_{21}, ..., a_{n-1,n-2}, a_{n,n-1}],     # Lower tridiag
#            [a_{11}, a_{22}, ..., a_{n-1,n-1}, a_{nn}],        # Diag
#            [a_{12}, a_{23}, ..., a_{n-1,n},   0]]             # Upper tridiag
#
# out: LU = [[0,      l_{21}, ..., l_{n-1,n-2}, l_{n,n-1}],
#            [r_{11}, r_{22}, ..., r_{n-1,n-1}, r_{nn}],
#            [r_{12}, r_{23}, ..., r_{n-1,n},   0]
def LUT(A):         #----<<<--------<<<<
    m = A.shape[1]-1
    for k in range(0, m):
        A[0, k+1] = A[0, k+1] / A[1, k]
        A[1, k+1] = A[1, k+1] - A[0, k+1] * A[2, k]
    return A


# Lineares Gleichungssystem A*x = b lösen.
def linsolve(A, b):
    A = LU(A)
    x = fbSubs(A, b)
    #[A, idx] = LU_pivot(A)
    #x = fbSubs_pivot(A, b, idx)
    return x


# ============================================================================================ #


n = 7 # Grösse der Matrizen
# test LUT
for k in range(1000):
    m = np.random.rand(3,n)     # Zufällige Matrix M erzeugen
    m[0][0], m[-1][-1] = 0, 0   # nicht verwendete Einträge löschen
    A = np.diag(m[0][1:], k=-1) + np.diag(m[1], k=0) + np.diag(m[2][:-1], k=1) # volle Matrix A erzeugen (nur für Test)
    LU = LUT(m)
    L,U = np.diag(LU[0][1:] , k=-1)+ np.identity(n), np.diag(LU[1], k=0) + np.diag(LU[2][:-1], k=1) # L, U Matrizen
    assert(np.linalg.norm(L@U-A) < 1e-8)
print("Test LUT OK")


# test fbSubsT
for k in range(1000):
    m = np.random.rand(3,n)
    m[0][0], m[-1][-1] = 0, 0
    A = np.diag(m[0][1:], k=-1) + np.diag(m[1], k=0) + np.diag(m[2][:-1], k=1)

    x1 = np.random.rand(n,1)   # Lösungsvektor
    b = A@x1                   # rechte Seite des LGS

    LU = LUT(m)
    x2 = fbSubsT(LU, b)
    assert(np.linalg.norm(x1-x2) < 1e-10)
print("Test fbSubsT OK")


n = 100
x = np.linspace(0,1,n+1)
f = 1 # f(x) = 1 , gegeben
# Dirichlet Randwerte
u0 = 0
un = 0

# System Matrix
A = np.zeros((3,n-1))
A[0,1:] = -1
A[1,:] = 2
A[2,:-1] = -1

# Rechte Seite
h = 1./n
b = np.ones(n-1)*h**2 * f
b[0] = b[0] + u0
b[-1] = b[-1] + un

LU = LUT(A)
u = np.zeros((n+1))
u[0] = u0;   # Randwert links
u[-1] = un;  # Randwert rechts
u[1:-1] = fbSubsT(LU, b)

ue = -0.5*x*(x-1); # Loesung von u''(x) = 1, u(0) = u(1) = 0

plt.plot(x, u)
plt.plot(x, ue,'--')
plt.grid()
plt.show()


