import numpy as np
import numpy.linalg as lin
import numpy.random as rnd


# random orthogonal matrix
def rndOrtho(n):
    S = rnd.rand(n, n)
    S = S - S.T
    O = lin.solve(S - np.identity(n), S + np.identity(n))
    return O


# random matrix with specified condition number
def rndCond(n, cond):
    d = np.logspace(-np.log10(cond) / 2, np.log10(cond) / 2, n)
    A = np.diag(d)
    U, V = rndOrtho(n), rndOrtho(n)
    return U @ A @ V.T


def absmaxND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)


# forwardSubs: vorwärtseinsetzen
# in:
#  - Matrix LR, die das Ergebnis einer LU-Zerlegung enthält
#  - Vektor b: rechte Seite des LGS
# out: Lösung x des LGS
def fbSubs(LR, b):
    y = []

    for i in range(len(b)):
        y.append(b[i])
        for j in range(i):
            y[i] = y[i] - (LR[i, j] * y[j])

    # Rückwärts
    x = np.zeros_like(y)
    x[-1] = y[-1] / LR[-1, -1]

    for i in range(len(b), 0, -1):
        x[i - 1] = (1 / LR[i - 1, i - 1]) * (y[i - 1] - np.dot(LR[i - 1, i:], x[i:]))
    return x


def fbSubs_pivot(LR, b, P):
    y = []

    bcpy = b.copy()
    for i in range(len(P)):
        bcpy[i] = b[P[i]]
    b = bcpy.copy()

    for i in range(len(b)):
        y.append(b[i])
        for j in range(i):
            y[i] = y[i] - (LR[i, j] * y[j])

    # Rückwärts
    x = np.zeros_like(y)
    x[-1] = y[-1] / LR[-1, -1]

    for i in range(len(b), 0, -1):
        x[i - 1] = (1 / LR[i - 1, i - 1]) * (y[i - 1] - np.dot(LR[i - 1, i:], x[i:]))
    return x


# LU-Zerlegung der quadratischen Matrix A
# in: quadratische Matrix A
# out:
# - A wird überschrieben, unteres Dreieck = L (ohne Diagonale), oberes Dreieck = R
# - idx: Indexvektor der Zeilenvertauschungen
def LU(A):
    m = A.shape[0]
    idx = np.array(range(m))
    for k in range(0, m):
        for i in range(k + 1, m):
            A[i, k] = A[i, k] / A[k, k]
            for j in range(k + 1, m):
                A[i, j] = A[i, j] - A[i, k] * A[k, j]
    return A, idx


def LU_pivot(A):
    m = A.shape[0]
    idx = np.array(range(m))

    for k in range(0, m):
        # pivot: largest abs
        B = A[:, k].tolist()
        abso = absmaxND(A[:, k][k:])
        index = B.index(abso)

        tmp = idx[k].copy()
        idx[k] = idx[index]
        idx[index] = tmp

        tmp = A[k, :].copy()
        A[k, :] = A[index, :]
        A[index, :] = tmp

        for i in range(k + 1, m):
            A[i, k] = A[i, k] / A[k, k]
            for j in range(k + 1, m):
                A[i, j] = A[i, j] - A[i, k] * A[k, j]
    return A, idx


# Lineares Gleichungssystem A*x = b lösen.
def linsolve(A, b):
    [A, idx] = LU(A)
    x = fbSubs(A, b)
    #[A, idx] = LU_pivot(A)
    #x = fbSubs_pivot(A, b, idx)
    return x


def linsolve_pivot(A, b):
    [A, idx] = LU_pivot(A)
    x = fbSubs_pivot(A, b, idx)
    return x


# ============================================================================================ #


# Test fbSubs
n = 7  # Dimension der Koeffizientenmatrix
for k in range(1000):  # Testläufe
    LR = np.array(np.random.rand(n, n))  # zufällige Matrix LR
    rhs = np.array(np.random.rand(n))  # zufällige rechte Seite des LGS
    x = fbSubs(LR, rhs)  # Aufruf der eigenen Funktion
    L, R = np.tril(LR, -1) + np.identity(n), np.triu(LR)  # L und R extrahieren
    assert (np.linalg.norm(rhs - L @ R @ x) < 1e-8)  # Test, mit numerischer Toleranz
print("Test fbSubs OK")


# Test LU
n = 7
for k in range(1000):
    A = np.array(np.random.rand(n, n))  # zufällige Matrix A erzeugen
    LR, idx = LU(A.copy())  # LU-Zerlegung von A
    L, R = np.tril(LR, -1) + np.identity(n), np.triu(LR)  # Matrizen L, R extrahieren
    assert (np.linalg.norm(L @ R - A[idx]) < 1e-8)
print("Test LU OK")


# Test linsolve
n = 7
for k in range(1000):
    A = np.random.rand(n, n)
    rhs = np.random.rand(n)

    x = linsolve(np.copy(A), rhs)
    x_piv = linsolve_pivot(np.copy(A), rhs)

    assert (lin.norm(rhs - A @ x) < 1e-8)
    assert (lin.norm(rhs - A @ x_piv) < 1e-8)
print("Test linsolve OK")


# Test genauigkeit
n = 7
meanerr = 0.0
meanerr_piv = 0.0
for k in range(1000):
    A = rndCond(n, 1e14)
    rhs = np.random.rand(n)

    sol = linsolve(np.copy(A), rhs)
    error = (sol - lin.solve(np.copy(A), rhs)) / sol
    sol_piv = linsolve_pivot(np.copy(A), rhs)
    error_piv = (sol_piv - lin.solve(np.copy(A), rhs)) / sol_piv
    meanerr += abs(error)
    meanerr_piv += abs(error_piv)

meanerr /= 1000
meanerr_piv /= 1000

print("Mean error: " + str(round(lin.norm(meanerr)*100, 4)) + " %")
print("Mean error_piv: " + str(round(lin.norm(meanerr_piv)*100, 4)) + " %")
