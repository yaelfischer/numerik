import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy as sy
import scipy.sparse.linalg
import sympy as sp


# <editor-fold desc="Aufgabe 2 Lineare Ausgleichsrechnung">
#%%
data = np.array([[1, 1.5, 2, 2.5],[3.7, 3, 1.5, 1.8]])

N = len(data[0,:])
unknown = 3

A = np.zeros((N,unknown))
b = data[1,:]

# y = a1*sin(2x) + a2*ln(x^2) + 0.5*a3
for index, line in enumerate(A):
    t = data[0, index]
    line[0] = np.sin(2*t)
    line[1] = np.log(t**2)
    line[2] = 0.5

print("Überbestimmtes Gleichungssystem: Ax=b")
print("A: ")
print(A)
print("b: ")
print(b)

# Normalgleichungen
btrans = np.transpose(A.copy())@b
Atrans = np.transpose(A.copy())@A

print("\nNormalgleichung:")
print("A.T@A: ")
print(Atrans)
print("A.T@b: ")
print(btrans)
print(" ")

QQ, RR = np.linalg.qr(Atrans)   # oder nur A

print("\nQR Zerlegung")
print("Q: ")
print(QQ)
print("R: ")
print(RR)

alpha = la.solve_triangular(RR,np.transpose(QQ)@btrans) # oder nur b
print("\nAlphas:" + str(alpha))

print("\nResiduenquadratsumme: ")
print(la.norm(b-A@alpha)**2)

xausgl = np.linspace(1, 2.5, 401)
yausgl = alpha[0]*np.sin(2*xausgl) + alpha[1]*np.log(xausgl**2) + alpha[2]*0.5
plt.scatter(data[0,:], data[1,:])
plt.plot(xausgl, yausgl)
plt.show()
#%%
# </editor-fold>

# <editor-fold desc="Aufgabe 4 Runge Kutta">
#%%
def ImpRK(anfangs, X, N, f, but):
    counter = 0
    s = len(but[:,0])-1
    [x0, y0] = anfangs

    h = (X-x0) / N
    x = []
    y = []
    x.append(x0)
    y.append(np.array(y0))

    k = np.zeros((len(y0),s))
    for i in range(N):
        x.append(x[i] + h)
        for j in range(s):
            k[:,j] = sy.optimize.fsolve(lambda imp: f(x[i]+but[j,0]*h,y[i]+h*but[j,1:j+1]@np.transpose(k[:,0:j])+h*but[j,j+1]*imp)-imp, np.array([k[:,j-1]]), xtol=1.49012e-12)
            counter += 1

        y.append(y[i] + h*np.dot(k,but[-1,1:]))

    return np.array(x),np.array(y), counter


heun = [[0, 0,   0],
        [1, 1,   0],
        [0, 0.5, 0.5]]
heun = np.array(heun)


# y' = (0            1) Y       y(0) = (1,-1)
#      (-2(2x^2+1) -4x)
def f(x,y):
    return (np.sin(x)+y)/y


X = 3
N = [20,200,2000]
start = [-1,[-2]]
err1 = []
for ns in N:
    x1, y1, _ = ImpRK(start, X, ns, f, heun)
    err1.append(np.linalg.norm(y1[-1] - (-0.5729868923)))
print("Fehler Exp: " + str(err1))

startLog = 10.0
stopLog = 10000.0


plt.loglog(N,err1)
plt.loglog(np.arange(startLog,stopLog,1), 1/np.arange(startLog,stopLog,1), linewidth=0.5, color='black')
plt.loglog(np.arange(startLog,stopLog,1), 1/np.arange(startLog,stopLog,1)**2, linewidth=0.5, color='black', linestyle='dashed')
plt.loglog(np.arange(startLog,stopLog,1), 1/np.arange(startLog,stopLog,1)**3, linewidth=0.5, color='black', linestyle='dotted')
plt.loglog(np.arange(startLog,stopLog,1), 1/np.arange(startLog,stopLog,1)**4, linewidth=0.5, color='black', linestyle='dashdot')


plt.legend(['Heun', 'O1', 'O2', 'O3', 'O4'])
plt.show()
#%%
# </editor-fold>

# <editor-fold desc="Aufgabe 5 Runge Kutta System">
#%%
def ImpRK(anfangs, X, N, f, but):
    counter = 0
    s = len(but[:,0])-1
    [x0, y0] = anfangs

    h = (X-x0) / N
    x = []
    y = []
    x.append(x0)
    y.append(np.array(y0))

    k = np.zeros((len(y0),s))
    for i in range(N):
        x.append(x[i] + h)
        for j in range(s):
            k[:,j] = sy.optimize.fsolve(lambda imp: f(x[i]+but[j,0]*h,y[i]+h*but[j,1:j+1]@np.transpose(k[:,0:j])+h*but[j,j+1]*imp)-imp, np.array([k[:,j-1]]), xtol=1e-12)
            counter += 1

        y.append(y[i] + h*np.dot(k,but[-1,1:]))

    return np.array(x),np.array(y), counter


mittel = [[0.5, 0.5],
          [0, 1]]
mittel = np.array(mittel)


# y' = (0            1) Y       y(0) = (1,-1)
#      (-2(2x^2+1) -4x)
def f(x,y):
    return np.array([y[1],y[1]**2+y[0]*(y[0]-1)-1])


X = 2
N = 20
start = [0,[1,0]]
err1 = []
err2 = []
x1, y1, _ = ImpRK(start, X, N, f, mittel)
err1.append(max(abs(y1[:,0] - np.cos(x1))))
err2.append(max(abs(y1[:,1] - (-np.sin(x1)))))
print("Fehler y: " + str(err1))
print("Fehler y': " + str(err2))
#%%
# </editor-fold>

# <editor-fold desc="Aufgabe 6 Newton, nicht lin. Gleichungssysteme">
#%%
def newton(func, vars, start, tol=1e-10, nMax = 200, Df = None):   # Startwert, Fehlertoleranz, max Iteratione

    if not callable(Df):
        jac = func.jacobian(vars)
        Df = sp.lambdify(vars, jac)
    f = sp.lambdify(vars, func)

    i = 1

    k = 0
    x = start
    y = f(*x)
    r = np.linalg.norm(y)   # Startwerte

    while r>tol and k<nMax:
        J = Df(*x)
        dx = np.linalg.solve(J, -y)
        k = k+1
        x = (x+np.transpose(dx))[0,:]
        if(i<=3):
            print("Schritt " + str(i) + ": " + str(x))
        y = f(*x)
        r = np.linalg.norm(y)
        i += 1
    #print("Newtoniterationen: " + str(k))
    if k>=nMax:
        print("No zero found")
        return np.ones((1,2))
    print("Anzahl Schritte: " + str(i))
    return x


i1, i2, ff = sp.symbols('i1,i2,ff')
vars = [i1,i2]

r1 = 100
r2 = 20
r3 = 50
u0 = 5

ff1 = sp.Matrix([i2*r3+0.2*sp.ln(1+i1/0.01)+i1*r1-u0,
                 i2*r3+(i2-i1)*r2-u0])

res = newton(ff1, vars, np.array([0.25,0.25]), tol=10**(-9))
print(str(res) + "\n")

print(0.2*np.log(1+res[0]/0.01))

#%%
# </editor-fold>

