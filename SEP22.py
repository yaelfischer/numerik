import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy as sy
import scipy.sparse.linalg
import sympy as sp


# <editor-fold desc="Aufgabe 2a Lineare Ausgleichsrechnung mit NORMALGLEICHUNGEN Aufgabe 2b Spline-Interpolation">
#%%

data = np.array([[0e-3,5e-3,10e-3,15e-3,20e-3],[-1.5, 1.5, 2.5, -0.5, -1.5]])

N = len(data[0,:])
unknown = 3

T = 20e-3

A = np.zeros((N,unknown))
b = data[1,:]

for index, line in enumerate(A):
    t = data[0, index]
    line[0] = 1
    line[1] = np.cos(((2*np.pi)/T)*t)
    line[2] = np.sin(((2*np.pi)/T)*t)

print(A)
print(" ")

# Normalgleichungen
btrans = np.transpose(A.copy())@b
Atrans = np.transpose(A.copy())@A

QQ, RR = np.linalg.qr(Atrans)

print(QQ)
print(" ")
print(RR)

alpha = la.solve_triangular(RR,np.transpose(QQ)@btrans)
print("y = " + str(btrans))
print("Alphas:" + str(alpha))

xausgl = np.linspace(0, 20e-3, 401)
yausgl = alpha[0] + alpha[1]*np.cos(((2*np.pi)/T)*xausgl) + alpha[2]*np.sin(((2*np.pi)/T)*xausgl)
plt.plot(xausgl, yausgl)


data = np.array([[0e-3,5e-3,10e-3,15e-3,20e-3],[-1.5, 1.5, 2.5, -0.5, -1.5]])
N = len(data[0,:])


def SplinePer(points):
    # ind = np.lexsort((points[:,1],points[:,0]))
    # points = points[ind]
    n = len(points[0,:])
    h = points[0,1]-points[0,0]

    A = np.zeros((n-2,n-2))
    b = np.zeros((n-2,1))
    for i in range(n-2):
        A[i,i] = 16

        if i != 0:
            A[i,i-1] = 4
        if i != n-3:
            A[i,i+1] = 4

        b[i] = (4*points[1,i]-8*points[1,i+1]+4*points[1,i+2])/(h**2)

    b[0] = (5*points[1,0]-9*points[1,1]+4*points[1,2])/h**2 + (points[1,-1]-points[1,-2])/h**2
    b[-1] = (4*points[1,-3]-9*points[1,-2]+5*points[1,-1])/h**2 - (points[1,1]-points[1,0])/h**2

    A[0,-1] = -1
    A[-1,0] = -1
    A[0,0] = 15
    A[-1,-1] = 15


    M = np.linalg.solve(A,6*b)
    M = M[:,0].tolist()
    M.append((3/(2*h))*((points[1,1]-points[1,0])/h-(points[1,-1]-points[1,-2])/h)-(h*M[0]/6)-(h*M[-1]/6))
    M.insert(0,M[-1])
    M = np.array(M)

    #np.concatenate(([[0]], M, [[0]]), axis=0)

    c = np.zeros((n-1))
    d = np.zeros((n-1))
    for i in range(n-1):
        c[i] = (points[1,i+1]-points[1,i])/h - (h/6)*(M[i+1]-M[i])
        d[i] = points[1,i]-((h**2)/6)*M[i]

    xss = []
    ys = []
    for i in range(n-1):
        x = getSupports(100,points[0,i],points[0,i+1])
        for xs in x:
            xss.append(xs)
            ys.append(((points[0,i+1]-xs)**3)/(6*h)*M[i]+((xs-points[0,i])**3)/(6*h)*M[i+1]+c[i]*(xs-points[0,i])+d[i])
    ys.append(((points[0,n-1]-points[0,-1])**3)/(6*h)*M[n-1]+((points[0,-1]-points[0,n-2])**3)/(6*h)*M[n-1]+c[n-2]*(points[0,-1]-points[0,n-2])+d[n-2])
    xss.append(points[0,-1])
    return xss,ys


def getSupports(number, start = -5, stop = 5):
    stepSize = (stop-start) / (number+1)
    x = np.arange(start, stop - stepSize, stepSize)
    return x

# xx, yy = SplinePer(np.array([x,y]))
xx, yy = SplinePer(data)
plt.scatter(data[0,:], data[1,:])
plt.plot(xx,yy)
plt.legend(['Ausgleich', 'Data',  'Spline'])
plt.show()

error = abs(yausgl - yy)
plt.plot(xx, error)
plt.show()
print(max(error))

#%%
# </editor-fold>

# <editor-fold desc="Aufgabe 3 Finite-Differenzen-Methode (Lineares Randwertproblem)">
#%%
def DifferenzenMethode(N):
    # Dirichlet Randwerte
    u0 = 0
    un = 3 / np.e

    Xend = 1
    h = Xend / (N + 1)
    x = np.linspace(u0 + h, Xend - h, N)

    # System Matrix
    # y'' +4y' +8y = 0
    const = -12 * x * np.exp(-x ** 2)
    Css = 1
    Cs = 2 * x
    C = 0

    A = np.zeros((3, N))
    b = np.ones((N, 1))
    b = const

    b[0] = b[0] - (Css / h ** 2 - Cs[0] / (2 * h)) * u0
    b[-1] = b[-1] - (Css / h ** 2 + Cs[-1] / (2 * h)) * un

    A[1] = C - (2 * Css) / (h ** 2)
    A[2] = Css / h ** 2 - Cs / (2 * h)
    A[0] = Css / h ** 2 + Cs / (2 * h)

    A[0] = np.roll(A[0],1)
    A[2] = np.roll(A[2],-1)

    print(A)
    print(" ")
    print(b)

    y = la.solve_banded((1, 1), A, b)

    return x, y


def f(x):
    return np.exp(-x ** 2) * 3 * x


x, y = DifferenzenMethode(3)
analytic = f(x)
print(x)
print(y)
print(analytic)
plt.plot(x, y)
plt.show()
err = np.abs(y - analytic)
print(err)
#%%
# </editor-fold>

# <editor-fold desc="Aufgabe 4 Implizit Runge Kutta">
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


Crouzeix = [[0.5+np.sqrt(3)/6, 0.5+np.sqrt(3)/6, 0],
            [0.5-np.sqrt(3)/6, -np.sqrt(3)/3, 0.5+np.sqrt(3)/6],
            [0, 0.5, 0.5]]
Crouzeix = np.array(Crouzeix)


# y' = (0            1) Y       y(0) = (1,-1)
#      (-2(2x^2+1) -4x)
def f(x,y):
    return -1/(2*y)-x/y+y


X = 3
N = [30,300,3000]
start = [0,[1]]
err1 = []
for ns in N:
    x1, y1, _ = ImpRK(start, X, ns, f, Crouzeix)
    err1.append(np.linalg.norm(y1[-1] - 2))
print("Fehler Exp: " + str(err1))

startLog = 10.0
stopLog = 10000.0


plt.loglog(N,err1)
plt.loglog(np.arange(startLog,stopLog,1), 1/np.arange(startLog,stopLog,1), linewidth=0.5, color='black')
plt.loglog(np.arange(startLog,stopLog,1), 1/np.arange(startLog,stopLog,1)**2, linewidth=0.5, color='black', linestyle='dashed')
plt.loglog(np.arange(startLog,stopLog,1), 1/np.arange(startLog,stopLog,1)**3, linewidth=0.5, color='black', linestyle='dotted')
plt.loglog(np.arange(startLog,stopLog,1), 1/np.arange(startLog,stopLog,1)**4, linewidth=0.5, color='black', linestyle='dashdot')


plt.legend(['Cruzy', 'O1', 'O2', 'O3', 'O4'])
plt.show()

#%%
# </editor-fold>

# <editor-fold desc="Aufgabe 5 Implizit Runge Kutta (System)">
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


RK4 = [[0.0, 0.0, 0.0, 0.0, 0.0],
       [0.5, 0.5, 0.0, 0.0, 0.0],
       [0.5, 0.0, 0.5, 0.0, 0.0],
       [1.0, 0.0, 0.0, 1.0, 0.0],
       [0,   1/6, 1/3, 1/3, 1/6]]
RK4 = np.array(RK4)


# y' = (0            1) Y       y(0) = (1,-1)
#      (-2(2x^2+1) -4x)
def f(x,y):
    return np.array([y[1],y[1]**2+4*y[0]*(y[0]-2)+2])


def exakt(x):
    return np.array([np.cos(x)**2,-2*np.sin(x)*np.cos(x)])   #y und y' exakt

X = 2
N = [20]
start = [0,[1,0]]
err1 = []
err2 = []
for ns in N:
    x1, y1, _ = ImpRK(start, X, ns, f, RK4)
    err1.append(max(y1[:,0] - exakt(x1)[0]))
    err2.append(max(y1[:,1] - exakt(x1)[1]))
plt.plot(x1,y1)
plt.show()
print("Fehler Exp: " + str(err1))
print("Fehler Exp: " + str(err2))

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


x1, x2, ff = sp.symbols('x1,x2,ff')
vars = [x1,x2]

a0 = 0.007
b0 = 0.003
c0 = 0
d0 = 0

k1 = 1.2e4
k2 = 7.0e4

ff1 = sp.Matrix([k1*x1**2 + k1*x1*x2 - (1+k1*(a0+b0))*x1 + (1-k1*b0)*x2 + k1*a0*b0 - c0,
                 -k2*x1**2 + k2*x2**2 + k2*(a0-c0)*x1 - (1+k2*(a0+c0))*x2 + k2*a0*c0 - d0])

print("A")
res = newton(ff1, vars, np.array([0,0]), tol=10**(-9))
print(str(res) + "\n")

print("a=" + str(a0-res[0]-res[1]))
print("b=" + str(b0-res[0]))
print("c=" + str(c0+res[0]-res[1]))
print("d=" + str(d0+res[1]))
#%%
# </editor-fold>



