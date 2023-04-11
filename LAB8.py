import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import sympy as sp
import scipy as scipy
from scipy.linalg import solve_triangular


def ExpEuler(start, X, N, f):
    [x0, y0]=start

    h = (X-x0) / N
    x = []
    y = []
    x.append(x0)
    y.append(y0)

    for i in range(N):
        x.append(x[i] + h)
        k = f(x[i],y[i])
        y.append(y[i] + h*k)

    return np.array(x),np.array(y)


# y' = -4y, y(0) = 1
def f(x,y):
    return -4*y

# Explizit
n = range(100,10000,100)
err = []
for ns in n:
    x,y = ExpEuler([0,1], 2, ns, f)
    analytic = np.exp(-4 * x)
    err.append(np.linalg.norm(y-analytic,np.inf)) # ya(x) ist die exakte Lösung

plt.loglog(n,err,'-')
plt.xlabel('n')
plt.ylabel(r'$\max_k \|e(x_k,h)\|$')
plt.grid()
#plt.show()


# Implizit


def ImpEuler(start, X, N, f, df ,tol=1e-8):
    [x0, y0]=start

    h = (X-x0) / N
    x = []
    y = []
    x.append(x0)
    y.append(y0)
    k = f(x[0], y[0])

    for i in range(N):
        x.append(x[i] + h)
        r = k - f(x[i]+h,y[i]+h*k)
        
        while np.abs(r) > tol:
            J = df(x[i]+h,y[i]+h*k)
            dk = -r/(1-h*J)
            k = k + dk
            r = k - f(x[i]+h,y[i]+h*k)

        y.append(y[i] + h*k)

    return np.array(x),np.array(y)


# y' = -4y, y(0) = 1
def f(x,y):
    return -4*y


def df(x,y):
    return -4

n = range(100,10000,100)
err = []
for ns in n:
    x,y = ImpEuler([0,1], 2, ns, f, df)
    analytic = np.exp(-4 * x)
    err.append(np.linalg.norm(y-analytic,np.inf)) # ya(x) ist die exakte Lösung

plt.loglog(n,err,'-')
plt.xlabel('n')
plt.ylabel(r'$\max_k \|e(x_k,h)\|$')
plt.legend(['Explizit', 'Implizit'])
plt.grid()
plt.show()


# Aufgabe 6
# f = -x^2/y(x)    y(0) = -4

def f(x,y):
    return -(x**2)/y

def df(x,y):
    return (x**2)/(y**2)


x = np.array(range(0,2000))/1000
analytic = -np.sqrt(16-(2*x**3)/3)

j = np.array([1,2,3,4,5,6,7,8])

errImp = []
errExp = []

for js in j:

    N = 3 ** js
    x1,y1 = ExpEuler([0,-4], 2, N, f,)
    x2,y2 = ImpEuler([0,-4], 2, N, f, df)

    errImp.append(np.abs(analytic[-1] - y1[-1]))
    errExp.append(np.abs(analytic[-1] - y2[-1]))

plt.plot(3**j, errExp, 3**j, errImp)
plt.legend(['Explizit', 'Implizit'])
plt.show()

x,y = np.meshgrid(np.linspace(0,2,10),np.linspace(-5,-2,10))
u = 1
v = f(x,y)
plt.quiver(x,y,u,v, scale=15, width=0.002)
plt.plot(x1,y1,x2,y2)
plt.legend(['Feld', 'Explizit', 'Implizit'])
plt.show()