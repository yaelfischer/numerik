import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy as sy
import sympy as sp

# <editor-fold desc="Aufgabe 2 Finite-Differenzen-Methode (Lineares Randwertproblem)">
#%%
def DifferenzenMethode(N):
    x = np.linspace(0,3,N)

    # Dirichlet Randwerte
    u0 = 0
    un = np.exp(-6)*np.sin(6)

    Xend = 3
    h = Xend/(N+1)

    # System Matrix
    # y'' +4y' +8y = 0
    const = 0
    Css = 1
    Cs = 4
    C = 8

    A = np.zeros((3,N))
    b = np.ones((N,1))
    b = b*const

    b[0] = b[0] - (Css/h**2 - Cs/(2*h))*u0
    b[-1] = b[-1] - (Css/h**2 + Cs/(2*h))*un

    A[1] = C - (2*Css)/(h**2)
    A[2] = Css/h**2 - Cs/(2*h)
    A[0] = Css/h**2 + Cs/(2*h)

    A[0] = np.roll(A[0],1)
    A[2] = np.roll(A[2],-1)

    y = la.solve_banded((1,1),A,b)
    return x,y

def f(x):
    return np.exp(-2*x)*np.sin(2*x)


j = [1,2,3,4,5,6]

for js in j:
    x,y = DifferenzenMethode(10**js)
    analytic = f(x)
    plt.plot(x, y)
    err = np.abs(y[:,0] - analytic)

    print("Max error at j=" + str(js) + " = " + str(np.amax(err)))

x,y = DifferenzenMethode(10**3)
analytic = f(x)
plt.plot(x,analytic)
plt.legend(['j=1','j=2','j=3','j=4','j=5','j=6', 'analytic'])
plt.show()

# </editor-fold>

# <editor-fold desc="Aufgabe 3 Lineare Ausgleichsrechnung">
#%%
## A
# y = a*((x-1)/(x+3)) + b*exp(-4*(x+1)/5) + c*0.5

data = np.loadtxt('S14A3a.csv', delimiter=',')
plt.plot(data[0,:], data[1,:])

N = len(data[0,:])
unknown = 3

A = np.zeros((N,unknown))
b = data[1,:]

for index, line in enumerate(A):
    x = data[0, index]

    line[0] = (x-1)/(x+3)
    line[1] = np.exp(-4*(x+1)/5)
    line[2] = 0.5

Q, R = np.linalg.qr(A)
alpha = la.solve_triangular(R,np.transpose(Q)@b)
print("y = " + str(np.transpose(Q)@b))
print("Alphas:" + str(alpha))

y = []
err = []
for l in data.T:
    x = l[0]
    im = alpha[0]*(x-1)/(x+3) + alpha[1]*np.exp(-4*(x+1)/5) + alpha[2]*0.5
    err.append((l[1] - im)**2)
    y.append(im)

plt.plot(data[0],y)
plt.show()
print("Error: " + str(sum(err)))

## B

data = np.loadtxt('S14A3b.csv', delimiter=',')
plt.plot(data[0,:], data[1,:])

N = len(data[0,:])
unknown = 3

A = np.zeros((N,unknown))
b = data[1,:]

for index, line in enumerate(A):
    x = data[0, index]

    line[0] = 1
    line[1] = 1/(x**2 + 1)
    line[2] = np.sqrt(x + 2)

Q, R = np.linalg.qr(A)
alpha = la.solve_triangular(R,np.transpose(Q)@b)
print("y = " + str(np.transpose(Q)@b))
print("Alphas:" + str(alpha))

y = []
err = []
for l in data.T:
    x = l[0]
    im = alpha[0] + alpha[1]*1/(x**2 + 1) + alpha[2]*np.sqrt(x + 2)
    err.append((l[1] - im)**2)
    y.append(im)

plt.plot(data[0],y)
plt.show()
print("Error: " + str(sum(err)))
#%%
# </editor-fold>

# <editor-fold desc="Aufgabe 4 Newton, nicht lin. Gleichungssysteme">
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

# x^2 + 2*y^2 -3 = 0
# 2*sin(x)*sin(y-1) - 1 = 0

xx, yy, ff = sp.symbols('xx,yy,ff')
vars = [xx,yy]

ff1 = sp.Matrix([xx**2 + 2*yy**2 - 3, 2*sp.sin(xx)*sp.sin(yy-1)-1])
ff2 = sp.Matrix([sp.exp(xx**2 - 2*yy**2) + 2*xx*yy, yy**2-2*xx-1])
ff3 = sp.Matrix([3*xx**2 + 4*sp.exp(yy) - 5, 6*xx**2 + 2*sp.sin(xx) + (yy-1)**2 -3])

print("A")
res = newton(ff1, vars, np.array([1,1]), tol=10**(-9))
print(str(res) + "\n")
res = newton(ff1, vars, np.array([-1,-1]), tol=10**(-9))
print(str(res))

print("\n\nB")
res = newton(ff2, vars, np.array([1,1]), tol=10**(-9))
print(str(res) + "\n")
res = newton(ff2, vars, np.array([-1,-1]), tol=10**(-9))
print(str(res) + "\n")
res = newton(ff2, vars, np.array([-2,-2]), tol=10**(-9))
print(str(res))

print("\n\nC")
res = newton(ff3, vars, np.array([-1,1]), tol=10**(-9))
print(str(res) + "\n")
res = newton(ff3, vars, np.array([1,-1]), tol=10**(-9))
print(str(res))
#%%
# </editor-fold>

# <editor-fold desc="Aufgabe 5 Explizit Runge Kutta (gDgl 1. Ordnung) Anfangswertproblem">
#%%
def ExpRK(anfangs, X, N, f, but):
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
            k[:,j] = f(x[i]+but[j,0]*h,y[i]+h*but[j,1:]@np.transpose(k))
            counter += 1

        y.append(y[i] + h*np.dot(k,but[-1,1:]))

    return np.array(x),np.array(y), counter


RK4 = [[0.0, 0.0, 0.0, 0.0, 0.0],
       [0.5, 0.5, 0.0, 0.0, 0.0],
       [0.5, 0.0, 0.5, 0.0, 0.0],
       [1.0, 0.0, 0.0, 1.0, 0.0],
       [0,   1/6, 1/3, 1/3, 1/6]]
RK4 = np.array(RK4)

aufgabe = [[0,   0,   0],
           [5/8, 5/8, 0],
           [0,   1/5, 4/5]]
aufgabe = np.array(aufgabe)

# xy' - 3x^2y^2 = 5x, y(-2) = 1  -> umstellen nach y'
def f(x,y):
    return (5*x+3*x**2*y**2)/x

N = [5,10,20,40,50]
X = 0
start = [-2,[1]]
yEnd = 2.7810285

err1 = []
err2 = []

for ns in N:
    x1, y1, _ = ExpRK(start, X, ns, f, RK4)
    x2, y2, _ = ExpRK(start, X, ns, f, aufgabe)
    err1.append(np.abs(y1[-1] - yEnd))
    err2.append(np.abs(y2[-1] - yEnd))

plt.loglog(N,err1, 'o-')
plt.loglog(N,err2, 'x-')
plt.loglog(np.arange(0,100,1), 1/np.arange(0.0,100.0,1), linewidth=0.5, color='black')
plt.loglog(np.arange(0,100,1), 1/np.arange(0.0,100.0,1)**2, linewidth=0.5, color='black', linestyle='dashed')
plt.loglog(np.arange(0,100,1), 1/np.arange(0.0,100.0,1)**4, linewidth=0.5, color='black', linestyle='dashdot')
plt.loglog(np.arange(0,100,1), 1/np.arange(0.0,100.0,1)**6, linewidth=0.5, color='black', linestyle='dotted')

plt.legend(['RK4', 'aufgabe', 'O1', 'O2', 'O4', 'O6'])
plt.show()
#%%
# </editor-fold>

# <editor-fold desc="Aufgabe 6 Explizit Runge Kutta (gDgl 2. Ordnung) Anfangswertproblem">
#%%
def ExpRK(anfangs, X, N, f, but):
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
            k[:,j] = f(x[i]+but[j,0]*h,y[i]+h*but[j,1:]@np.transpose(k))
            counter += 1

        y.append(y[i] + h*np.dot(k,but[-1,1:]))

    return np.array(x),np.array(y), counter


RK4 = [[0.0, 0.0, 0.0, 0.0, 0.0],
       [0.5, 0.5, 0.0, 0.0, 0.0],
       [0.5, 0.0, 0.5, 0.0, 0.0],
       [1.0, 0.0, 0.0, 1.0, 0.0],
       [0,   1/6, 1/3, 1/3, 1/6]]
RK4 = np.array(RK4)

# y'' + 2xy' -y = sin(x),  y(-1)=3, y'(-1)=0
# -> y'' = sin(x) + y + 2xy'
def f(x,y):
    return np.array([y[1],np.sin(x)+y[0]-2*x*y[1]])


N = 2**np.array([4, 5, 6, 7])
X = 4
start = [-1,[3,0]]
yEnd = 23.15276

err1 = []

for ns in N:
    x1, y1, _ = ExpRK(start, X, ns, f, RK4)
    err1.append(np.abs(y1[-1,0] - yEnd))

print("Fehler: " + str(err1))

plt.loglog(N,err1, 'o-')
plt.loglog(np.arange(0,100,1), 1/np.arange(0.0,100.0,1), linewidth=0.5, color='black')
plt.loglog(np.arange(0,100,1), 1/np.arange(0.0,100.0,1)**2, linewidth=0.5, color='black', linestyle='dashed')
plt.loglog(np.arange(0,100,1), 1/np.arange(0.0,100.0,1)**4, linewidth=0.5, color='black', linestyle='dashdot')
plt.loglog(np.arange(0,100,1), 1/np.arange(0.0,100.0,1)**6, linewidth=0.5, color='black', linestyle='dotted')

plt.legend(['RK4', 'O1', 'O2', 'O4', 'O6'])
plt.show()
#%%
# </editor-fold>

# <editor-fold desc="Aufgabe 7 Implizit Runge Kutta (System von linearen gDgln)">
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


EulerVor = [[0, 0],
            [0, 1]]
EulerVor = np.array(EulerVor)

EulerRuck = [[1, 1],
             [0, 1]]
EulerRuck = np.array(EulerRuck)

# y' = (0            1) Y       y(0) = (1,-1)
#      (-2(2x^2+1) -4x)
def f(x,y):
    return np.array([y[1], -2*(2*x**2+1)*y[0]-4*x*y[1]])


def exakt(x):
    return np.array([(1-x)*np.exp(-x**2), (2*x**2-2*x-1)*np.exp(-x**2)])

X = 1.2
N = (X/np.array([0.1,0.05,0.025])).astype(int)
start = [0,[1,-1]]
err1 = []
err2 = []
for ns in N:
    x1, y1, _ = ImpRK(start, X, ns, f, EulerVor)
    x2, y2, _ = ImpRK(start, X, ns, f, EulerRuck)
    err1.append(np.linalg.norm(y1[-1] - exakt(x1[-1])))
    err2.append(np.linalg.norm(y2[-1] - exakt(x2[-1])))
print("Fehler Exp: " + str(err1))
print("Fehler Imp: " + str(err2))

X = 3
N = (X/np.array([0.1,0.05,0.025])).astype(int)
start = [0,[1,-1]]
err1 = []
err2 = []
for ns in N:
    x1, y1, _ = ImpRK(start, X, ns, f, EulerVor)
    x2, y2, _ = ImpRK(start, X, ns, f, EulerRuck)
    err1.append(np.linalg.norm(y1[-1] - exakt(x1[-1])))
    err2.append(np.linalg.norm(y2[-1] - exakt(x2[-1])))
print("Fehler Exp: " + str(err1))
print("Fehler Imp: " + str(err2))

X = 30
N = int(X/0.1)
start = [0,[1,-1]]
err1 = []
err2 = []
x1, y1, _ = ImpRK(start, X, N, f, EulerVor)
x2, y2, _ = ImpRK(start, X, N, f, EulerRuck)
err1.append(np.linalg.norm(y1[-1] - exakt(x1[-1])))
err2.append(np.linalg.norm(y2[-1] - exakt(x2[-1])))
print("Fehler Exp: " + str(err1))
print("Fehler Imp: " + str(err2))
#%%
# </editor-fold>

# <editor-fold desc="Aufgabe 8 Lineare Ausgleichsrechnung (Kondi log)">
#%%
# I(t) = U0/R * exp(-t/RC)
# ln(I(t)) = ln(U0)-ln(R) + (-t/RC)

data = np.array([[1,2,4],[6.2e-6, 3.5e-6, 1.3e-6]])

N = len(data[0,:])
unknown = 2

R = 1e6

A = np.zeros((N,unknown))
b = np.log(data[1,:]) + np.log(R)

for index, line in enumerate(A):
    t = data[0, index]
    line[0] = 1
    line[1] = -t/R

QQ, RR = np.linalg.qr(A)
alpha = la.solve_triangular(RR,np.transpose(QQ)@b)
print("y = " + str(np.transpose(QQ)@b))
print("Alphas:" + str(alpha))

U0 = np.exp(alpha[0])
C = 1/alpha[1]

print("U0: " + str(U0))
print("C: " + str(C))

t = np.arange(0,5,0.1)
I = U0/R * np.exp(-t/(R*C))

plt.plot(t,I)
plt.show()
#%%
# </editor-fold>

# <editor-fold desc="Aufgabe 13 Nichtlineare Ausgleichsrechnung">
#%%
def GausNewton(func, vars, start, tol=1e-10, maxIt=100, Df = None, damped = False):

    if not callable(Df):
        jac = func.jacobian(vars)
        print(jac)
        Df = sp.lambdify(vars, jac)

    print("lambdify start")
    f = sp.lambdify(vars, func)
    print("lambdify stop")

    k = 0
    x = start
    J = Df(*x)
    y = f(*x)
    r = np.linalg.norm(J.T @ y)
    s = np.linalg.norm(y)**2
    while r > tol and k < maxIt:
        k = k + 1
        left = J.T @ J
        right = -(J.T @ y)
        dx = np.linalg.solve(left, right)

        if damped is True:
            m = 2*(J.T @ y).T @ dx
            m = np.linalg.norm(m)
            u = 1
            inter = x + u * dx[:, 0]
            y = f(*inter)
            while (np.linalg.norm(y)**2) > (s + u*(1/2)*m):
                u = 0.5 * u
                inter = x + u*dx[:, 0]
                y = f(*inter)
        else:
            u = 1

        x = x + u*dx[:, 0]
        J = Df(*x)
        y = f(*x)
        r = np.linalg.norm(J.T @ y)
        s = np.linalg.norm(y) ** 2
        if (k <= 3):
            print("Schritt " + str(k+1) + ": " + str(x))

    return x


data13 = np.array([[0.6,  0.8,  0.85, 0.95, 1.0,  1.1,  1.2,  1.3,  1.45, 1.6,  1.8],
                   [0.08, 0.06, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05, 0.04]])

start = [0,-1,-2]

# y = A/x + B/x * exp(C*x^2)  --> y - ... = 0

A, B, C, f = sp.symbols('A,B,C,f')
vars = [A,B,C]

func = []
for j in range(len(data13[0, :])):
    func.append([data13[1, j] - (A/data13[0, j] + B*sp.exp(C*data13[0, j]**2)/data13[0, j])])

f = sp.Matrix(func)

x = GausNewton(f,vars,start,1e-9)
print("A,B,C am Ende: " + str(x))
#%%
# </editor-fold>