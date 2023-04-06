import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import sympy as sp
import scipy as scipy
from scipy.linalg import solve_triangular


def GausNewton(func, vars, start, tol=1e-10, maxIt=100, Df = None, damped = False):

    if not callable(Df):
        jac = func.jacobian(vars)
        Df = sp.lambdify(vars, jac)

    print("lambdify start")
    f = sp.lambdify(vars, func)
    print("lambdify stop")

    k = 0
    x = start
    J = Df(*x)
    y = f(*x)
    r = lin.norm(J.T @ y)
    s = lin.norm(y)**2

    while r > tol and k < maxIt:
        k = k + 1
        left = J.T @ J
        right = -(J.T @ y)
        dx = lin.solve(left, right)

        if damped is True:
            m = 2*(J.T @ y).T @ dx
            m = lin.norm(m)
            u = 1
            inter = x + u * dx[:, 0]
            y = f(*inter)
            while (lin.norm(y)**2) > (s + u*(1/2)*m):
                u = 0.5 * u
                inter = x + u*dx[:, 0]
                y = f(*inter)
        else:
            u = 1

        x = x + u*dx[:, 0]
        J = Df(*x)
        y = f(*x)
        r = lin.norm(J.T @ y)
        s = lin.norm(y) ** 2
        print("Iteration: " + str(k))

    return x


data = np.array([
    [0.1, 0.3, 0.7, 1.2, 1.6, 2.2, 2.7, 3.1, 3.5, 3.9],
    [0.558, 0.569, 0.176, -0.207, -0.133, 0.132, 0.055, -0.090, -0.069, 0.027]
]).T

a, tau, omega, phi = sp.symbols('a,tau,omega,phi')
vars = sp.Matrix([a, tau, omega, phi])

func = []
for j in range(len(data[:, 0])):
    func.append([data[j, 1] - a * sp.exp(-tau * data[j, 0]) * sp.sin(omega * data[j, 0] + phi)])

f = sp.Matrix(func)

x = GausNewton(f, vars, np.array([1, 1, 3, 1], dtype=float))
print(x)

reconst = []
time = np.linspace(0, 4, 200)
for j in time:
    reconst.append(x[0] * sp.exp(-x[1] * j) * sp.sin(x[2] * j + x[3]))

plt.scatter(data[:, 0], data[:, 1])
plt.plot(time, reconst, 'r')
plt.show()

# Aufgabe 3
data = np.loadtxt('LAB7_data.txt')
dataMin = min(data[:, 0])
dataMax = max(data[:, 0])
plt.subplot(2,1,1)
plt.plot(data[:, 0], data[:, 1])
data[:, 0] = (data[:, 0] - dataMin) / (dataMax - dataMin)
plt.subplot(2,1,2)
plt.plot(data[:, 0],data[:, 1])


# Lineare ausgleichsrechnung für startwerte
s0 = 0.2
x0 = 0.35

A = np.zeros((len(data[:, 0]), 5))
b = data[:, 1]

for index, line in enumerate(A):
    line[0] = 1
    line[1] = data[index, 0]
    line[2] = data[index, 0] ** 2
    line[3] = data[index, 0] ** 3
    line[4] = (data[index, 0] - x0) / (s0 / 2)
    line[4] = 1 / (1 + line[4] ** 2)

Q, R = lin.qr(A)
alpha = solve_triangular(R,np.transpose(Q)@b)

x = data[:, 0]
xs = (x-x0)/(s0/2)
f1 = alpha[0] + x*alpha[1] + x**2*alpha[2] + x**3*alpha[3] + alpha[4]*1/(1+xs**2)

plt.plot(data[:, 0],f1)
#plt.show()

a1, a2, a3, a4, a5, a6, a7, f = sp.symbols('a1,a2,a3,a4,a5,a6,a7,f')
vars = [a1, a2, a3, a4, a5, a6, a7]

func = sp.Matrix([-(a1 + a2*f + a3*(f**2) + a4*(f**3) + a5 * (1 / (1 + ((f-a6) / (a7/2))**2)))])
grad = [func.diff(xi) for xi in vars]

func = []
df = []
for j in range(len(data[:, 0])):
    z = data[j, 0]
    gradient = [xi.subs(f,z) for xi in grad]
    # gradient = sp.Matrix(gradient)
    df.append(gradient)
    func.append([data[j, 1] - (a1 + a2*z + a3*(z**2) + a4*(z**3) + a5 * (1 / (1 + ((z-a6) / (a7/2))**2)))])

df = sp.Matrix(df)
Df = sp.lambdify(vars, df)

f = sp.Matrix(func)
x1 = GausNewton(f, vars, np.hstack((alpha,[x0, s0])), Df = Df)
x2 = GausNewton(f, vars, 1.1*np.hstack((alpha,[x0, s0])), Df = Df)
x3 = GausNewton(f, vars, 0.8*np.hstack((alpha,[x0, s0])), Df = Df, maxIt=200 ,damped=True)

# reconstruct
f = data[:, 0]
xs = (f-x1[5])/(x1[6]/2)
f1 = x1[0] + f*x1[1] + f**2*x1[2] + f**3*x1[3] + x1[4]*1/(1+xs**2)
xs = (f-x2[5])/(x2[6]/2)
f2 = x2[0] + f*x2[1] + f**2*x2[2] + f**3*x2[3] + x2[4]*1/(1+xs**2)
xs = (f-x3[5])/(x3[6]/2)
f3 = x3[0] + f*x3[1] + f**2*x3[2] + f**3*x3[3] + x3[4]*1/(1+xs**2)
plt.plot(f, f1, f, f2, f, f3)
plt.legend(['Data', 'Schätzung', 'Gute Startwerte', 'Startwerte * 1.1', 'Startwerte * 0.8'])
plt.show()
