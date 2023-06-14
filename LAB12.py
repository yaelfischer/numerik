import numpy as np
import matplotlib.pyplot as plt
import scipy as sy

def ExpRK(anfangs, end, dx, f, but):
    counter = 0
    s = len(but[:,0])-1
    [x0, y0] = anfangs

    h = dx
    x = []
    y = []
    x.append(x0)
    y.append(np.array(y0))

    k = np.zeros((len(y0),s))
    deltaT = 1
    i = 0
    while deltaT >= end:
        x.append(x[i] + h)
        for j in range(s):
            k[:,j] = f(x[i]+but[j,0]*h,y[i]+h*but[j,1:]@np.transpose(k))
            counter += 1

        y.append(y[i] + h*np.dot(k,but[-1,1:]))

        deltaT = np.linalg.norm(y[i - 1] - y[i])
        i += 1

    return np.array(x),np.array(y), counter

def ImpRK(anfangs, end, dx, f, but):
    counter = 0
    s = len(but[:,0])-1
    [x0, y0] = anfangs

    h = dx
    x = []
    y = []
    x.append(x0)
    y.append(np.array(y0))

    k = np.zeros((len(y0),s))
    deltaT = 1
    i = 0
    while deltaT >= end:
        x.append(x[i] + h)
        for j in range(s):
            k[:,j] = sy.optimize.fsolve(lambda imp: f(x[i]+but[j,0]*h,y[i]+h*but[j,1:j+1]@np.transpose(k[:,0:j])+h*but[j,j+1]*imp)-imp, np.array([k[:,j-1]]), xtol=1.49012e-12)
            counter += 1

        y.append(y[i] + h*np.dot(k,but[-1,1:]))

        deltaT = np.linalg.norm(y[i - 1] - y[i])
        i += 1

    return np.array(x),np.array(y), counter

EulerVor = [[0, 0],
            [0, 1]]
EulerVor = np.array(EulerVor)

mittelPunktregel = [[0.5, 0.5],
                    [0,   1]]
mittelPunktregel = np.array(mittelPunktregel)

implTrap = [[0, 0,   0  ],
            [1, 0.5, 0.5],
            [0, 0.5, 0.5]]
implTrap = np.array(implTrap)

Crouzeix = [[0.5+np.sqrt(3)/6, 0.5+np.sqrt(3)/6, 0],
            [0.5-np.sqrt(3)/6, -np.sqrt(3)/3, 0.5+np.sqrt(3)/6],
            [0, 0.5, 0.5]]
Crouzeix = np.array(Crouzeix)


# dT/dt - lam/rho*c * ddT/dx^2 = 0
Tl = 100
Tr = 0

a = 3.8e-6
h = 0.1/30

dt = (h**2)/(2*a)   # 1.46 sekunden
dt = 0.8*dt

platten = np.hstack((np.ones(15)*100, np.zeros(15)))

def f1(x,u):
    up = np.zeros(len(u))
    for i in range(1,len(u)-1):
        up[i] = a*(u[i-1]-2*u[i]+u[i+1])/(h**2)
    up[0] = a*(100-2*u[0]+u[1])/(h**2)
    up[-1] = a*(u[-2]-2*u[-1]+0)/(h**2)
    return up

def f2(x,u):
    up = np.zeros(len(u))
    for i in range(1,len(u)-1):
        up[i] = a*(u[i-1]-2*u[i]+u[i+1])/(h**2)
    up[0] = a*(-2*u[0]+2*u[1])/(h**2)
    up[-1] = a*(2*u[-2]-2*u[-1])/(h**2)
    return up

x1, y1, _ = ExpRK([0,platten], 0.1, dt, f1, EulerVor)
x2, y2, _ = ExpRK([0,platten], 0.1, dt, f2, EulerVor)
x3, y3, _ = ImpRK([0,platten], 10e-5, 20, f1, implTrap)
pos = np.arange(0,0.1,h)
plt.subplot(3,1,1)
plt.plot(pos, y1[0,:])
plt.plot(pos, y1[49,:])
plt.plot(pos, y1[99,:])
plt.plot(pos, y1[150,:])
plt.legend(['y0', 'y50', 'y100'])
plt.title("Fix auf 100° und 0°")

plt.subplot(3,1,2)
plt.plot(pos, y2[0,:])
plt.plot(pos, y2[49,:])
plt.plot(pos, y2[99,:])
plt.plot(pos, y2[300,:])
plt.legend(['y0', 'y50', 'y100', 'y300'])
plt.title("Keine Wärmezufuhr oder Abfluss")

plt.subplot(3,1,3)
plt.plot(pos, y3[0,:])
plt.plot(pos, y3[4,:])
plt.plot(pos, y3[-1,:])
plt.legend(['y0', 'y45', 'y90'])
plt.title("Fix auf 100° und 0°, implizit")
plt.show()























