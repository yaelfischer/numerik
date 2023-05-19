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


EulerVor = [[0, 0],
            [0, 1]]
EulerVor = np.array(EulerVor)

mittelPunktregel = [[0.5, 0.5],
                    [0,   1]]
mittelPunktregel = np.array(mittelPunktregel)


# dT/dt - lam/rho*c * ddT/dx^2 = 0
Tl = 100
Tr = 0

a = 3.8e-6
h = 0.1/30

dt = (h**2)/(2*a)
#dt = dt*0.8
dt = 1
platten = np.hstack((np.ones(15)*100, np.zeros(15)))

def f(x,u):
    up = np.zeros(len(u))
    for i in range(1,len(u)-1):
        up[i] = a*(u[i-1]-2*u[i]+u[i+1])/(h**2)
    up[0] = a*(100-2*u[0]+u[1])/(h**2)
    up[-1] = a*(0-2*u[0]+u[1])/(h**2)
    return up


x, y, _ = ExpRK([0,platten], 0.1, dt, f, EulerVor)
pos = np.arange(0,0.1+h,h)
plt.plot(pos, y[0,-1])
plt.show()























