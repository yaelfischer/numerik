import numpy as np
import matplotlib.pyplot as plt
import scipy as sy


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

## Butcher ##
RK4 = [[0.0, 0.0, 0.0, 0.0, 0.0],
       [0.5, 0.5, 0.0, 0.0, 0.0],
       [0.5, 0.0, 0.5, 0.0, 0.0],
       [1.0, 0.0, 0.0, 1.0, 0.0],
       [0,   1/6, 1/3, 1/3, 1/6]]
RK4 = np.array(RK4)

mittelPunktregel = [[0.5, 0.5],
                    [0,   1]]
mittelPunktregel = np.array(mittelPunktregel)


rho = 1.184
Ca = 0.45
Radk = 0.1
m = 0.05
g = 9.81
ks = 0.05

v0 = 1

kl = (1/2)*Ca*rho*(np.pi*Radk**2)

def f(x,y):
    Vrel = y[1]+v0
    return np.array([y[1],-(ks/m)*y[0]+g-(kl/m)*np.sign(Vrel)*Vrel**2])

x0 = 0
y0 = [0,0]

x, y, _ = ExpRK([x0,y0], 100, 200, f, RK4)

plt.plot(x,y[:,0])
plt.plot(x,y[:,1])
#plt.legend(['y0', 'y1'])
#plt.show()

x1, y1, _ = ImpRK([x0,y0], 100, 200, f, mittelPunktregel)

plt.plot(x1,y1[:,0])
plt.plot(x1,y1[:,1])
plt.legend(['y0 Explizit','y1 Explizit','y0 Implizit', 'y1 Implizit'])
plt.show()


# Auslenkung bei t=inf. : x = (m*g-kl*v0^2) / ks

xinf = (m*g-kl*v0**2) / ks

err = abs(xinf - y[-1,0])
print("Error Explizit " + str(err))

err2 = abs(xinf - y1[-1,0])
print("Error Implizit " + str(err2))


## V0 damit sich Kugel nicht bewegt
## Fa = Fg
## k*vrel^2 = m*g -> vrel = sqrt(m*g/k) = v0

v0 = np.sqrt(m*g/kl)
x2, y2, _ = ExpRK([x0,y0], 100, 200, f, RK4)
x3, y3, _ = ImpRK([x0,y0], 100, 200, f, mittelPunktregel)

plt.plot(x2,y2[:,0])
plt.plot(x2,y2[:,1])
plt.plot(x3,y3[:,0])
plt.plot(x3,y3[:,1])
plt.legend(['y0','y1', 'y0Imp', 'y1Imp'])
plt.show()

## Jacobi

v0 = [0, 4, np.sqrt(ks*m)/kl, np.sqrt(m*g/kl)]
def jacobi(v): return np.array([[0, 1],[-ks/m, -(2*kl*v)/m]])
eig1 = np.linalg.eig(jacobi(v0[0]))
eig2 = np.linalg.eig(jacobi(v0[1]))
eig3 = np.linalg.eig(jacobi(v0[2]))
eig4 = np.linalg.eig(jacobi(v0[3]))

print(eig1[0])
print(eig2[0])
print(eig3[0])
print(eig4[0])

x = [x.real for x in eig1[0]]
y = [y.imag for y in eig1[0]]
plt.scatter(x, y)
x = [x.real for x in eig2[0]]
y = [y.imag for y in eig2[0]]
plt.scatter(x, y)
x = [x.real for x in eig3[0]]
y = [y.imag for y in eig3[0]]
plt.scatter(x, y)
x = [x.real for x in eig4[0]]
y = [y.imag for y in eig4[0]]
plt.scatter(x, y)
plt.legend(['eig1', 'eig2', 'eig3', 'eig4'])
plt.grid()
plt.show()


## Energie
v0 = 0
kl = 0
ks = 0.05
x4, y4, _ = ExpRK([x0,y0], 100, 125, f, RK4)
x5, y5, _ = ImpRK([x0,y0], 100, 125, f, mittelPunktregel)


Eexp = 0.5*ks*(y4[:,0]-(m*g)/ks)**2 + 0.5*m*y4[:,1]**2
Eimp = 0.5*ks*(y5[:,0]-(m*g)/ks)**2 + 0.5*m*y5[:,1]**2

plt.plot(x4, Eexp, x5, Eimp)
plt.legend(['Eexp', 'Eimp'])
plt.show()


