import numpy as np
import matplotlib.pyplot as plt


def ExpRK(anfangs, X, N, f, but):
    counter = 0
    s = len(but[:,0])-1
    [x0, y0] = anfangs

    h = (X-x0) / N
    x = []
    y = []
    x.append(x0)
    y.append(y0)

    k = np.zeros(s)
    for i in range(N):
        x.append(x[i] + h)
        for j in range(s):

            k[j] = f(x[i]+but[j,0]*h,y[i]+h*np.dot(but[j,1:],k))
            counter += 1

        y.append(y[i] + h*np.dot(k,but[-1,1:]))

    return np.array(x),np.array(y), counter



#### Butcher-Tableau #####
    # c1 | 0   0   0
    # c2 | a21 0   0
    # c3 | a31 a32 0
    # ------------------
    #    | b1  b2  b3

# # butcher2x2
# butcher = [[c1, a11, a12],
#            [c2, a21, a22],
#            [0,  b1,  b2]]
runge = [[0,   0,   0],
         [0.5, 0.5, 0],
         [0,   0,   1]]
runge = np.array(runge)

heun = [[0, 0,   0],
        [1, 1,   0],
        [0, 0.5, 0.5]]
heun = np.array(heun)

# # butcher3x3
# butcher = [[c1, a11, a12, a13],
#            [c2, a21, a22, a23],
#            [c3, a31, a32, a33],
#            [0,  b1,  b2,  b3]]

# # butcher4x4
# butcher = [[c1, a11, a12, a13, a14],
#            [c2, a21, a22, a23, a24],
#            [c3, a31, a32, a33, a34],
#            [c4, a41, a42, a43, a44],
#            [0,  b1,  b2,  b3,  b4]]
RK4 = [[0.0, 0.0, 0.0, 0.0, 0.0],
       [0.5, 0.5, 0.0, 0.0, 0.0],
       [0.5, 0.0, 0.5, 0.0, 0.0],
       [1.0, 0.0, 0.0, 1.0, 0.0],
       [0,   1/6, 1/3, 1/3, 1/6]]
RK4 = np.array(RK4)

## Test func
# y' = -4y, y(0) = 1
def f1(x,y):
    return -4*y


x1, y1, _ = ExpRK([0,1],1,10,f1,runge)
x2, y2, _ = ExpRK([0,1],1,10,f1,heun)
x3, y3, _ = ExpRK([0,1],1,10,f1,RK4)

analytic = np.exp(-4 * x1)

plt.plot(x1,y1)
plt.plot(x2,y2, linestyle='dashed')
plt.plot(x3,y3)
plt.plot(x1,analytic,linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Runge', 'Heun', 'RK4', 'Analytic'])
plt.show()

## Func
def f2(x,y):
    return -(x**2)/y

Ns = np.array([1,2,3,4,5,6,7,8])
Ns = 3**Ns

endVal = -4*np.sqrt(2/3)

x1Err = []
x2Err = []
x3Err = []

count1 = []
count2 = []
count3 = []

for i in Ns:
    x1, y1, c1 = ExpRK([0, -4], 2, i, f2, runge)
    x2, y2, c2 = ExpRK([0, -4], 2, i, f2, heun)
    x3, y3, c3 = ExpRK([0, -4], 2, i, f2, RK4)
    count1.append(c1)
    count2.append(c2)
    count3.append(c3)
    x1Err.append(np.abs(endVal - y1[-1]))
    x2Err.append(np.abs(endVal - y2[-1]))
    x3Err.append(np.abs(endVal - y3[-1]))

plt.subplot(2,1,1)
plt.loglog(Ns, x1Err, 'o-')
plt.loglog(Ns, x2Err, 'o-')
plt.loglog(Ns, x3Err, 'o-')
plt.xlabel('Anzahl Schritte')
plt.ylabel('Fehler')
plt.legend(['Runge', 'Heun', 'RK4'])

plt.subplot(2,1,2)
plt.loglog(count1, x1Err, 'o-')
plt.loglog(count2, x2Err, 'o-')
plt.loglog(count3, x3Err, 'o-')
plt.xlabel('Aufwand')
plt.ylabel('Fehler')
plt.legend(['Runge', 'Heun', 'RK4'])
plt.show()