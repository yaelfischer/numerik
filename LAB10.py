import numpy as np
import matplotlib.pyplot as plt
import scipy as sy


def ImpRK(anfangs, X, N, f, but):
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
            k[j] = sy.optimize.fsolve(lambda imp: f(x[i]+but[j,0]*h,y[i]+h*np.dot(but[j,1:j+1],k[0:j])+h*but[j,j+1]*imp)-imp, np.array([k[j-1]]), xtol=1.49012e-12)
            counter += 1

        y.append(y[i] + h*np.dot(k,but[-1,1:]))

    return np.array(x),np.array(y), counter



#### Butcher-Tableau #####
    # c1 | a11 0   0
    # c2 | a21 a22 0
    # c3 | a31 a32 a33
    # ------------------
    #    | b1  b2  b3


mittelPunktregel = [[0.5, 0.5],
                    [0,   1]]
mittelPunktregel = np.array(mittelPunktregel)

# # butcher2x2
# butcher = [[c1, a11, a12],
#            [c2, a21, a22],
#            [0,  b1,  b2]]

trapezregel = [[0, 0,   0],
               [1, 0.5, 0.5],
               [0, 0.5, 0.5]]
trapezregel = np.array(trapezregel)

SDIRK2 = [[0.5+np.sqrt(3)/6, 0.5+np.sqrt(3)/6, 0],
          [0.5-np.sqrt(3)/6, -np.sqrt(3)/3,    0.5+np.sqrt(3)/6],
          [0,                0.5,              0.5]]
SDIRK2 = np.array(SDIRK2)

# # butcher3x3
# butcher = [[c1, a11, a12, a13],
#            [c2, a21, a22, a23],
#            [c3, a31, a32, a33],
#            [0,  b1,  b2,  b3]]
xNor = 1.06858
norsett = [[xNor, xNor, 0, 0],
           [0.5, 0.5-xNor, xNor, 0],
           [1-xNor, 2*xNor, 1-4*xNor, xNor],
           [0,  1/(6*(1-2*xNor)**2),  ((3*(1-2*xNor)**2)-1)/(3*(1-2*xNor)**2), 1/(6*(1-2*xNor)**2)]]
norsett = np.array(norsett)

# # butcher4x4
# butcher = [[c1, a11, a12, a13, a14],
#            [c2, a21, a22, a23, a24],
#            [c3, a31, a32, a33, a34],
#            [c4, a41, a42, a43, a44],
#            [0,  b1,  b2,  b3,  b4]]

## Test func
# y' = -4y, y(0) = 1
def f1(x,y):
    return -4*y


x1, y1, _ = ImpRK([0,1],1,10,f1,mittelPunktregel)
x2, y2, _ = ImpRK([0,1],1,10,f1,trapezregel)
x3, y3, _ = ImpRK([0,1],1,10,f1,SDIRK2)
x4, y4, _ = ImpRK([0,1],1,10,f1,norsett)

analytic = np.exp(-4 * x1)

plt.plot(x1,y1)
plt.plot(x2,y2)
plt.plot(x3,y3)
plt.plot(x4,y4)
plt.plot(x1,analytic,linestyle='dashed')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Mittelpunkt', 'Trapez', 'SDIRK2', 'Norsett', 'Analytic'])
plt.show()


def f2(x,y):
    return -(x**2)/y

Ns = np.array([1,2,3,4,5,6,7,8])
Ns = 3**Ns

endVal = -4*np.sqrt(2/3)

x1Err = []
x2Err = []
x3Err = []
x4Err = []

count1 = []
count2 = []
count3 = []
count4 = []

for i in Ns:
    x1, y1, c1 = ImpRK([0, -4], 2, i, f2, mittelPunktregel)
    x2, y2, c2 = ImpRK([0, -4], 2, i, f2, trapezregel)
    x3, y3, c3 = ImpRK([0, -4], 2, i, f2, SDIRK2)
    x4, y4, c4 = ImpRK([0, -4], 2, i, f2, norsett)
    count1.append(c1)
    count2.append(c2)
    count3.append(c3)
    count4.append(c4)
    x1Err.append(np.abs(endVal - y1[-1]))
    x2Err.append(np.abs(endVal - y2[-1]))
    x3Err.append(np.abs(endVal - y3[-1]))
    x4Err.append(np.abs(endVal - y4[-1]))

plt.loglog(Ns, x1Err, 'o-')
plt.loglog(Ns, x2Err, 'o-')
plt.loglog(Ns, x3Err, 'o-')
plt.loglog(Ns, x4Err, 'o-')
plt.xlabel('Anzahl Schritte')
plt.ylabel('Fehler')
plt.legend(['Mittelpunkt', 'Trapez', 'SDIRK2', 'Norsett'])
plt.show()
