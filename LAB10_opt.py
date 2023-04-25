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


def ExpSchRK(anfangs, X, f, but, tol=10e-6, beta=0.8):
    counter = 0
    s = len(but[:,0])-2
    [x0, y0] = anfangs

    h = X-x0
    x = []
    y = []
    x.append(x0)
    y.append(y0)

    k = np.zeros(s)

    i = 0
    while x[i] < X:
        x.append(x[i] + h)
        for j in range(s):
            k[j] = f(x[i]+but[j,0]*h,y[i]+h*np.dot(but[j,1:],k))
            counter += 1
        yi = y[i] + h*np.dot(k,but[-2,1:])
        y.append(yi)
        yid = y[i] + h*np.dot(k,but[-1,1:])

        e = abs(yi-yid)

        if (beta*tol)/20 <= e <= (beta*tol):   # Akzeptieren, h<-h
            i += 1

        if e < (beta*tol)/20:   # Akzeptieren, h<-2h
            i += 1
            h = 2*h

        if e > (beta*tol):    # Ablehnen, h<-h/2
            h = h/2
            x.pop()
            y.pop()

    if x[-1] > X:
        x.pop()
        y.pop()
        h = X - x[-1]
        x.append(x[-1] + h)
        for j in range(s):
            k[j] = f(x[-1] + but[j, 0] * h, y[-1] + h * np.dot(but[j, 1:], k))
            counter += 1
        y.append(y[-1] + h * np.dot(k, but[-2, 1:]))

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
RKF45_4 = [[0.0, 0.0,    0.0,      0.0,    0.0,   0.0,   0.0],
           [2/9, 2/9,    0.0,      0.0,    0.0,   0.0,   0.0],
           [1/3, 1/12,   1/4,      0.0,    0.0,   0.0,   0.0],
           [3/4, 69/128, -243/128, 135/64, 0.0,   0.0,   0.0],
           [1.0, -17/12, 27/4,     -27/5,  16/15, 0.0,   0.0],
           [5/6, 65/432, -5/16,    13/16,  4/27,  5/144, 0.0],
           [0.0, 1/9,    0.0,      9/20,   16/45, 1/12,  0.0]]
RKF45_4 = np.array(RKF45_4)

RKF45_5 = [[0.0, 0.0,    0.0,      0.0,    0.0,    0.0,   0.0],
           [2/9, 2/9,    0.0,      0.0,    0.0,    0.0,   0.0],
           [1/3, 1/12,   1/4,      0.0,    0.0,    0.0,   0.0],
           [3/4, 69/128, -243/128, 135/64, 0.0,    0.0,   0.0],
           [1.0, -17/12, 27/4,     -27/5,  16/15,  0.0,   0.0],
           [5/6, 65/432, -5/16,    13/16,  4/27,   5/144, 0.0],
           [0.0, 47/450, 0.0,      12/25,  32/225, 1/30,  6/25]]
RKF45_5 = np.array(RKF45_5)

RKF45 = [[0.0, 0.0,    0.0,      0.0,    0.0,    0.0,   0.0],
         [2/9, 2/9,    0.0,      0.0,    0.0,    0.0,   0.0],
         [1/3, 1/12,   1/4,      0.0,    0.0,    0.0,   0.0],
         [3/4, 69/128, -243/128, 135/64, 0.0,    0.0,   0.0],
         [1.0, -17/12, 27/4,     -27/5,  16/15,  0.0,   0.0],
         [5/6, 65/432, -5/16,    13/16,  4/27,   5/144, 0.0],
         [0.0, 1/9,    0.0,      9/20,   16/45,  1/12,  0.0],
         [0.0, 47/450, 0.0,      12/25,  32/225, 1/30,  6/25]]
RKF45 = np.array(RKF45)


## Func
def f1(x,y):
    return -(x**2)/y

def ex(x):
    return -np.sqrt(16-(2/3)*x**3)

Ns = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
Ns = 2**Ns

endVal = -4*np.sqrt(2/3)

x1Err = []
x2Err = []
x3Err = []

count1 = []
count2 = []
count3 = []

for k,i in enumerate(Ns):
    print(k)
    x1, y1, c1 = ExpRK([0, -4], 2, i, f1, RKF45_4)
    x2, y2, c2 = ExpRK([0, -4], 2, i, f1, RKF45_5)
    x3, y3, c3 = ExpSchRK([0, -4], 2, f1, RKF45, tol=(10.0**(-k)))
    count1.append(c1)
    count2.append(c2)
    count3.append(c3)
    x1Err.append(np.abs(endVal - y1[-1]))
    x2Err.append(np.abs(endVal - y2[-1]))
    Err = []
    for j, xs in enumerate(x3): Err.append(abs(y3[j]-ex(x3[j])))
    x3Err.append(max(Err))


plt.loglog(count1, x1Err, 'o-')
plt.loglog(count2, x2Err, 'o-')
plt.loglog(count3, x3Err, 'o-')
plt.xlabel('Aufwand')
plt.ylabel('Fehler')
plt.legend(['RK45_4', 'RK45_5', 'RK45'])
plt.show()