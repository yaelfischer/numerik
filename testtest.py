import numpy as np
import matplotlib.pyplot as plt
import scipy as sy

def SplineNat(points):
    # ind = np.lexsort((points[:,1],points[:,0]))
    # points = points[ind]
    n = len(points[0,:])
    h = points[0,1]-points[0,0]

    A = np.zeros((n-2,n-2))
    b = np.zeros((n-2,1))
    for i in range(n-2):
        A[i,i] = 4
        if i != 0:
            A[i,i-1] = 1
        if i != n-3:
            A[i,i+1] = 1

        b[i] = (points[1,i]-2*points[1,i+1]+points[1,i+2])/(h**2)

    M = np.linalg.solve(A,6*b)
    M = M[:,0].tolist()
    M.append(0)
    M.insert(0,0)
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
        x = getSupports(20,points[0,i],points[0,i+1])
        for xs in x:
            xss.append(xs)
            ys.append(((points[0,i+1]-xs)**3)/(6*h)*M[i]+((xs-points[0,i])**3)/(6*h)*M[i+1]+c[i]*(xs-points[0,i])+d[i])
    ys.append(((points[0,n-1]-points[0,-1])**3)/(6*h)*M[n-1]+((points[0,-1]-points[0,n-2])**3)/(6*h)*M[n-1]+c[n-2]*(points[0,-1]-points[0,n-2])+d[n-2])
    xss.append(points[0,-1])

    return xss,ys


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
        x = getSupports(20,points[0,i],points[0,i+1])
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

data = np.array([[2, 4, 6, 8, 10, 12, 14],
                 [2, 3, 4, 2, 8,  7,  3]])


xx, yy = SplineNat(data)
plt.plot(xx, yy)
plt.scatter(data[0],data[1])
xx, yy = SplinePer(data)
plt.plot(xx, yy)
plt.show()