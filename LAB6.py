import numpy as np
import numpy.linalg as lin
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.patches as mpatches
import scipy as scipy
import sympy
from scipy.linalg import solve_triangular
from sympy import sin, cos, Matrix
from sympy.abc import rho, phi

metadata = dict(title='Trajektorie', artist='Your Name',
                comment='Movie')
writer = FFMpegWriter(fps=60, metadata=metadata)

# Aufgabe 1 Funktion
# fx = 2*cos(T1) + cos(T1+T2)
# fy = 2*sin(T1) + sin(T1+T2)
# 2*cos(asin((sin(z+m)-2)/2))+cos(asin((sin(z+m)-2)/2)+m)


def newton(f, vars, start, tol, nMax):   # Startwert, Fehlertoleranz, max Iteratione
    Df = f.jacobian(vars)

    k = 0
    x = start
    y = f.subs(zip(vars,x[0,:]))
    y = np.array(y).astype(np.float64)
    r = lin.norm(y)   # Startwerte

    while r>tol and k<nMax:
        J = Df.subs(zip(vars,x[0,:]))
        J = np.array(J).astype(np.float64)
        dx = lin.solve(J, -y)
        k = k+1
        x = x+np.transpose(dx)
        y = f.subs(zip(vars, x[0,:]))
        y = np.array(y).astype(np.float64)
        r = lin.norm(y)
    #print("Newtoniterationen: " + str(k))
    if k>=nMax:
        #print("No zero found")
        return np.ones((1,2))
    x = np.array(x).astype(np.float64)
    return x

xpos = -1.18
ypos = -0.29

# startwertschätzer
def findAngle(pos,scaler):  # Findet Quadrant vom ersten Gelenk und Quadrant vom zweiten Gelenk
    start = 0
    newpos = pos.copy()
    if pos[0] >= 0:
        if pos[1] >= 0:
            start = np.deg2rad(45)
            newpos -= scaler
        else:
            start = np.deg2rad(325)
            newpos[0] -= scaler
            newpos[1] += scaler
    else:
        if pos[1] >= 0:
            start = np.deg2rad(135)
            newpos += scaler
        else:
            start = np.deg2rad(225)
            newpos[0] += scaler
            newpos[1] -= scaler

    return start, newpos


startVal = [0,0]
startVal[0], newpos = findAngle([xpos, ypos], np.sqrt(2))
startVal[1], newpos = findAngle(newpos, np.sqrt(2))
print("Startwerte sind kritisch. Darum Implementation von Startwertschätzer.")
print("Start values: " + str(np.rad2deg(startVal[0])) + " " + str(np.rad2deg(startVal[1])))

f = Matrix([2*cos(phi) + cos(phi+rho)-xpos,2*sin(phi) + sin(phi+rho)-ypos])

vars = [phi, rho]

x = newton(f, vars, np.array([startVal]), 10**-8, 500)

x[0, 0] %= 2*np.pi
x[0, 1] %= 2*np.pi

print("Erster Winkel: " + str(np.rad2deg(x[0, 0])))
print("Zweiter Winkel: " + str(np.rad2deg(x[0, 1])))
print("Eine Position in der Ebene kann mit verschiedenen Kombinationen von Winkel erreicht werden (2)")


# Abfahren einer Trajektorie

fig = plt.figure(figsize=(6,6))
l1, = plt.plot([], [])
l2, = plt.plot([], [],'o')

# p: Funktion zur Berechnung der Punkte auf der Trajektorie
def p(t):
    p0 = np.array([-2, 1])
    d = 1 / np.sqrt(17) * np.array([1, -4])
    test = t * d
    return p0 + test

# ti: wird auch für die Berechnung der Winkel benutzt
ti = np.linspace(0,4,int(4/.05+1))

plt.subplot(2,1,1)
plt.plot(*np.array([p(tii) for tii in ti]).T)
plt.title("Trajektorie")

plt.xlim(-3,3)
plt.ylim(-3,3)
plt.gca().set_aspect(1)
plt.gca().add_patch(mpatches.Circle((0,0), 2-1,alpha=0.1))
plt.gca().add_patch(mpatches.Circle((0,0), 2+1,alpha=0.1))
plt.grid()
plt.xlabel('x')
plt.ylabel('y')

# si: Liste der Winkel fuer die Trajektorie
si = []
startVal=[np.pi,np.pi/2]
Df = f.jacobian(vars)
for tii in ti:
    pos = p(tii)
    f = Matrix([2 * cos(phi) + cos(phi + rho) - pos[0], 2 * sin(phi) + sin(phi + rho) - pos[1]])
    sol = newton(f, vars, np.array([startVal]), 1e-6, 10)
    startVal = [sol[0,0], sol[0,1]]
    si.append([sol[0,0], sol[0,1]])

si = np.array(si)
# PG: liefert Drehpunkte und Endpunkt des Roboters. Bsp:
#       array([[0.        , 0.        ],
#              [0.58856217, 1.91143783],
#              [1.        , 1.        ]])
def PG(angles):
    PG = np.zeros((3, 2))
    PG[1] = 2*cos(angles[0]) + 2*sin(angles[0])
    PG[2] = PG[1] + cos(angles[1]) + sin(angles[1])
    return PG

print("Gelernt: Sympy ist langsam!!")

plt.subplot(2,1,2)
plt.plot(ti, np.rad2deg(si[:,0]), ti, np.rad2deg(si[:,1]))
plt.title("Angles")
plt.legend(['phi', 'rho'])
plt.show()

# with writer.saving(fig, 'Trajektorie.mp4',400):   # Fehler trotz Installation
#     for s in si:
#         l1.set_data(*PG(*s).T)
#         l2.set_data(*PG(*s).T)
#
#         writer.grab_frame()