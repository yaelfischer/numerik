import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.cos(x)


x0 = 1


# exakter Wert der Ableitung
y = -np.sin(1)

DeltaRechts = []
DeltaLinks = []
DeltaZentral = []
Hs = 10.**np.arange(-20, -1)  # logarithmische Schrittweite

for h in Hs:
    # Fehler des rechtsseitigen Differenzenquotient
    DeltaRechts.append(np.abs(y-(f(x0+h)-f(x0))/h))

    # Fehler des linksseitigen Differenzenquotient
    DeltaLinks.append(np.abs(y-(f(x0)-f(x0-h))/h))

    # Fehler des zentralen Differenzenquotient
    DeltaZentral.append(np.abs(y-(f(x0+h)-f(x0-h))/(2*h)))


plt.loglog(Hs, DeltaRechts, 'o-', label='Vorwaertsdifferenzenquotient')
# die beiden folgenden Zeilen können Sie nach dem Implementieren
# der weiteren Differenzenquotienten aktivieren
plt.loglog(Hs, DeltaLinks, '.-', label='Rueckwaertsdifferenzenquotient')
plt.loglog(Hs, DeltaZentral, '.-', label='Zentraler Diff-quotient')
plt.xlabel('Schrittweite')
plt.ylabel('absoluter Fehler')
plt.title('Fehlerentwicklung der Differenzenquotienten für h->0')
plt.legend()
plt.grid()
plt.show()
plt.savefig("plots/plot.png")
