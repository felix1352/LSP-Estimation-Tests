import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Variablen deklarieren
m = 8
d = 0.7
k = 5
F = 2
T = 20 #simulation Time
print('Die echten Werte für d, k, m sind: ', [d, k, m])
def f(t, x):
    dxdt = x[1]
    dxdtdt = 1/m*(F-k*x[0]-d*x[1])
    return np.array([dxdt,dxdtdt])

#RK4 Verfahren

dt = 0.001 #schrittweite
t = 0 #startzeitpunkt
x0 = 2 #startwert für x
y0 = 0  #startwert für y
tdata = np.array([])
xdata = np.array([])

while t<T:
    k1 = dt * f(t, [x0, y0])
    k2 = dt * f(t+dt/2, [x0 + k1[0]/2, y0+k1[1]/2])
    k3 = dt * f(t+dt/2, [x0+k2[0]/2, y0+k2[1]/2])
    k4 = dt * f(t+dt, [x0+k3[0], y0+k3[1]])
    x0 += (k1[0]+ 2*k2[0] +2*k3[0]+k4[0])/6
    y0 += (k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
    t += dt
    tdata = np.append(tdata, t)
    xdata = np.append(xdata, x0)

plt.subplot(1,2,1)
plt.plot(tdata, xdata, label='echte Funktion')

#Parameter Estimation nach IdS-Skript

psi = np.array([-(xdata[2]-2*xdata[2-1]+xdata[2-2])/(dt**2), -(xdata[2]-xdata[2-1])/dt, F])

for i in range(3,len(xdata)):
    psiadd = np.array([-(xdata[i]-2*xdata[i-1]+xdata[i-2])/(dt**2), -(xdata[i]-xdata[i-1])/dt, F])
    psi = np.vstack((psi, psiadd))

psitrans = np.transpose(psi)
psipsi = np.dot(psitrans, psi)
psipsiinv = np.linalg.inv(psipsi)

psitransdoteta = np.dot(psitrans, xdata[2:len(xdata)])

theta = np.dot(psipsiinv, psitransdoteta)

kest = 1/theta[2]
dest = theta[1]*kest
mest = theta[0]*kest
paramest = [dest, kest, mest]
print('Die geschätzten Werte für d, k, m sind: ', paramest)

f = [abs(d-dest), abs(k-kest), abs(m-mest)]
print('Der absolute Fehler der Werte ist: ', f)

#Plotten

#RK4 Verfahren

t = 0 #startzeitpunkt
x0 = 2 #startwert für x
y0 = 0  #startwert für y
xest = np.array([])

def fest(t, x):
    dxdt = x[1]
    dxdtdt = 1/mest*(F-kest*x[0]-dest*x[1])
    return np.array([dxdt,dxdtdt])

while t<T:
    k1 = dt * fest(t, [x0, y0])
    k2 = dt * fest(t+dt/2, [x0 + k1[0]/2, y0+k1[1]/2])
    k3 = dt * fest(t+dt/2, [x0+k2[0]/2, y0+k2[1]/2])
    k4 = dt * fest(t+dt, [x0+k3[0], y0+k3[1]])
    x0 += (k1[0]+ 2*k2[0] +2*k3[0]+k4[0])/6
    y0 += (k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
    t += dt
    xest = np.append(xest, x0)

plt.plot(tdata, xest,'--', label='geschätzte Funktion')
plt.xlabel('Zeit in s')
plt.ylabel('Auslenkung in m')
plt.legend(loc='lower right')
plt.title('Vergleich der Funktionen')

plt.subplot(1,2,2)
plt.plot(tdata, abs(xest-xdata))
plt.xlabel('Zeit in s')
plt.ylabel('Fehler Auslenkung in m')
plt.title('Fehler der Schätzung')
plt.show()
