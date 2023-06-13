import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Variablen deklarieren
m = 8
d = 0.1
k = 5
F = 2
T = 10 #simulation Time
print('Die echten Werte für d, k, m sind: ', [d, k, m])
def f(t, x):
    dxdt = x[1]
    dxdtdt = 1/m*(F-k*x[0]-d*x[1])
    return np.array([dxdt,dxdtdt])

#RK4 Verfahren

dt = 0.01 #schrittweite
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


theta = np.array([])
yk = np.array([])
ykdt = np.array([])
ykddt = np.array([])

yk = np.append(yk, 0)
ykdt = np.append(ykdt, 0)
ykddt = np.append(ykddt, 0)

psi = np.array([-xdata[1]/dt, -xdata[1]/(dt**2), F])
psi2 = np.array([-(xdata[2]-2*xdata[1])/(dt**2), -(xdata[2]-xdata[1])/dt, F])
psi = np.vstack((psi, psi2))
print(psi)
