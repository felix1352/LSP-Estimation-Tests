#In diesem Beispiel versuche ich die Least Square Parameter Estimation auf ein Polynom anzuwenden. Dabei sei der Grad des Polynoms bekannt. Zunächst versuche ich nur einen Parameter zu schätzen bevor ich dann alle versuche zu schätzen

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

a0 = 3
a1 = -3
a2 = 0
a3 = 0
params = np.array([a0, a1, a2, a3])
print("Die Parameter sind: ", params)

#Rauschen hinzufügen
noise = np.random.normal(0,500,100)
ttrain = np.linspace(0, 10, 100)
yreal = ttrain**4 + a3*ttrain**3+a2*ttrain**2+a1*ttrain+a0
ytrain = yreal + noise

#LSP Estimation

def f(params, t):
    a0est, a1est, a2est, a3est = params
    erg = t**4 + a3est* t**3 + a2est* t**2 + a1est* t + a0est
    return erg

def residuals(params, t, y):
    return f(params,t) - y

params_initial = np.array([0,0,0,0])

result = least_squares(residuals, params_initial, args=[ttrain, ytrain])
params_est = result.x
print("Die geschätzten parameter sind: ", params_est)
fehler = abs(params_est - params)
print("Der Fehler beträgt: ", fehler)

yest = ttrain**4 + params_est[3]* ttrain**3+ params_est[2]*ttrain**2 + ttrain*params_est[1] + params_est[0]
plt.plot(ttrain, yreal, label='echte Funktion')
plt.plot(ttrain, yest, '--', label='geschätzte Funktion')
plt.legend(loc='upper left')
plt.xlabel('t in s')
plt.xlabel('y')
plt.show()
