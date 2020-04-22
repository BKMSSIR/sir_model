import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize

I_kor = np.load("./Desktop/sir_model/I_active.npy")


"Korea case"
population = 51640000 #한국인구 5164만 2018년 기준, 구글출처


"""시점 변경"""
ydata = I_kor
xdata = np.arange(len(ydata))

N = population
inf0 = ydata[0]
sus0 = N-inf0
rec0 = 0.0

def sir_model(y, x, beta, gamma):
    sus = -beta*y[0]*y[1]/N
    rec = gamma*y[1]
    inf = -(sus + rec)
    return sus, inf, rec

def fit_odeint(x, beta, gamma):
    return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:, 1]

popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
fitted = fit_odeint(xdata, *popt)


plt.figure()
plt.plot(xdata, ydata, 'o', label='actural')
plt.plot(xdata, fitted, label='prediction')
plt.legend()
plt.show()