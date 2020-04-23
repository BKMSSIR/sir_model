# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:27:38 2020

@author: TaeHyun Hwang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize

I_t_china = np.load(r'C:\Users\yalhl\sir_model\taehyun\I_t_china.npy')
R_t_china = np.load(r'C:\Users\yalhl\sir_model\taehyun\R_t_china.npy')

I_t_korea = np.load(r'C:\Users\yalhl\sir_model\taehyun\I_t_kor.npy')
R_t_korea = np.load(r'C:\Users\yalhl\sir_model\taehyun\R_t_kor.npy')

def dydt(y_t, t, beta, gamma): 
    """
    y_t = (S(t), I(t), R(t))
    beta = transmission rate
    gamma = recovery rate
    dy/dt = (dS/dt, dI/dt, dR/dt) the system of ODE in SIR model
    """
    dSdt = -beta*y_t[0]*y_t[1]/np.sum(y_t) #np.sum(y_t) = N
    dRdt = gamma*y_t[1]
    dIdt = -(dSdt + dRdt)
    
    return dSdt, dIdt, dRdt



#China Fitting
N = 1393000000 # china prpulation from google
ydata = I_t_china
xdata = np.arange(len(ydata))

inf0 = ydata[0]
sus0 = N-inf0
rec0 = 0.0
y_0 = (sus0, inf0, rec0)

def fit_odeint(t, beta, gamma):
    """
    a tool used to sovle a system of ODEs.
    """
    return integrate.odeint(dydt, y_0, t, args=(beta, gamma))[:, 1]


popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
yhat = fit_odeint(xdata, *popt)

t_domain = np.arange(0, 150, 1)
I_t = fit_odeint(t_domain, *popt)

plt.figure()
plt.plot(xdata, ydata, 'o', label='actual')
plt.plot(t_domain, I_t, label='prediction')
plt.legend()
plt.title('beta:%f, gamma:%f'%(popt[0], popt[1]))
plt.show()

#Using gamma of China fitting 
gamma = popt[1]
N = 51640000 # china prpulation from google
ydata = I_t_korea
xdata = np.arange(len(ydata))

inf0 = ydata[0]
sus0 = N-inf0
rec0 = 0.0
y_0 = (sus0, inf0, rec0)

def fit_odeint(t, beta):
    """
    a tool used to sovle a system of ODEs.
    """
    return integrate.odeint(dydt, y_0, t, args=(beta, gamma))[:, 1]


popt1, pcov1 = optimize.curve_fit(fit_odeint, xdata, ydata)
yhat = fit_odeint(xdata, *popt)

t_domain = np.arange(0, 15000, 1)
I_t = fit_odeint(t_domain, *popt1)

plt.figure()
plt.plot(xdata, ydata, 'o', label='actual')
plt.plot(t_domain, I_t, label='prediction')
plt.legend()
plt.title('beta:%f, gamma:%f'%(popt[0], popt[1]))
plt.show()

















