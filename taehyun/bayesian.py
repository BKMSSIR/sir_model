# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:13:32 2020

@author: TaeHyun Hwang
"""

import numpy as np
from bayes_opt import BayesianOptimization #need to install using pip: pip install bayesian-optimization
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from scipy import integrate
import matplotlib.pyplot as plt

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

N = 51640000 # china prpulation from google
ydata_I = I_t_korea
ydata_R = R_t_korea
xdata = np.arange(len(ydata_I))

inf0 = ydata_I[0]
sus0 = N-inf0
rec0 = ydata_R[0]
y_0 = (sus0, inf0, rec0)

def fit_odeint(t, beta, gamma):
    """
    a tool used to sovle a system of ODEs.
    """
    return integrate.odeint(dydt, y_0, t, args=(beta, gamma))[:, 1]

def black_box(p, q):
    I_hat = fit_odeint(xdata, beta=p, gamma=q)
#    R_hat = fit_odeint(xdata, beta=p, gamma=q)[:, 1]
    
#    Loss = np.sqrt(np.sum((I_hat - I_t_korea)**2) + np.sum((R_hat - R_t_korea)**2))
    Loss = np.sqrt(np.sum((I_hat - I_t_korea)**2))
    
    return -1*Loss

#Bayesian Optimization
pbounds = {'p':(0,100), 'q':(0, 100)}
optimizer = BayesianOptimization(f = black_box, pbounds=pbounds, random_state=1)
optimizer.maximize(n_iter=100)
beta_hat = optimizer.max['params']['p']
gamma_hat = optimizer.max['params']['q']

print(beta_hat, gamma_hat)

yhat = fit_odeint(xdata, beta_hat, gamma_hat)

t_domain = np.arange(0, 150, 1)
I_t = fit_odeint(t_domain, beta_hat, gamma_hat)
R_t = fit_odeint(t_domain, beta_hat, gamma_hat)[:, 1]

plt.figure()
plt.plot(xdata, ydata_I, 'o', label='actual infection')
plt.plot(xdata, ydata_R, '^', label='actual recovery')
plt.plot(t_domain, I_t, label='infection prediction')
#plt.plot(t_domain, R_t, label='recovery prediction')
plt.legend()
plt.title('beta:%f, gamma:%f'%(beta_hat, gamma_hat))
plt.show()
















