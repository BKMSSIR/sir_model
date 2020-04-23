# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:27:14 2020

@author: TaeHyun Hwang
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize

f = open('./kaggle_covid_kor/Time.csv', 'r')
data = []
for line in f.readlines():
    data.append(line.split(','))
f.close()

I_cum, R_cum, D_cum = [], [], [] #cum = cummulative number
for i in range(1, len(data)):
    I_cum.append(float(data[i][4]))
    R_cum.append(float(data[i][5]))
    D_cum.append(float(data[i][6]))

I_cum = np.array(I_cum)
R_cum = np.array(R_cum)
D_cum = np.array(D_cum)

plt.figure()
plt.plot(I_cum, label='# of Cummulative Infection')
plt.plot(R_cum, label='# of Cummulative Released')
plt.plot(D_cum, label='# of Cummulative Decesed')
plt.legend(loc = 'upper left')
plt.show()

I_active = [1]
for i in range(len(I)-1):
    I_active.append(I[i+1] - I[i])

I_active = np.array(I_active)

I_t = I - (R+D)

plt.figure()
plt.plot(I_t, label='# of infected people at time t')
#plt.plot(I_active, label='# of daily confirmed case at time t')
plt.legend()
plt.show()



for t in range(10, 70, 10):
    population = 51640000 #한국인구 5164만 2018년 기준, 구글출처
    ydata = I_t[:t]
    xdata = np.arange(len(ydata))
    
    N = population
    inf0 = ydata[0]
    sus0 = N-inf0
    rec0 = 0.0
    gamma = 0.8
    
#    def sir_model(y, x, beta, gamma):
#        sus = -beta*y[0]*y[1]/N
#        rec = gamma*y[1]
#        inf = -(sus + rec)
#        return sus, inf, rec

    def sir_model(y, x, beta):
        sus = -beta*y[0]*y[1]/N
        rec = gamma*y[1]
        inf = -(sus + rec)
        return sus, inf, rec

    def fit_odeint(x, beta):
        return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta,))[:, 1]

    
#    def fit_odeint(x, beta, gamma):
#        return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:, 1]
    
#    popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata, bounds=([0,0], [1,1]))
    popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
    fitted = fit_odeint(xdata, *popt)
    
    t_lin = np.arange(0,100,1)
    t_fitted = fit_odeint(t_lin, *popt)
    
    plt.figure()
    plt.plot(xdata, ydata, 'o', label='actual data')
    plt.plot(t_lin, t_fitted, label='SIR prediction')
    plt.legend()
#    plt.title('until %d, beta: %f, gamma: %f'%(t, popt[0], popt[1]))
    plt.title('until %d, beta: %f'%(t, popt[0]))
    plt.savefig('./fig%d.png'%t)
    plt.show()




