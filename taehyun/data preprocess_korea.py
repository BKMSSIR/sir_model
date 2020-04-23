# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:20:23 2020

@author: TaeHyun Hwang
"""

import numpy as np
import matplotlib.pyplot as plt

f = open('./kaggle_covid_kor/Time.csv', 'r')
data = []
for line in f.readlines():
    data.append(line.split(','))
f.close()

I, R, D = [], [], []
for i in range(1, len(data)):
    I.append(float(data[i][4]))
    R.append(float(data[i][5]))
    D.append(float(data[i][6]))

I = np.array(I)
R = np.array(R)
D = np.array(D)

plt.figure()
plt.plot(I, label='Infection')
plt.plot(R, label='Released')
plt.plot(D, label='Decesed')
plt.legend()
plt.show()

I_t = I - (R+D)
R_t = R+D

plt.figure()
plt.plot(I_t, label='# of infected people at time t')
plt.plot(R_t, label='# of recovered people at time t')
plt.legend()
plt.show()

np.save('./I_t_kor.npy', I_t)
np.save('./R_t_kor.npy', R_t)

