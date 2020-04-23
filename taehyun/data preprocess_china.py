# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:53:04 2020

@author: TaeHyun Hwang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#china data preprocess
#f = open('./kaggle_covid_world/covid_19_data.csv', 'r')
#data_china = []
#for line in f.readlines():
#    data_china.append(line.split(','))
#f.close()

#Using dataframe of pandas
df_china = pd.read_csv('./kaggle_covid_world/covid_19_data.csv')
df_china = df_china[df_china['Country/Region'] == 'Mainland China'] #China 관련 데이터만
#날짜별로 숫자를 합침
df_china = df_china[['ObservationDate', 'Confirmed', 'Deaths', 'Recovered']].groupby(df_china['ObservationDate']).sum()

cum_I_china = np.array(df_china['Confirmed']) # # of cummulative infected people in china
cum_D_china = np.array(df_china['Deaths']) # # of cummulative deceased people in china
cum_R_china = np.array(df_china['Recovered']) # #of cummulative recovered people in china

#plotting
plt.figure()
plt.plot(cum_I_china, label='infected')
plt.plot(cum_D_china, label='deceased')
plt.plot(cum_R_china, label='recovered')
plt.legend()
plt.title('cummulative')
plt.show()

#active number at time t
I_t_china = cum_I_china - (cum_D_china + cum_R_china)
R_t_china = cum_R_china + cum_D_china

#plotting
plt.figure()
plt.plot(I_t_china, label='infection at t')
plt.plot(R_t_china, label='recovery at t')
plt.legend()
plt.show()

#save the ndarray as npy format file.
np.save('./I_t_china.npy', I_t_china)
np.save('./R_t_china.npy', R_t_china)