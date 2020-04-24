# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 20:21:32 2020

@author: TaeHyun Hwang
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

I_t_kor = np.load('./I_t_kor.npy')[20:]
R_t_kor = np.load('./R_t_kor.npy')[20:]
N = 51640000

t = np.arange(len(I_t_kor), dtype=float)
I_t = I_t_kor/N
R_t = R_t_kor/N
S_t = 1 - (I_t+R_t)

plt.figure()
plt.plot(S_t, label='S_t')
plt.legend()
plt.show()

plt.figure()
plt.plot(I_t, label='I_t')
plt.plot(R_t, label='R_t')
plt.legend()
plt.show()


S_net = tf.keras.Sequential()
S_net.add(tf.keras.layers.Dense(units=256, activation='tanh', input_shape=(1,)))
S_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
S_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
S_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
S_net.add(tf.keras.layers.Dense(units=1, activation='softplus'))

I_net = tf.keras.Sequential()
I_net.add(tf.keras.layers.Dense(units=256, activation='tanh', input_shape=(1,)))
I_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
I_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
I_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
I_net.add(tf.keras.layers.Dense(units=1, activation='softplus'))

R_net = tf.keras.Sequential()
R_net.add(tf.keras.layers.Dense(units=256, activation='tanh', input_shape=(1,)))
R_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
R_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
R_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
R_net.add(tf.keras.layers.Dense(units=1, activation='softplus'))

beta_net = tf.keras.Sequential()
beta_net.add(tf.keras.layers.Dense(units=256, activation='tanh', input_shape=(1,)))
beta_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
beta_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
beta_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
beta_net.add(tf.keras.layers.Dense(units=1, activation='softplus'))

gamma_net = tf.keras.Sequential()
gamma_net.add(tf.keras.layers.Dense(units=256, activation='tanh', input_shape=(1,)))
gamma_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
gamma_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
gamma_net.add(tf.keras.layers.Dense(units=256, activation='tanh'))
gamma_net.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model_concat = tf.keras.layers.concatenate([S_net.output, I_net.output, R_net.output, beta_net.output, gamma_net.output], axis=-1)





#######################################################################################################################
def numeric_gradient(f, x, h):
    xh = np.array([[x+h]])
    x_h = np.array([[x-h]])
    return (f(xh) - f(x_h))/(2*h)


def gradient(f):
    with tf.GradientTape() as t:
        t.watch()
with tf.GradientTape() as t:
    t.watch(S_net)
    z =S_net(np.array([[0]]))
    
dz_dx = t.gradient(z, x)
print(dz_dx)

model = tf.keras.Sequential()
model.add(tf.keras.layers.concatenate([S_net.output, I_net.output], axis=-1))
concat = tf.keras.layers.concatenate([S_net.output, I_net.output], axis=-1)
model.add(concat)

def neural_ode(x):
    dS_dt = numeric_gradient(S_net, x, 1e-5)
    dI_dt = numeric_gradient(I_net, x, 1e-5)
    dR_dt = numeric_gradient(R_net, x, 1e-5)
    
    return np.array([dS_dt, dI_dt, dR_dt, beta_net(x), gamma_net(x)])

def loss_fuc(y_pred, y_true):
    """
    y_pred = np.array([dS_dt, dI_dt, dR_dt, beta_net(x), gamma_net(x)])
    y_ture = np.array([S(t), I(t), R(t)])
    """
    eq1 = (y_pred[0] + y_pred[3]*y_true[0]*y_true[1])**2
    eq2 = (y_pred[1] + y_pred[4] - y_pred[3]*y_true[0]*y_true[1] )**2
    eq3 = (y_pred[2] - y_pred[4]*y_true[1])**2
    eq4 = ((S_net(np.array([[0.0]])) - S_t[0]))**2
    eq5 = ((I_net(np.array([[0.0]])) - I_t[0]))**2
    eq6 = ((R_net(np.array([[0.0]])) - R_t[0]))**2
    
    return eq1+eq2+eq3+eq4+eq5+eq6

X_train = np.array(t).reshape(-1,1)
y_train = np.concatenate((S_t.reshape(-1,1), I_t.reshape(-1,1), R_t.reshape(-1,1)), axis=1)


S_net.predict([0])




