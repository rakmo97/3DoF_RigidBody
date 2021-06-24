# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:22:37 2020

@author: Omkar
"""

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
from keras.metrics import RootMeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.optimizers import sgd
from keras.callbacks import EarlyStopping
from keras import Input
from keras import Model

# import mat73

# Load in training and testing data
print("Loading mat file")
matfile = loadmat('ANN2_data.mat')

Xfull = matfile['Xfull_2']
tfull = matfile['tfull_2']
X_train = matfile['Xtrain2'].reshape(-1,7)
t_train = matfile['ttrain2']
X_test = matfile['Xtest2'].reshape(-1,7)
t_test = matfile['ttest2']

# Xfull = matfile['Xfull_2'].reshape(-1,7)
# tfull = matfile['tfull_2'][:,2].reshape(-1,1)
# X_train = matfile['Xtrain2'].reshape(-1,7)
# t_train = matfile['ttrain2'][:,2].reshape(-1,1)
# X_test = matfile['Xtest2'].reshape(-1,7)
# t_test = matfile['ttest2'][:,2].reshape(-1,1)



# Shuffle data
shuffler = np.random.permutation(len(X_train))
X_train_shuffled = X_train[shuffler]
t_train_shuffled = t_train[shuffler]
    
activation = "relu"
# activation = "tanh"
#
# n_neurons = 2000
# n_neurons = 250
n_neurons = 75


# Define ANN Architecture
TF = Sequential()
# TF.add(layers.BatchNormalization())
TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal',input_dim=7))
TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# TF.add(layers.Dense(25, activation=activation,kernel_initializer='normal'))
# TF.add(layers.BatchNormalization())
TF.add(layers.Dense(t_train.shape[1], activation='linear',kernel_initializer='normal'))


# Compile ANN
opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
# opt = sgd(learning_rate=0.01, momentum=0)
# opt = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)♣

es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

TF.compile(optimizer=opt, loss='mean_squared_error', metrics = ["mean_squared_error"])
# TF.compile(optimizer=opt, loss='mean_absolute_error', metrics = ["mean_absolute_error"])

#Fit ANN
TF.fit(X_train_shuffled, t_train_shuffled, batch_size=100, epochs=10000, validation_split=0.05,callbacks=[es])

# Evaluating model
results = TF.evaluate(X_test,t_test)
print("Test Loss: ", results[0])
print("Test Accuracy: ", results[1])


# Plotting histories
plt.figure(1)
plt.plot(TF.history.history['loss'])
plt.plot(TF.history.history['val_loss'])
plt.legend(['train', 'validation'], loc='best')
plt.title('Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')

plt.figure(2)
plt.plot(TF.history.history['mean_squared_error'])
plt.plot(TF.history.history['val_mean_squared_error'])
# plt.plot(TF.history.history['accuracy'])
# plt.plot(TF.history.history['val_accuracy'])
plt.legend(['train', 'validation'], loc='best')
plt.title('Metric')

i = 5;
y_test = TF.predict(X_test[i].reshape(1,-1))
# y_test = TF.predict(X_test[i].reshape(1,-1,1))
print("X_test[i] = ", X_test[i])
print("t_test[i] = ", t_test[i])
print("y_test[i] = ", y_test)


# plt.figure(3)
# plt.plot(Xfull[:,0],tfull[:,0],'.')
# Save model
print("\nSaving ANN!")
saveout_filename = "ANN2_703_{}_n{}.h5".format(activation,n_neurons)
print('Filename: ' + saveout_filename)
TF.save(saveout_filename)

#

# Plotting
print('Plotting')
# idxs = range(000,X_test.shape[0])
idxs = range(000,300)
# yvis = TF.predict(X_test[idxs].reshape(-1,7));
yvis = TF.predict(X_test[idxs].reshape(-1,7));



plt.figure(3)
plt.subplot(311)
plt.plot(t_test[idxs,0])
plt.plot(yvis[:,0])
plt.xlabel('Index (-)')
plt.ylabel('Tx (N)')
plt.legend(['ocl','ann'])
plt.subplot(312)
plt.plot(t_test[idxs,1])
plt.plot(yvis[:,1])
plt.xlabel('Index (-)')
plt.ylabel('Ty (N)')
plt.subplot(313)
plt.plot(t_test[idxs,2])
plt.plot(yvis[:,2])
plt.xlabel('Index (-)')
plt.ylabel('Tz (N)')
plt.tight_layout()


# plt.figure(3)

# plt.plot(t_test[idxs])
# plt.plot(yvis)
# plt.xlabel('Index (-)')
# plt.ylabel('Tx (N)')
# plt.legend(['ocl','ann'])

