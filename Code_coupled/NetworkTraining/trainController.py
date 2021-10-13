# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:22:37 2020

@author: Omkar
"""

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
from keras.metrics import RootMeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers import Nadam
# from tensorflow.keras.optimizers import sgd
from tensorflow.keras.callbacks import EarlyStopping

# import mat73

# Load in training and testing data
print("Loading mat file")
matfile = loadmat('ANN2_data.mat')
# matfile = loadmat('ANN2_decoupled_data.mat')


# matfile = loadmat('ANN1_data_notaug.mat')
Xfull = matfile['Xfull_2']
tfull = matfile['tfull_2']
X_train = matfile['Xtrain2'].reshape(-1,7)
t_train = matfile['ttrain2']
# t_train = matfile['ttrain2'][:,1]
X_test = matfile['Xtest2'].reshape(-1,7)
t_test = matfile['ttest2']
# t_test = matfile['ttest2'][:,1]

# Split data to train and test
# X_train, X_test, t_train, t_test = train_test_split(Xfull, tfull, test_size=0.01, random_state=42)

activation = "relu"
# activation = "tanh"

# n_neurons = 750
n_neurons = 100


# Define ANN Architecture
TF = Sequential()
TF.add(layers.BatchNormalization())
TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal',input_dim=7))
TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# TF.add(layers.Dense(n_neurons, activation=activation,kernel_initializer='normal'))
# TF.add(layers.Dense(25, activation=activation,kernel_initializer='normal'))
TF.add(layers.Dense(2, activation='linear',kernel_initializer='normal'))

# Compile ANN
opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
# opt = sgd(learning_rate=0.01, momentum=0)
# opt = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)♣

es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

TF.compile(optimizer=opt, loss='mean_squared_error', metrics = ["mean_squared_error"])

#Fit ANN
history = TF.fit(X_train, t_train, batch_size=100, epochs=10000, validation_split=0.05,callbacks=[es])

# Evaluating model
results = TF.evaluate(X_test,t_test)
print("Test Loss: ", results[0])
print("Test Accuracy: ", results[1])

plt.close('all')

# Plotting histories
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validation'], loc='best')
plt.title('Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')

plt.figure(2)
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
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
yvis = TF.predict(X_test[idxs].reshape(-1,7));
# yvis = TF.predict(X_test[300:400].reshape(-1,12,1));
# yvis = TF.predict(X_test[200:300]);


plt.figure(3)
plt.subplot(211)
plt.plot(t_test[idxs,0])
plt.plot(yvis[:,0])
plt.xlabel('Index (-)')
plt.ylabel('Tx (N)')
plt.legend(['ocl','ann'])
plt.subplot(212)
plt.plot(t_test[idxs,1])
plt.plot(yvis[:,1])
plt.xlabel('Index (-)')
plt.ylabel('Ty (N)')
plt.tight_layout()

# plt.figure(3)

# plt.plot(t_test[idxs])
# plt.plot(yvis)
# plt.xlabel('Index (-)')
# plt.ylabel('Tx (N)')
# plt.legend(['ocl','ann'])

