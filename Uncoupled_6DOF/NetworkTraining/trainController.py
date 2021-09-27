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
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
# from keras.optimizers import Nadam
# from keras.optimizers import sgd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
from tensorflow.keras import Model
import tensorflow_addons as tfa

# import mat73

# Load in training and testing data
print("Loading mat file")
matfile = loadmat('ANN2_data.mat')
# matfile = loadmat('ANN2_decoupled_data.mat')


# matfile = loadmat('ANN1_data_notaug.mat')
Xfull = matfile['Xfull_2']
tfull = matfile['tfull_2']
X_train = matfile['Xtrain2'].reshape(-1,13)
t_train = matfile['ttrain2']
X_test = matfile['Xtest2'].reshape(-1,13)
t_test = matfile['ttest2']

# Choose Ouptuts to predict
out_idxs = range(0,6)
tfull = tfull[:,out_idxs]
t_train = t_train[:,out_idxs]
t_test = t_test[:,out_idxs]

# Shuffle data
shuffler = np.random.permutation(len(X_train))
X_train_shuffled = X_train[shuffler]
t_train_shuffled = t_train[shuffler]
    

# Get input/output dimensions
n_in = Xfull.shape[1]
n_out = tfull.shape[1]

# Split data to train and test
# X_train, X_test, t_train, t_test = train_test_split(Xfull, tfull, test_size=0.01, random_state=42)

activation = "relu"
# activation = "tanh"

n_neurons = 1000
# n_neurons = 50
# n_neurons = 25
n_neuronsA = 750
n_neuronsB = 750

# Define ANN Architecture
inputA = Input(shape=(n_in,))

x = layers.Lambda(lambda w: w)(inputA)
x = Model(inputs=inputA, outputs=x)

y = tfa.layers.SpectralNormalization(layers.Dense(n_neuronsA, activation=activation,kernel_initializer='normal'))(x.output)
# y = layers.Dense(n_neuronsA, activation=activation,kernel_initializer='normal')(x.output)
# y =  tfa.layers.SpectralNormalization(layers.Dense(n_neuronsA, activation=activation,kernel_initializer='normal'))(y)
# y = layers.Dense(n_neuronsA, activation=activation,kernel_initializer='normal')(y)
y = layers.Dense(3, activation='linear',kernel_initializer='normal')(y)

z = layers.BatchNormalization()(x.output)
# z = layers.Dense(n_neuronsB, activation=activation,kernel_initializer='normal')(x.output)
z = tfa.layers.SpectralNormalization(layers.Dense(n_neuronsB, activation=activation,kernel_initializer='normal'))(z)
# z = tfa.layers.SpectralNormalization(layers.Dense(n_neuronsB, activation=activation,kernel_initializer='normal'))(z)
# z = tfa.layers.SpectralNormalization(layers.Dense(n_neuronsB, activation=activation,kernel_initializer='normal'))(z)
# z = layers.Dense(n_neuronsB, activation=activation,kernel_initializer='normal')(z)
# z = layers.Dense(n_neuronsB, activation=activation,kernel_initializer='normal')(z)
z = layers.Dense(3, activation='linear',kernel_initializer='normal')(z)

TF = Model(inputs=x.input, outputs=[y,z])


# Compile ANN
opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
# opt = sgd(learning_rate=0.01, momentum=0)
# opt = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)♣

es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

TF.compile(optimizer=opt, loss='mean_squared_error', metrics = ["mean_squared_error"])

#Fit ANN
# TF.fit(X_train, t_train, batch_size=100, epochs=10000, validation_split=0.05,callbacks=[es])
# TF.fit(X_train, [t_train[:,0:3],t_train[:,3:6]], batch_size=100, epochs=10000, validation_split=0.05,callbacks=[es])
history = TF.fit(X_train_shuffled, [t_train_shuffled[:,0:3],t_train_shuffled[:,3:6]], batch_size=100, epochs=10000, validation_split=0.05,callbacks=[es])
# history = TF.fit(X_train_shuffled, [t_train_shuffled[:,0:3],t_train_shuffled[:,3:6]], batch_size=100, epochs=1, validation_split=0.05,callbacks=[es])

# Evaluating model
results = TF.evaluate(X_test,[t_test[:,0:3],t_test[:,3:6]])
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
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
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
saveout_filename = "ANN2_{}_n{}_nA{}_nB{}.h5".format(activation,n_neurons,n_neuronsA,n_neuronsB)
print('Filename: ' + saveout_filename)
TF.save(saveout_filename)

#

# Plotting
print('Plotting')
idxs = range(000,300)
yvis = TF.predict(X_test[idxs].reshape(-1,n_in));
yvis = np.hstack((yvis[0],yvis[1]));

plt.figure(3)
plt.subplot(231)
plt.plot(t_test[idxs,0])
plt.plot(yvis[:,0],'--')
plt.xlabel('Index (-)')
plt.ylabel('Fx (N)')
plt.legend(['ocl','ann'])
plt.subplot(232)
plt.plot(t_test[idxs,1])
plt.plot(yvis[:,1],'--')
plt.xlabel('Index (-)')
plt.ylabel('Fy (N)')
plt.tight_layout()
plt.subplot(233)
plt.plot(t_test[idxs,2])
plt.plot(yvis[:,2],'--')
plt.xlabel('Index (-)')
plt.ylabel('Fz (N)')
plt.tight_layout()
plt.subplot(234)
plt.plot(t_test[idxs,3])
plt.plot(yvis[:,3],'--')
plt.xlabel('Index (-)')
plt.ylabel('L (Nm)')
plt.tight_layout()
plt.subplot(235)
plt.plot(t_test[idxs,4])
plt.plot(yvis[:,4],'--')
plt.xlabel('Index (-)')
plt.ylabel('M (Nm)')
plt.tight_layout()
plt.subplot(236)
plt.plot(t_test[idxs,5])
plt.plot(yvis[:,5],'--')
plt.xlabel('Index (-)')
plt.ylabel('N (Nm)')
plt.suptitle('Openloop Test')
plt.tight_layout()

# plt.figure(3)
# plt.subplot(131)
# plt.plot(t_test[idxs,0])
# plt.plot(yvis[:,0])
# plt.xlabel('Index (-)')
# plt.ylabel('L (Nm)')
# plt.tight_layout()
# plt.subplot(132)
# plt.plot(t_test[idxs,1])
# plt.plot(yvis[:,1])
# plt.xlabel('Index (-)')
# plt.ylabel('M (Nm)')
# plt.tight_layout()
# plt.subplot(133)
# plt.plot(t_test[idxs,2])
# plt.plot(yvis[:,2])
# plt.xlabel('Index (-)')
# plt.ylabel('N (Nm)')
# plt.tight_layout()
