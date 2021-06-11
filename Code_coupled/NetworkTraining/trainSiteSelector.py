# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:22:37 2020

@author: Omkar
"""

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras import layers
from keras.layers import Dense
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
from keras.layers import concatenate


# import mat73

# Load in training and testing data
print("Loading mat file")
matfile = loadmat('ANN1_data.mat')


# matfile = loadmat('ANN1_data_notaug.mat')
Xtrain_1A = matfile['Xtrain_1A']
Xtrain_1B = matfile['Xtrain_1B'].reshape(-1,200,2,1)
ttrain1 = matfile['ttrain1']
Xtest_1A = matfile['Xtest_1A']
Xtest_1B = matfile['Xtest_1B'].reshape(-1,200,2,1)
ttest1 = matfile['ttest1']
surfXtrain = matfile['surfXtrain']
surfYtrain = matfile['surfYtrain']
surfXtest = matfile['surfXtest']
surfYtest = matfile['surfYtest']


activation = "relu"
# activation = "tanh"

n_neurons = 150


# Define ANN Architecture
inputA = Input(shape=(9,))
inputB = Input(shape=(200,2,1))

x = Dense(20, activation="relu")(inputA)
x = Dense(20, activation="relu")(x)
x = Dense(20, activation="relu")(x)
x = Model(inputs=inputA, outputs=x)

# y = layers.Conv2D(64, (2, 2), padding="same", activation="relu")(inputB)
y = layers.Conv2D(64, (2, 2), padding="same", activation="relu")(inputB)
y = layers.Flatten()(y)
y = Dense(40, activation="relu")(y)
y = Dense(40, activation="relu")(y)
y = Model(inputs=inputB, outputs=y)

combined = concatenate([x.output, y.output])

z = Dense(20, activation="relu")(combined)
z = Dense(20, activation="relu")(z)
z = Dense(20, activation="relu")(z)
z = Dense(20, activation="relu")(z)
z = Dense(20, activation="relu")(z)
z = Dense(2, activation="linear")(z)

TF = Model(inputs=[x.input, y.input], outputs=z)

# Compile ANN
opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
# opt = sgd(learning_rate=0.01, momentum=0)
# opt = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)♣

es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)

TF.compile(optimizer=opt, loss='mean_squared_error', metrics = ["mean_squared_error"])

#Fit ANN
TF.fit(x=[Xtrain_1A, Xtrain_1B], y=ttrain1, batch_size=1, epochs=10000, validation_split=0.05,callbacks=[es])

# Evaluating model
results = TF.evaluate(x=[Xtest_1A, Xtest_1B], y=ttest1)
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
y_test = TF.predict([Xtest_1A[i].reshape(1,-1),Xtest_1B[i].reshape(1,200,2,1)])
# y_test = TF.predict(X_test[i].reshape(1,-1,1))
print("X_test[i] = ", Xtest_1A[i])
print("t_test[i] = ", ttest1[i])
print("y_test[i] = ", y_test)


plt.figure(3)
plt.plot(surfXtest[i,:],surfYtest[i,:])
plt.plot(y_test[0,0],y_test[0,1],'*')
plt.plot(ttest1[i,0],ttest1[i,1],'+')
plt.plot(Xtest_1A[i,0],Xtest_1A[i,1],'x')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.legend(['Surface','ANN Predicted Target','OCL Optimized Target','Objective'])


# Save model
print("\nSaving ANN!")
saveout_filename = 'ANN1_'+activation+'.h5'
TF.save(saveout_filename)

#

# Plotting
print('Plotting')
# idxs = range(000,300)
idxs = range(000,ttest1.shape[0]-1)
yvis = TF.predict([Xtest_1A[idxs].reshape(-1,9),Xtest_1B.reshape(-1,200,2,1)]);
# yvis = TF.predict(X_test[300:400].reshape(-1,12,1));
# yvis = TF.predict(X_test[200:300]);


plt.figure(4)
plt.subplot(221)
plt.plot(ttest1[idxs,0])
plt.plot(yvis[:,0])
plt.xlabel('Index (-)')
plt.ylabel('X (m)')
plt.legend(['ocl','ann'])
plt.subplot(222)
plt.plot(ttest1[idxs,1])
plt.plot(yvis[:,1])
plt.xlabel('Index (-)')
plt.ylabel('Y (m)')
plt.tight_layout()

# plt.figure(3)

# plt.plot(t_test[idxs])
# plt.plot(yvis)
# plt.xlabel('Index (-)')
# plt.ylabel('Tx (N)')
# plt.legend(['ocl','ann'])

