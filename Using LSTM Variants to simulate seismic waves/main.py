# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:15:17 2024

@author: jimmy
"""

# Import the libraries
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt  # for 畫圖用
from obspy import read
import DataPreprocess as dp
from scipy.signal import find_peaks
import time
import random

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')

#%load_ext tensorboard
#%tensorboard --logdir logs

#%%
def CreateSet(dataset_scaled, time_steps):
    
    X = []   
    Y = []

    N = len(dataset_scaled)

    for i in range(time_steps, N):
        X.append(dataset_scaled[i - time_steps : i, 0])
        Y.append(dataset_scaled[i, 0])
        
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, Y
#%%
# fix random seed for reproducibility
random.seed(7)
tf.random.set_seed(7)
np.random.seed(7)

# parameters for model
layer_units = 60
time_steps = 60
input_sizes = 1 
output_sizes = 1
batch_sizes = 128
n_epochs = 10
n_layers = 2
activateType = 'relu'


# parameters for data
rangeTime = 0.5

#%%
import os
from sklearn.preprocessing import MinMaxScaler
import Models as md


path = './ED_TRAIN/'
folder = os.listdir(path)

modelType, model = md.BILSTM(time_steps, input_sizes, output_sizes, layer_units, n_layers, activateType)
sc = MinMaxScaler(feature_range = (-1, 1))

isFit = False
numTrain = 0
stopNum = 5

start_time = time.time()

for file in folder :
    
    numTrain += 1
    print(numTrain)
    print(file)
    
    #read stream data
    file_path = path + file
    rawData_train = read(file_path)
    
    trace_train = dp.FindTrace(rawData_train)
    
    #dataframe of dataset of train
    datasframe_train = dp.Data_Trim_Right(trace_train, rangeTime)
    
    #data of dataset of train
    dataset_train = datasframe_train.iloc[:, 0:1].values
    
    # Feature Scaling
    if(not isFit):
        dataset_train_scaled = sc.fit_transform(dataset_train)
        isFit = True
    else:
        dataset_train_scaled = sc.transform(dataset_train)
        
    X_train, Y_train = CreateSet(dataset_train_scaled, time_steps)
    
    #tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    #history = model.fit(X_train, Y_train, epochs = epochNum, batch_size = batchNum, callbacks=[tf_callback])
    
    #training model
    history = model.fit(X_train, Y_train, epochs = n_epochs, batch_size = batch_sizes, shuffle=False)
    
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    if(numTrain == stopNum):
        break;
    
#%%
import os

path = './ED_TEST/'
folder = os.listdir(path)

num = 0
rmseAvg = 0
lossAvg = 0
stopNum = 5

for file in folder :
    
    num += 1
    print(num)
    print(file)
    
    #read stream data
    file_path = path + file
    rawData_test = read(file_path)
    
    trace_test = dp.FindTrace(rawData_test)
    
    #dataframe of dataset of train
    datasframe_test = dp.Data_Trim_Right(trace_test, rangeTime)
    
    #data of dataset of train
    dataset_test = datasframe_test.iloc[:, 0:1].values
    
    dataset_test = dataset_test.reshape(-1,1)
        
    dataset_test_scaled = sc.transform(dataset_test) # Feature Scaling

    X_test, Y_test = CreateSet(dataset_test_scaled, time_steps)
    
    
    
    
    # predict
    dataset_predicted = model.predict(X_test)
    
    
    dataset_predicted = sc.inverse_transform(dataset_predicted)  # to get the original scale
    
    
    
    
    #shifts
    average_test = np.average(dataset_test)
    average_predicted = np.average(dataset_predicted)
    dataset_predicted -= average_predicted
    dataset_predicted += average_test
    
    
    
    
    
    # calculate 均方根誤差(root mean squared error)
    #可以認為迴歸效果相比真實值平均相差的量
    from sklearn.metrics import mean_squared_error

    rmse = np.sqrt(mean_squared_error(dataset_test[time_steps:], dataset_predicted))
    print('Train Score: %.2f RMSE' % (rmse))

    score = model.evaluate(X_test, Y_test, batch_size=batch_sizes, verbose=1)
    print('loss value = {}'.format(score))
    print('metrics values = {}'.format(score[1]))
    
    
    
    
    dt = 100
    # Visualising the results
    
    px = np.arange(0, len(dataset_test) + 1)
    px_time = np.arange(0, len(dataset_test)*0.01, 0.01)
    
    if(len(px) > len(dataset_test)):
        px = px[:-1]
    if(len(px_time) > len(dataset_test)):
        px_time = px_time[:-1]
    
    plt.figure(figsize=(16,9), dpi=72)
    plt.plot(px_time[:], dataset_test, color = 'red', label = 'Real Seismic Wave')  # 紅線表示真實
    plt.plot(px_time[time_steps:], dataset_predicted, color = 'blue', alpha=0.5, label = 'Simulated Seismic Wave')  # 藍線表示預測
    plt.title('{} Seismic Wave Simulation'.format(modelType),loc = 'center')
    plt.title('Data : ' + str(numTrain) + '  Train Score: %.2f RMSE' % (rmse), loc = 'left')
    plt.title('epochs : ' + str(n_epochs) + '   batch_size : ' + str(batch_sizes) + '  Final Loss: %.6f' % (history.history["loss"][len(history.history["loss"])-1]), loc = 'right')
    plt.xlabel('Time(Second)', loc = 'center')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    plotSize = 2
    plt.figure(figsize=(16,9), dpi=72)
    
    ds = 0.01
    
    freq_X, intel_Y = dp.DPFFT(ds, dataset_test[time_steps+1:].flatten())
    N = len(freq_X)
    #intel_Y = intel_Y.flatten()
    plt.plot(freq_X[:N//plotSize], 2/N * np.abs(intel_Y[:N//plotSize]), linestyle='-', color='red', label = 'Real Seismic Wave Frequency Spectrum')
    
    freq_Xs, intel_Ys = dp.DPFFT(ds, dataset_predicted[1:].flatten())
    N = len(freq_Xs)
    #intel_Y = intel_Y.flatten()
    rmse = np.sqrt(mean_squared_error(2/N * np.abs(intel_Y[:N//plotSize]), 2/N * np.abs(intel_Ys[:N//plotSize])))
    
    plt.plot(freq_Xs[:N//plotSize], 2/N * np.abs(intel_Ys[:N//plotSize]), linestyle='-', color='blue', alpha=0.5, label = 'Simulated Seismic Wave Frequency Spectrum')
    plt.title('Train Score: %.2f RMSE' % (rmse), loc = 'left')
    plt.title('Frequency Spectrum',loc = 'center')
    plt.xlabel('Frequency (Hz)', loc = 'center')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()
    
    
    maxIndex = np.argmax(dataset_test)
    dr = 1
    
    rmse = np.sqrt(mean_squared_error(dataset_test[maxIndex - dr * dt + time_steps : maxIndex + dr * dt], dataset_predicted[maxIndex - dr * dt : maxIndex + dr * dt - time_steps]))
    print("rmse : {}".format(rmse))
    rmseAvg += rmse
    lossAvg += score[0]
    
    plt.figure(figsize=(16,9), dpi=72)
    plt.plot(px_time[maxIndex - dr * dt + time_steps : maxIndex + dr * dt], dataset_test[maxIndex - dr * dt + time_steps : maxIndex + dr * dt], color = 'red', label = 'Real Seismic Wave')  # 紅線表示真實
    plt.plot(px_time[maxIndex - dr * dt + time_steps: maxIndex + dr * dt], dataset_predicted[maxIndex - dr * dt : maxIndex + dr * dt - time_steps], color = 'blue', alpha=0.5, label = 'Simulated Seismic Wave')  # 藍線表示預測
    plt.title('{} Seismic Wave Simulation Zoom In'.format(modelType),loc = 'center')
    plt.title('Data : ' + str(numTrain) + '  Train Score: %.2f RMSE' % (rmse), loc = 'left')
    #plt.title('epochs : ' + str(n_epochs) + '   batch_size : ' + str(batch_sizes) + '  Loss: %.6f' % (score[0]), loc = 'right')
    plt.xlabel('Time(Second)', loc = 'center')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    plotSize = 2
    plt.figure(figsize=(16,9), dpi=72)
    
    freq_X, intel_Y = dp.DPFFT(ds, dataset_test[maxIndex - dr * dt + time_steps + 1 : maxIndex + dr * dt].flatten())
    N = len(freq_X)
    #intel_Y = intel_Y.flatten()
    plt.plot(freq_X[:N//plotSize], 2/N * np.abs(intel_Y[:N//plotSize]), linestyle='-', color='red', label = 'Real Seismic Wave Frequency Spectrum')
    
    freq_Xs, intel_Ys = dp.DPFFT(ds, dataset_predicted[maxIndex - dr * dt + 1 : maxIndex + dr * dt - time_steps].flatten())
    N = len(freq_Xs)
    
    rmse = np.sqrt(mean_squared_error(2/N * np.abs(intel_Y[:N//plotSize]), 2/N * np.abs(intel_Ys[:N//plotSize])))
    
    
    #intel_Y = intel_Y.flatten()
    plt.plot(freq_Xs[:N//plotSize], 2/N * np.abs(intel_Ys[:N//plotSize]), linestyle='-', color='blue', alpha=0.5, label = 'Simulated Seismic Wave Frequency Spectrum')
    plt.title('Frequency Spectrum Zoom In',loc = 'center')
    plt.title('Train Score: %.2f RMSE' % (rmse), loc = 'left')
    plt.xlabel('Frequency (Hz)', loc = 'center')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()
    
    
    if(num == stopNum):
        break;

print("Average rmse : {}".format(rmseAvg/5))
print("Average loss : {}".format(lossAvg/5))
end_time = time.time()

execution_time = end_time - start_time
print("run time: {:.2f} seconds".format(execution_time))

