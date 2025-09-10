# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:11:27 2024

@author: jimmy
"""
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import find_peaks
#import tensorflow as tf
#import matplotlib.pyplot as plt  # for 畫圖用
from obspy.core import UTCDateTime
#from obspy import read

#return dataframe
def Data_Trim(input_trace, rangeTime):
    
    maxIndex = np.argmax(input_trace.data)
    
    endIndex = maxIndex + rangeTime * input_trace.stats.sampling_rate * 60
    startIndex = maxIndex - rangeTime * input_trace.stats.sampling_rate * 60

    startTime = UTCDateTime(input_trace.times("timestamp")[int(startIndex)])
    endTime = UTCDateTime(input_trace.times("timestamp")[int(endIndex)])
    
    trace_result = copy.deepcopy(input_trace)
    trace_result = trace_result.trim(starttime = startTime, endtime = endTime).copy()
    
    return pd.DataFrame({'Amplitude' : trace_result.data})

def Data_Trim_Right(input_trace, rangeTime):
        
    maxIndex = np.argmax(input_trace.data)
    startIndex = 0
    leftover = 0
    if(maxIndex - rangeTime * input_trace.stats.sampling_rate * 60 < 0):
        leftover = rangeTime * input_trace.stats.sampling_rate * 60 - maxIndex
    else:
        startIndex = maxIndex - rangeTime * input_trace.stats.sampling_rate * 60
        
    endIndex = maxIndex + rangeTime * input_trace.stats.sampling_rate * 60 + leftover

    startTime = UTCDateTime(input_trace.times("timestamp")[int(startIndex)])
    endTime = UTCDateTime(input_trace.times("timestamp")[int(endIndex)])
    
    trace_result = copy.deepcopy(input_trace)
    trace_result = trace_result.trim(starttime = startTime, endtime = endTime).copy()
    
    return pd.DataFrame({'Amplitude' : trace_result.data})

#return dataframe
def Data_TrimFill(input_trace, rangeTime, time_steps):
    
    maxIndex = np.argmax(input_trace.data)
    
    startIndex = maxIndex - rangeTime * input_trace.stats.sampling_rate * 60 - time_steps + 1
    endIndex = maxIndex - rangeTime * input_trace.stats.sampling_rate * 60

    startTime = UTCDateTime(input_trace.times("timestamp")[int(startIndex)])
    endTime = UTCDateTime(input_trace.times("timestamp")[int(endIndex)])
    
    trace_result = copy.deepcopy(input_trace)
    trace_result = trace_result.trim(starttime = startTime, endtime = endTime).copy()
    
    return pd.DataFrame({'Amplitude' : trace_result.data})

#return dataframe
def Data_Trim_Trace(input_trace, rangeTime):
    
    maxIndex = np.argmax(input_trace.data)
    
    endIndex = maxIndex + rangeTime * input_trace.stats.sampling_rate * 60
    startIndex = maxIndex - rangeTime * input_trace.stats.sampling_rate * 60

    startTime = UTCDateTime(input_trace.times("timestamp")[int(startIndex)])
    endTime = UTCDateTime(input_trace.times("timestamp")[int(endIndex)])
    
    trace_result = copy.deepcopy(input_trace)
    trace_result = trace_result.trim(starttime = startTime, endtime = endTime).copy()
    
    return trace_result

#trturn trace
def FindTrace(stream):
    
    nMax = 0
    index = 0
    for ind in range(0,3):
        trace = stream[ind]
        if (trace.stats.npts >= nMax):
            nMax = trace.stats.npts
            index = ind
    
    #return trace
    return stream[index]


def RawDataPreprocess(name):
    raw_Data = open(name, 'r')
    
    lineCount = 0
    npts = 0
    dt = 0
    
    dataset = np.zeros(1)
    
    for line in raw_Data:
        
        if(lineCount == 3):
            npts = int(line[5:12])
            dt = float(line[17:25])
        
        if(lineCount >= 4):
            lineData = line[:-1].split(' ')
            lineData = [i for i in lineData if i]
            #print(lineData)
            lineData = [float(i) for i in lineData[:] if i != ' ']
            dataset = np.append(dataset, lineData)
        
        lineCount += 1
        
    dataset = np.delete(dataset, 0)
    
    dataset = dataset.reshape(len(dataset), 1)
        
    raw_Data.close()
    
    return dt, dataset

def DPFFT(dt, dataset):
    N = len(dataset)
    freq_X = fftfreq(N,dt)
    intel_Y = fft(dataset)[1:]
    
    return freq_X, intel_Y

def DPFFTPlot(freq_X, intel_Y, plotSize = 2):
    
    N = len(freq_X)
    
    plt.figure(figsize=(16,9), dpi=72)
    plt.plot(freq_X[:N//plotSize], 2/N * np.abs(intel_Y[:N//plotSize]), linestyle='-', color='blue')
    
    intel_Y = intel_Y.flatten()
    
    #max intensity
    intel_Y_max = np.max(2 / N * np.abs(intel_Y[1:N//plotSize])) #2/N是因為FFT後的數據為對稱圖形，只用一邊
    peaksIndex, properties = find_peaks(2/N*np.abs(intel_Y[1:N//plotSize]), height = intel_Y_max)
    
    plt.plot(freq_X[peaksIndex], intel_Y_max, 'x', color = 'red')
    
    plt.show()
    
    
def FindChannel(path):
    folder = os.listdir(path)
    i = 0
    maxA = 0
    maxP = ''
    resultName = np.array([], dtype = str)
    
    for file in folder:

        file_path = path + file
        
        dt, dataset = RawDataPreprocess(file_path)
        
        if(np.max(dataset) >= maxA):
            maxA = np.max(np.abs(dataset))
            maxP = file
        
        if(i % 3 == 2):
            if(maxA < 1e-5):
                maxA = 0
                maxP = ''
                continue;
            else:
                print(maxA)
                maxA = 0
                resultName = np.append(resultName, maxP)
                maxP = ''
        
        i += 1
    
    return resultName


def FileNamesArray(path):
    
    folder = os.listdir(path)
    
    resultName = np.array([], dtype = str)
    
    for file in folder:
        resultName = np.append(resultName, file)
    
    return resultName
        

