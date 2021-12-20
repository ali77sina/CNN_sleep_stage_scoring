# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:32:19 2021

@author: lina3953
"""

from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, lfilter, sosfilt,iirfilter, sosfiltfilt, filtfilt
import numpy as np
import matplotlib.pyplot as plt

#definision of filters

def notch_filter(data,f0,Q,fs):
    b, a = signal.iirnotch(f0, Q, fs)
    y = signal.lfilter(b,a,data)
    return y
def butter_bandpass_sos(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band',output ='sos')
    return sos

def butter_lowpass_sos(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = butter(order, high, btype='low',output ='sos')
    return sos

def butter_highpass_sos(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = butter(order, high, btype='high',output ='sos')
    return sos

def butter_bandpass_sos(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band',output ='sos')
    return sos

def butter_bandpass_filter_sos(data, lowcut, highcut, fs, order=6):
    sos = butter_bandpass_sos(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def butter_lowpass_filter_sos(data, highcut, fs, order=2):
    sos = butter_lowpass_sos( highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def butter_highpass_filter_sos(data, highcut, fs, order=2):
    sos = butter_highpass_sos( highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

import scipy
#fundtion for moving average
def running_mean(x, N = 20):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1
    return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def bad_label(x, window_length = 150, threshold = 25):
    mean = np.mean(x)
    std = np.std(x)
    high_lim = mean + 2.5*std
    low_lim = mean - 2.5*std
    label = np.zeros(len(x))
    for num,i in enumerate(x):
        s = 0
        for j in x[num*window_length:(num+1)*window_length]:
            if j > high_lim or j < low_lim:
                s += 1
        if s>threshold:
            label[num*window_length:(num+1)*window_length] = 1
    return label
                  

def include_nan(x):
    for i in np.isnan(x):
        if i:
            return True
    return False

def roc_calc(predictions, y, threshold = 0.8):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    preds = np.zeros(len(predictions))
    for i in range(len(preds)):
        if predictions[i] > threshold:
            preds[i] = 1
        else:
            preds[i] = 0
    for i in range(len(preds)):
        if preds[i] == y[i]:
            if preds[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if preds[i] == 0:
                fn += 1
            else:
                fp += 1
    return tp,fp,tn,fn

def plot_roc(preds, y):
    thrs = np.linspace(0,1,500)
    tpr = []    #tp/(fn+tp)
    fpr = []    #fp/(tn+fp)
    for i in thrs:
        tp,fp,tn,fn = roc_calc(preds, y, threshold = i)
        tpr.append(tp/(fn+tp))
        fpr.append(fp/(tn+fp))
    x_ran = np.linspace(0,1,50)
    y_ran = np.linspace(0,1,50)
    plt.scatter(fpr,tpr, s = 1)
    plt.plot(x_ran, y_ran)
    plt.show()