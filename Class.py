# -*- coding: utf-8 -*-
import numpy as np
import mne
from scipy.signal import hilbert, chirp
from Filters import notch_filter, butter_bandpass_filter, bad_label, include_nan, roc_calc
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Sleep:
    def __init__(self, raw_data_psg):
        self.ind_to_plot = []
        self.file_psg = ''
        self.hypnogram_file = ''
        self.delta_low = 1
        self.delta_high = 4
        self.theta_low = 4
        self.theta_high = 7
        self.alpha_low = 8
        self.alpha_high = 13
        self.beta_low = 15
        self.beta_high = 30
        self.gamma_low = 30
        self.gamma_high = 80
        self.epoch_length = 30*256
        self.fs = 256.0
        self.raw_data_psg = raw_data_psg
        self.Class = 0
        self.low_cut = 0
        self.high_cut = 10
        self.ind = 0
        
        

    

    #loading the hypnogram text file,
    #adjust this funtion to match your file
    def load_hypnogram(self):
        with open(self.hypnogram_file) as f:
            label_str = f.readlines()
        label_temp = np.array(label_str)[7:]
        label = []
        for i in label_temp:
            label.append(i.split(';')[-1].replace('\n','').replace(' ',''))
        #returning the labels as integers
        int_label = []
        for i in range(len(label)):
            if label[i] == 'N3':
                int_label.append(0)
            elif label[i] == 'N2':
                int_label.append(1)
            elif label[i] == 'N1':
                int_label.append(2)
            elif label[i] == 'REM':
                int_label.append(3)
            elif label[i] == 'Wake':
                int_label.append(4)
            else:
                int_label.append(5)
        int_label = np.array(int_label)
        #time for each hypnogram point
        time_hypno = np.linspace(0,len(self.raw_data_psg[0])/256,len(label))
        return int_label, time_hypno
    
    def amp_envelope(self,array):
        return np.abs(hilbert(array))
    
    def preprocessing_and_ae(self):
        notch_f = 50
        print(self.raw_data_psg.shape)
        new_sig = notch_filter(self.raw_data_psg[self.ind], notch_f, 5, fs=self.fs)
        new_sig = butter_bandpass_filter(new_sig, self.low_cut, self.high_cut, fs=self.fs, order=5)
        new_sig = self.amp_envelope(new_sig)
        label_0 = bad_label(new_sig)
        test_0 = np.zeros(len(label_0))
        for i in range(len(label_0)):
            if label_0[i] == 1:
                test_0[i] = np.nan
            else:
                test_0[i] = new_sig[i]
        return test_0
    
    def epoching(self):
        test_0 = self.preprocessing_and_ae()
        epochs_psg = []
        for i in range(int(len(self.raw_data_psg[0])/self.epoch_length)):
            epochs_psg.append(test_0[(i)*self.epoch_length:(i+1)*self.epoch_length])
        return np.array(epochs_psg)
    
    def labeled_dataset(self,epochs):
        int_label,_ = self.load_hypnogram()
        where_nan = []
        for i in epochs:
            where_nan.append(include_nan(i))
        x = []
        y = []
        for num,i in enumerate(epochs):
            if not where_nan[num]:
                if int_label[num] != 5: #dont append if artefact
                    x.append(i)
                    y.append(int_label[num])
        x = np.array(x)
        y = np.array(y)
        return x,y
    
    def creat_model(self,x):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(x.shape[-1],1)))
        model.add(tf.keras.layers.Conv1D(1, 50, use_bias = False))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        return model 

        

    
    def train(self, epoch_no = 30, batch_size = 10):
        
        epochs = self.epoching()
        x,y = self.labeled_dataset(epochs)
        model = self.creat_model(x)
        for num,i in enumerate(y):
            if i == self.Class:
                y[num] = 1
            else:
                y[num] = 0
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle = True)
        model.fit(x_train,y_train,batch_size = batch_size,epochs = epoch_no, validation_data = (x_test, y_test))
        return model, x_test, y_test

        
    def plot_roc(self,model, x_test, y_test):
        preds = model.predict(x_test)
        thrs = np.linspace(0,1,500)
        tpr = []    #tp/(fn+tp)
        fpr = []    #fp/(tn+fp)
        for i in thrs:
            tp,fp,tn,fn = roc_calc(preds, y_test, threshold = i)
            tpr.append(tp/(fn+tp))
            fpr.append(fp/(tn+fp))
        x_ran = np.linspace(0,1,50)
        y_ran = np.linspace(0,1,50)
        plt.scatter(fpr,tpr, s = 1)
        plt.plot(x_ran, y_ran)
        plt.title('For class {} frequency {} to {}'.format(self.Class,self.low_cut,self.high_cut))
        plt.legend()
        
       
        
edf_file_loc = ''   #the edf file location containing the recorded time-series signal
raw = mne.io.read_raw_edf(edf_file_loc)
raw_data_psg = raw.get_data()
raw_data_psg = np.array(raw_data_psg)
    
sleep = Sleep(raw_data_psg)
sleep.fs = 256
sleep.low_cut = 1
sleep.high_cut = 4
sleep.hypnogram_file = ''   #expert labeled file location
history, x_test, y_test = sleep.train()
sleep.plot_roc(history, x_test, y_test)    

