import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime
from dataclasses import dataclass
import sys

# Paul Xing Winter 2023

@dataclass
class Classifier:


    def __init__(self,window, nclass=3,filter=0):

        self.window = window        #window size
        self.nclass=nclass        #number of class
        self.filter=filter
        
        #mean and std of raw data
        self.mean1 = None
        self.mean2 = None
        self.std1 = None
        self.std2 = None

        self.featuresMean = None
        self.featuresStd=None
        self.weight = None

        #memory
        self.MAVS01=0
        self.MAVS02=0
        self.P0=np.asarray([0,1,0])
        self.P1=np.asarray([0,1,0])

        self.features=None

        #self.featuresNames =  ['MAV1','MAV2','MAVS1','MAVS2','VAR1','VAR2'\
        #    ,'RMS1','RMS2','WL1','WL2','ZC1','ZC2', 'SSC1','SSC2','ENV1','ENV2'\
        #        ,'RMSf1','RMSf2','VARf1','VARf2']

        #self.featuresNames = ['MAV1', 'MAV2', 'WL1', 'WL2', 'ZC1', 'ZC2', 'SSC1','SSC2']
        #self.featuresNames = ['MAV1', 'MAV2', 'MAVS1', 'MAVS2','VAR1','VAR2','RMS1','RMS2','WL1', 'WL2', 'ZC1', 'ZC2', 'SSC1','SSC2']
        self.featuresNames = ['MAV1', 'MAV2','VAR1','VAR2','RMS1','RMS2','WL1', 'WL2', 'ZC1', 'ZC2', 'SSC1','SSC2']

        self.loss_step = []
        self.W_step = []




    def centeringData(self,ch1,ch2):

        #centering data (voltage offset in data) and standardization
        ch1 = (ch1 - self.mean1)/self.std1
        ch2 = (ch2 - self.mean2)/self.std2

       
        return ch1, ch2



    def calculateFeatures(self, ch1, ch2):


        if self.filter:

            #convolution kernel for lowpass filtering
            kernel = np.hanning(5)
            #kernel = np.ones(self.window)
            kernel = kernel/np.sum(kernel)
        
            #extracting enveloppe
            for i in range(ch1.shape[0]):
                ch1[i,:] = np.convolve((ch1[i,:]), kernel, mode='same')
                ch2[i,:] = np.convolve((ch2[i,:]), kernel, mode='same')


        #reshaping into number of windows
        #ch1=np.reshape(ch1,(-1,self.window))
        #ch2=np.reshape(ch2,(-1,self.window))

        # mean absolute value 
        MAV1 = np.mean(np.abs(ch1),axis=1)
        MAV2 = np.mean(np.abs(ch2),axis=1)



        # MAV slope (memory effect)
        MAVS1 = np.diff(MAV1, prepend=self.MAVS01) 
        MAVS2 = np.diff(MAV2, prepend=self.MAVS02)

        #variance
        VAR1 = np.var(ch1, axis=1)
        VAR2 = np.var(ch2, axis=1)
      
        #root mean square
        RMS1 = np.sqrt(np.mean(ch1**2, axis=1))
        RMS2 = np.sqrt(np.mean(ch2**2, axis=1))

      
        #waveform length
        WL1 = np.sum(np.abs(ch1[:,1:]-ch1[:,:-1]),axis=1)
        WL2 = np.sum(np.abs(ch2[:,1:]-ch2[:,:-1]),axis=1)

        #Zero crossing threshold is because of data standardization
        thrs=4
        ZC1=np.sum( ((ch1[:,:-1]> 0) & (ch1[:,1:]<0) ) | ((ch1[:,:-1]< 0) & (ch1[:,1:]>0)) & (np.abs(ch1[:,:-1]-ch1[:,1:])>thrs),axis=-1)
        ZC2=np.sum( ((ch2[:,:-1]> 0) & (ch2[:,1:]<0) ) | ((ch2[:,:-1]< 0) & (ch2[:,1:]>0)) & (np.abs(ch2[:,:-1]-ch2[:,1:])>thrs),axis=-1)

        
        #signed change
        SSC1 = np.sum(((ch1[:,1:-1]>ch1[:,:-2]) & (ch1[:,1:-1]>ch1[:,2:])) \
            | ((ch1[:,1:-1]<ch1[:,:-2]) & (ch1[:,1:-1]<ch1[:,2:]))  \
                & ( ( np.abs(ch1[:,1:-1]-ch1[:,2:])>2)| (np.abs(ch1[:,1:-1]-ch1[:,:-2])>thrs)) ,axis=-1)

        SSC2 = np.sum(((ch2[:,1:-1]>ch2[:,:-2]) & (ch2[:,1:-1]>ch2[:,2:])) \
            | ((ch2[:,1:-1]<ch2[:,:-2]) & (ch2[:,1:-1]<ch2[:,2:]))  \
                & ( ( np.abs(ch2[:,1:-1]-ch2[:,2:])>2)| (np.abs(ch2[:,1:-1]-ch2[:,:-2])>thrs)) ,axis=-1)
        
   


        #features = np.stack((MAV1,MAV2,MAVS1,MAVS2,VAR1,VAR2,RMS1,RMS2,WL1,WL2,ZC1,ZC2, SSC1,SSC2,ENV1,ENV2,RMSf1,RMSf2,VARf1,VARf2),axis=-1)


        #features = np.stack((MAV1,MAV2,WL1,WL2,ZC1,ZC2, SSC1,SSC2),axis=-1)
        #features = np.stack((MAV1,MAV2,MAVS1,MAVS2,VAR1,VAR2,RMS1,RMS2,WL1,WL2,ZC1,ZC2, SSC1,SSC2),axis=-1)
        features = np.stack((MAV1,MAV2,VAR1,VAR2,RMS1,RMS2,WL1,WL2,ZC1,ZC2, SSC1,SSC2),axis=-1)

        
        return features


    def featuresNormalization(self,features):
        features = (features-self.featuresMean)/self.featuresStd[None,:]
        #features = (features)/self.featuresStd[None,:]

        return features



    #multimodal logistic regression
    #adapted from https://sophiamyang.github.io/DS/optimization/multiclass-logistic/multiclass-logistic.html
    # gradient descent and prediction were modified to include bias in the model and label shift of the data

    #*********************************************
    def softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=1).reshape(-1,1)

   

    def loss(self,X,W, Y):

     
        Z = - np.matmul(X, W)
 
        N = Z.shape[0]
      
        loss = 1/N * (np.trace(np.matmul(np.matmul(X,W), Y.T)) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))

        return loss



    def gradient(self,X, Y, W, mu):
 
        Z = - np.matmul(X, W)
        P = self.softmax(Z)
        N = X.shape[0]
        gd = 1/N * np.matmul(X.T, (Y - P)) + 2 * mu * W

        return gd


    def gradient_descent(self,X, y, max_iter=1000, eta=0.5, mu=0.01):

        X = np.concatenate((X,np.ones((X.shape[0],1))),axis=1) # include bias

        y=y+1

        #label binarizing
        Y=np.zeros((len(y),self.nclass))
        for i in range(self.nclass):
            Y[np.where(y==i),i]=1
        
        W = np.zeros((X.shape[1], self.nclass))
        #W = np.random.randn(X.shape[1], self.nclass)

        step = 0
 
        while step < max_iter:
            step += 1
            W -= eta * self.gradient(X, Y, W, mu)
            self.W_step.append(W.copy())
            self.loss_step.append(self.loss(X, W,Y))


        return  W


    def predict(self, H):

        H = np.concatenate((H,np.ones((H.shape[0],1))),axis=1)

        Z = - np.matmul(H, self.weight)
        P = self.softmax(Z)

        #adding memory
        P1=1/6*self.P0+1/6*self.P1+2/3*P
        self.P0=P
        self.P1=P1

        return np.argmax(P1, axis=1)-1

    #*********************************************






    def train(self, ch1, ch2,label, max_iter=1000, eta=0.5, mu=0.01):

        #calculate mean of raw data
        self.mean1 = np.mean(ch1) 
        self.mean2 = np.mean(ch2)
        self.std1 =np.std(ch1)
        self.std2 = np.std(ch2)


        ch1,ch2= self.centeringData(ch1,ch2)

       

        self.MAVS01 = np.mean(np.abs(ch1))  
        self.MAVS02 = np.mean(np.abs(ch2))  


        Features= self.calculateFeatures(ch1, ch2)
        self.featuresMean = np.mean(Features, axis=0)
        self.featuresStd = np.std(Features, axis=0)

        Features = self.featuresNormalization(Features)

        

        self.features=Features

        weight=self.gradient_descent(Features,label,max_iter, eta, mu)

        self.weight=weight

        #return df




    def test(self, ch1, ch2):
       

        ch1,ch2= self.centeringData(ch1,ch2)

        Features = self.calculateFeatures(ch1,ch2)


        self.MAVS01=Features[0,0].item()
        self.MAVS01=Features[0,1].item()
        

        Features = self.featuresNormalization(Features)
        
        label_i = self.predict(Features).item() 
    
        
        return label_i



def create_new_sampling_file(path, name):
    # complete file name with unique ID
    file_id = datetime.now()
    file_name = (path + name + "_" + file_id.strftime("%y-%m-%d_%H-%M-%S") + ".csv")

    # opening file in writing mode
    file = open(file_name, 'w', newline="")

    # header fields
    fields = ['voltage1 (V)', 'voltage2 (V)']
    csv.DictWriter(file, fieldnames=fields).writeheader()

    # closing file
    file.close()

    return file_name


def acquire_window(ch1, ch2, window_size, writer):
    # EMG data for current window
    data1 = []
    data2 = []

    # acquisition loop
    for i in range(window_size):
        data1.append(ch1.voltage)
        data2.append(ch2.voltage)
        writer.writerow([data1[i], data2[i]])

    return data1, data2


def acquire_training_dataset(ch1, ch2, window_size, num_window, file_name, training_label ):




    # file opening in append mode
    file = open(file_name, 'a', newline="")
    training_writer = csv.writer(file)

    start = time.time()
    # acquisition loop
    print("debut acquisition")
    for window in range(num_window):
        print(training_label [window])
        acquire_window(ch1, ch2, window_size, training_writer)
    file.close()

    end=time.time()

    # frequency sampling calculation
    fs = num_window/(end-start)
    print("frequence acquisition : " + str(round(fs, 2)) + " sps")


def visualize_sampling(file,window,training_label ):
    ...

    data = pd.read_csv(file)

    chan1=np.asarray(data['voltage1 (V)'])
    chan2=np.asarray(data['voltage2 (V)'])

    
    plt.figure
    plt.plot(chan1)

    plt.plot(chan2)
    plt.xlabel('samples')
    plt.ylabel('Voltage')
    plt.legend(['Chanel 1','Chanel 2'])

    plt.show()

    chan1=np.abs(chan1-np.mean(chan1))
    chan2=np.abs(chan2-np.mean(chan2))
    chan1=chan1/np.max(chan1)
    chan2=chan2/np.max(chan2)
    chan1=np.mean(np.reshape(chan1,(-1,window)),axis=1)
    chan2=np.mean(np.reshape(chan2,(-1,window)),axis=1)

    

    plt.figure()
    plt.plot(chan1)
    plt.plot(chan2)
    plt.plot(training_label )
    plt.xlabel('samples')
    plt.ylabel('Signal')
    plt.legend(['Chanel 1','Chanel 2'])
    plt.show()

    try:
        input('Press "Enter" to continue or "ctrl + c" to stop code')
    except KeyboardInterrupt:
        print("stop code")
        sys.exit(0)



def train_classifier(file_name, window, training_label,max_iter=1000, eta=0.5, mu=0.01, filter=0 ):
    print("Training in progress")
    start = time.time()


    data = pd.read_csv(file_name)

    chan1=np.asarray(data['voltage1 (V)'])
    chan2=np.asarray(data['voltage2 (V)'])
    
    #reshaping into number of windows
    chan1=np.reshape(chan1,(-1,window))
    chan2=np.reshape(chan2,(-1,window))
 

    #create instance of class
    classifier = Classifier(window,3,filter)
    classifier.train(chan1,chan2,training_label,max_iter, eta, mu)
    
    file_id = datetime.now()
    file_name = ('learning' + "_" + file_id.strftime("%y-%m-%d_%H-%M-%S") + ".csv")

    #df.to_csv(file_name)
    
    end=time.time()

    
    duration = (end-start)
    print("training time : " + str(duration) + " seconds")


    return classifier


def test_classifier(classif, ch1, ch2, window_size, num_window, buz, file_name,max_iter=1000, eta=0.5, mu=0.1):


    # iniate list of label
    label = []
    # file opening in append mode
    file = open(file_name, 'a', newline="")
    test_writer = csv.writer(file)

    # acquisition loop
    print("debut acquisition")
    for window in range(num_window):
        data1, data2=acquire_window(ch1, ch2, window_size, test_writer)

        data1=np.asarray(data1).reshape((1,-1))
        data2=np.asarray(data2).reshape((1,-1))

        #called test method of the trained classifier
        label_window = classif.test(data1,data2)

        label.append(label_window)

        #call buzzer if true
        if label_window==1:
            buz.ChangeFrequency(10)
            buz.start(10)
        elif label_window ==-1:
            buz.ChangeFrequency(100)
            buz.start(10)
        else:
            buz.stop()
        print(label_window)

    file.close()

    print(label)
    buz.stop()



    ...
    return label
