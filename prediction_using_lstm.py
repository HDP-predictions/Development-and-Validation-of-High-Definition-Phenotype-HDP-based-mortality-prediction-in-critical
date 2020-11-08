#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import sys
import linecache
import pandas as pd
import numpy as np
import psycopg2
import math
import itertools
import tensorflow as tf
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Activation, Dense, Dropout, SpatialDropout1D,Input,Masking,Bidirectional, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
from keras.models import Sequential, Model
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from random import seed
from sklearn.metrics import roc_auc_score
from keras.layers import Input, Dense
from keras.models import Model
from prettytable import PrettyTable
from random import shuffle
from pylab import rcParams
from data_visualization import *
import random

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

#70:30 split
def split_70(x):
    return int((round((x/15)*0.7))*15)

#Function for calculating upper and lower CI
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m,3), round(m-h,3), round(m+h,3)


#Function for converting data to multiples of 15
def range_finder(x):
    length = x
    fractional = (x/15.0) - math.floor(x/15.0)
    return int(round(fractional*15))

def make_lstm_visualize(gd,factor,trainingSet,testingSet):
    try:
        print('--------inside make_lstm')

        testingSet.fillna(-999,inplace=True)
        trainingSet.fillna(-999,inplace=True)

        if(factor != "fixed" and factor != "inter" and factor != "fixed_inter"):
            if 'heartrate' in trainingSet.columns:
                print('pass')
                trainingSet = trainingSet[trainingSet['heartrate']!=-999]
                if 'se_heartrate' in trainingSet.columns:
                    trainingSet = trainingSet[trainingSet['se_heartrate'] < 999]
            if 'heartrate' in testingSet.columns:
                print('pass')
                testingSet = testingSet[testingSet['heartrate']!=-999]
                if 'se_heartrate' in testingSet.columns:
                    testingSet = testingSet[testingSet['se_heartrate'] < 999]
        
        train = pd.DataFrame(columns=gd.columns)
        ids = trainingSet.uhid.unique()
        shuffle(ids)
        for i in ids:
            x = trainingSet[trainingSet['uhid']==i]
            x = x[range_finder(len(x)):len(x)]
            train = train.append(x,ignore_index=True)
    

        test = pd.DataFrame(columns=gd.columns)
        ids = testingSet.uhid.unique()
        shuffle(ids)
        #Training
        for i in ids:
            x = testingSet[testingSet['uhid']==i]
            x = x[range_finder(len(x)):len(x)]
            test = test.append(x,ignore_index=True)

        

        y_train = train['dischargestatus']
        X_train = train.drop('dischargestatus',axis=1)
        X_train = X_train.drop('uhid',axis=1)
        #X_train = X_train.drop('visittime',axis=1)

        y_test = test['dischargestatus']

        #print('train uhid length=',len(train.uhid.unique()), ' UHID =', train.uhid.unique())
        #print('test uhid length=',len(test.uhid.unique()), ' UHID =', test.uhid.unique())

        #train.to_csv('train_hdp.csv')
        #test.to_csv('test_hdp.csv')
        #how many rows of machine data is present for death and discharge cases in test and train
        deathTrain = train[train['dischargestatus']==1]
        dischargeTrain = train[train['dischargestatus']==0]
        #print('Train death case  total length =', len(deathTrain))
        #print('Train discharge case  total length =', len(dischargeTrain))
    
        #and test['spo2']!=-999

        deathTest = test[test['dischargestatus']==1]
        dischargeTest = test[test['dischargestatus']==0]
        #print('Test death case  total length =', len(deathTest))
        #print('Test discharge case  total length =', len(dischargeTest))

        #print('Test death case  machine correct length =', len(deathTest))
        #print('Test discharge case  machine correct length =', len(dischargeTest))

        X_test = test.drop('dischargestatus',axis=1)
        X_test = X_test.drop('uhid',axis=1)
        auc_roc_inter = []
        val_a = []
        train_a = []
        #converting the data into a numpy array
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        ytrain1 = []
        for i in range(0,len(y_train),15):
            #print(i)
            y1 = y_train[i:i+15]
            ytrain1.append(y1[-1])

        ytest1 = []
        for i in range(0,len(y_test),15):
            #print(i)
            y1 = y_test[i:i+15]
            ytest1.append(y1[-1])
        ytrain1 = np.array(ytrain1)
        ytest1 = np.array(ytest1)
        Xtrain = np.reshape(X_train, (-1, 15, X_train.shape[1]))
        Xtest = np.reshape(X_test, (-1, 15, X_test.shape[1]))
        return Xtrain,Xtest,ytrain1,ytest1,test,train
    except Exception as e:
            print('Error in make_lstm_visualize method',e)
            PrintException()
            return None


def visualizeLSTMOutput(xTestWithUHID,hdpPlotdict):
    try:
        currentFigure = None
        path = os.getcwd()       
        for i in xTestWithUHID.uhid.unique():            
            #print('Inside visualizeLSTMOutput = ',i)
            hdpAX = hdpPlotdict.get(i)
            #print('hdpAX = ',hdpAX)
            x = xTestWithUHID[xTestWithUHID['uhid']==i]
            deathOrDischargeCase = 'Death'
            #print(x.columns)
            #print('case is',x.dischargestatus.head(1))      
            if (x.iloc[0].dischargestatus == 1):
                deathOrDischargeCase = 'Death_Cases'
            elif (x.iloc[0].dischargestatus == 0):
                deathOrDischargeCase = 'Discharge_Cases'
            rcParams['figure.figsize'] = 20, 6
            if not(hdpAX is None) :
                currentFigure = hdpAX.get_figure()
                axes = currentFigure.gca()
                sns.set(font_scale = 2)
            if ('y_pred' in x.columns) and (not(hdpAX is None)):
                y_pred = np.array(x['y_pred']).flatten()
                y_df = y_pred
                #print('currentFigure=',currentFigure)
                #sns.scatterplot(y = y_df[0], x = np.arange(len(y_pred)),linewidth=0, legend='full')
                #print('case is',deathOrDischargeCase,' y_pred = ',y_pred)
                #hdpAX.plot(np.arange(len(y_pred)),y_pred, label=deathOrDischargeCase+i)
                hdpAX.plot(np.arange(len(y_pred)),y_pred)
                hdpAX.legend(loc="upper right") 
                hdpAX.legend()
                #plt.title(uhid)
                axes.set_xlabel('LOS in Time Steps of 15 Minutes')
                axes.set_ylabel('HDP Mortality Probability')
                axes.set_ylim([0,1])       
                axes.set_yticks((0.20,0.40,0.50,0.60,0.80))
            if not(hdpAX is None) :
                currentFigure.savefig(path+'/'+deathOrDischargeCase+'/'+str(i)+'.png',dpi = 300)
        return True
    except Exception as e:
        print ('Exception',e)
        PrintException()
        return None

#LSTM model
def lstm_model(n,gd,hdpPlotdict,factor,trainingSet,testingSet):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,restore_best_weights=True,patience=4)
    auc_roc_inter = []
    val_a = []
    train_a = []
    trainLSTM = pd.DataFrame()
    valLSTM = pd.DataFrame()
    print('-----------Inside lstm model-------------',n,len(gd))
    plt.figure()
    for iterationCounter in range(5):
        try:
            #print('-----------gd.uhid-------------=',gd.uhid.unique())
            #print('-----------training.uhid-------------=',len(trainingSet))
            #print('-----------testing.uhid-------------=',len(testingSet))
            n_dropout = [0.2]
            for dropout in n_dropout:
                print('-----------Iteration No-------------=',iterationCounter, '----DROPOUT=',dropout)
                Xtrain,Xtest,ytrain1,ytest1,xTestWithUHID,xTrainWithUHID = make_lstm_visualize(gd,factor,trainingSet,testingSet)
                #xTestWithUHID.to_csv('Pre_LSTMDataOutput_'+str(iterationCounter)+'.csv')

                #Building the LSTM model

                """
                X = Input(shape=(None, n), name='X')
                mX = Masking()(X)
                # X-> BidirectionalLSTM -> MASKING LAYER -> LSTM -> Dense -> y 
                lstm = Bidirectional(LSTM(units=512,activation='tanh',return_sequences=True,recurrent_dropout=0.5,dropout=0.3))
                #for timeseries data of varying length - in case one patient has 15 pulse rate values vs the second patient has 10
                # masking will handle this mismatch of data
                mX = lstm(mX)
                
                L = LSTM(units=128,activation='tanh',return_sequences=False)(mX)
                y = Dense(1, activation="sigmoid")(L)
                outputs = [y]
                inputs = [X]
                model = Model(inputs,outputs)

                """
                #X = Input(shape=(None, n), name='X')
                model = Sequential()
                inputNumberNeuron = 512
                multiplEachLayer = (1)
                n_steps = 15
                hiddenLayerNeuron1 = int(multiplEachLayer*inputNumberNeuron)
                #hiddenLayerNeuron2 = int(multiplEachLayer*hiddenLayerNeuron1)
                #model.add(LSTM(inputNumberNeuron, activation='tanh', return_sequences=True, input_shape=(15, n)))
                model.add(Bidirectional(LSTM(inputNumberNeuron, activation='tanh',return_sequences=True,dropout=dropout), input_shape=(n_steps, n)))
                #model.add(Dropout(0.4))
                #model.add(LSTM(hiddenLayerNeuron1,activation='tanh',return_sequences=True))
                model.add(LSTM(256,activation='tanh',dropout=dropout))
                model.add(Dense(1, activation="sigmoid"))
                model.summary()
                model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
                #model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
                #validation accuracy
                v_a = []
                #training accuracy
                t_a = []
                #fitting the model
                #history = model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
                history = model.fit(Xtrain, ytrain1, batch_size=15 ,validation_split=0.15,epochs=8)

                trainLSTM[str(iterationCounter)] = history.history['loss']
                valLSTM[str(iterationCounter)] = history.history['val_loss']

                #history = model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
                for i in range(len(model.history.history['val_accuracy'])):
                    v_a.append(model.history.history['val_accuracy'][i])
                    t_a.append(model.history.history['accuracy'][i])
                #predictions
                y_pred = model.predict(Xtest)
                #y_pred = y_pred.round()
                y_test = np.array(ytest1)
                y_pred = np.array(y_pred)
                y_test = pd.DataFrame(y_test)
                y_test = np.array(y_test)
                #if more than 0.5 then mark as dead, otherwise mark as Discharge
                def acc(x):
                    if x>0.5:
                        return 1
                    else:
                        return 0
                #y_model is the outcome of test
                y_model=[]
                for i in y_pred:
                    y_model.append(acc(i))
                #y_answer is the actual observation (discharge status) present in the dataset
                y_answer=[]
                for j in y_test:
                    y_answer.append(acc(j))
                #print('y_model',y_model,'y_answer',y_answer)
                print('------------visualization of output Started------------------')
                #since in previous operation data of 15 mins was processed as one block, so reconverting the results 
                #to repeat of 15 mins
                yNew = np.repeat(y_pred, 15)
                xTestWithUHID['y_pred'] = yNew
                visualizeLSTMOutput(xTestWithUHID,hdpPlotdict)
                visualizeLSTMOutput(xTrainWithUHID,hdpPlotdict)
                xTestWithUHID.to_csv('LSTMDataOutput_'+str(iterationCounter)+'.csv')
                print('------------visualization of output Done------------------')
                #append validation and training accuracy from each iteration
                val_a.append(v_a)
                train_a.append(t_a)
                #print(t_a)
                #print(train_a)
                #print('--------between t_a & train_a -------------y_answer=',y_answer,' y_pred=',y_pred)
                #So for all epochs (38 in our cases) the y_pred will be generated and compared with actual observed discharge status to generate 
                #roc auc curve
                roc_aoc_s = roc_auc_score(y_answer,y_pred)
                print('roc_auc_score',roc_aoc_s)
                auc_roc_inter.append(roc_aoc_s)
                #HS: For dropout increment iterationCounter
                #iterationCounter =  iterationCounter + 1
        except Exception as e:
            print('Exception inside lstm_model',iterationCounter,e)
            PrintException()
            continue
    print('-----------Done with lstm model-------------')
    plt.figure()
    color_dict = {'0': 'blue', '1': 'red', '2': 'green', '3': 'black', '4': 'orange'}
    #plt.plot(trainLSTM, color='blue', label='train')
    trainLSTM.plot(color=[color_dict.get(x, '#333333') for x in trainLSTM.columns])
    valLSTM.plot(color=[color_dict.get(x, '#333333') for x in valLSTM.columns])
    #plt.plot(valLSTM, color='orange', label='validation')
    plt.title('Different iterations train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    #so for each iteration we have also captured the validation and training accuracy, this is returned along with auc, roc values
    return auc_roc_inter,list(itertools.chain(*val_a)),list(itertools.chain(*train_a))

def convert_date(x):
    print(x)
    x = str(x)
    return pd.to_datetime(x)

def predictLSTM(gw, fixed, cont, inter,hdpPlotdict,trainingSet,testingSet):
    try:
        #print('Inside predictLSTM column count=',gw.columns)
        #defining the early stopping criteria
        f_a = []
        i_a = []
        c_a = []
        ci_a = []
        cf_a = []
        fi_a = []
        a = []

        f_training = []
        i_training = []
        c_training = []
        ci_training = []
        cf_training = []
        fi_training = []
        a_training = []
        #reduced 2 for uhid and dischargestatus, 1 extra for hour_series - temporary testing
        lengthOfFixed = len(fixed) - 2
        #reduced 2 for uhid and dischargestatus
        lengthOfIntermittent = len(inter) - 2
        #reduced 2 for uhid and dischargestatus
        lengthOfContinuous = len(cont) - 2
        
        #---------------FIXED------------------
        gd = gw[fixed]
        trainingSetgd = trainingSet[fixed]
        testingSetgd = testingSet[fixed]
        print('total length of gd=',len(gd),'gd count',gd.count())
        an = lstm_model(lengthOfFixed,gd,hdpPlotdict,'fixed',trainingSetgd,testingSetgd)
        f_a.append(an[0])
        f_training.append(an[1])
        print('-----AN-------',an)
        print('---------fixed----------',f_a)
        print(mean_confidence_interval(an[0]))
        
        #---------------INTER------------------
        gd = gw[inter]
        trainingSetgd = trainingSet[inter]
        testingSetgd = testingSet[inter]
        an = lstm_model(lengthOfIntermittent,gd,lstm_model,'inter',trainingSetgd,testingSetgd)
        i_a.append(an[0])
        i_training.append(an[1])
        print('inter',i_a)
        print(mean_confidence_interval(an[0]))
        
        #---------------CONT------------------
        gd = gw[cont]
        trainingSetgd = trainingSet[cont]
        testingSetgd = testingSet[cont]
        #this will remove all indexes where continuous data is not present
        gd = gd[gd["heartrate"] != -999]
        #print('---------------AFTER CHECK of SPO2 =-9999--------------')
        an = lstm_model(lengthOfContinuous,gd,hdpPlotdict,'cont',trainingSetgd,testingSetgd)
        c_a.append(an[0])
        c_training.append(an[1])
        print('----------c_a----------->',c_a)
        visualizeDataFrameDataset(gd,'cont')    
        print(mean_confidence_interval(an[0]))
        
        #---------------CONT+INTER------------------
        cont_inter = list(set(cont+inter))
        gd = gw[cont_inter]
        trainingSetgd = trainingSet[cont_inter]
        testingSetgd = testingSet[cont_inter]
        #this will remove all indexes where continuous data is not present
        gd = gd[gd["heartrate"] != -999]       
        an = lstm_model(lengthOfIntermittent+lengthOfContinuous,gd,hdpPlotdict,'cont_inter',trainingSetgd,testingSetgd)
        ci_a.append(an[0])
        print('cont_inter',ci_a)
        ci_training.append(an[1])
        print(mean_confidence_interval(an[0]))
        #---------------FIXED+INTER------------------
        fixed_inter = list(set(fixed+inter))
        gd = gw[fixed_inter]
        trainingSetgd = trainingSet[fixed_inter]
        testingSetgd = testingSet[fixed_inter]
        an = lstm_model(lengthOfFixed+lengthOfIntermittent,gd,hdpPlotdict,'fixed_inter',trainingSetgd,testingSetgd)
        fi_a.append(an[0])
        fi_training.append(an[1])
        print('fixed_inter',fi_a)
        print(mean_confidence_interval(an[0]))
        #---------------CONT+FIXED------------------
        cont_fixed = list(set(cont+fixed))
        gd = gw[cont_fixed]
        trainingSetgd = trainingSet[cont_fixed]
        testingSetgd = testingSet[cont_fixed]
        #this will remove all indexes where continuous data is not present
        gd = gd[gd["heartrate"] != -999]        
        an = lstm_model(lengthOfFixed+lengthOfContinuous,gd,hdpPlotdict,'cont_fixed',trainingSetgd,testingSetgd)
        cf_a.append(an[0])
        cf_training.append(an[1])
        print('cont_fixed',cf_a)
        print(mean_confidence_interval(an[0]))
        #---------------CONT+FIXED+INTER------------------
        all_cols = list(set(cont+inter+fixed))
        gd = gw[all_cols]
        trainingSetgd = trainingSet[all_cols]
        testingSetgd = testingSet[all_cols]
        #this will remove all indexes where continuous data is not present
        gd = gd[gd["heartrate"] != -999]        
        an = lstm_model(lengthOfFixed+lengthOfIntermittent+lengthOfContinuous,gd,hdpPlotdict,'all',trainingSetgd,testingSetgd)
        a.append(an[0])
        a_training.append(an[1])

        print('-------Testing Results------------')
   
        print('Fixed')
        print(mean_confidence_interval(list(itertools.chain(*f_a))))
        print('Inter')
        print(mean_confidence_interval(list(itertools.chain(*i_a))))
        print('Cont')
        print(mean_confidence_interval(list(itertools.chain(*c_a))))
        print('Cont+inter')
        print(mean_confidence_interval(list(itertools.chain(*ci_a))))
        print('Fixed+Inter')
        print(mean_confidence_interval(list(itertools.chain(*fi_a))))
        print('Cont+Fixed')
        print(mean_confidence_interval(list(itertools.chain(*cf_a))))
        print('All')
        print(mean_confidence_interval(list(itertools.chain(*a))))


        print('-------Training Results------------')

        
        print('Fixed')
        print(mean_confidence_interval(list(itertools.chain(*f_training))))
        print('Inter')
        print(mean_confidence_interval(list(itertools.chain(*i_training))))
        print('Cont')
        print(mean_confidence_interval(list(itertools.chain(*c_training))))
        print('Cont+inter')
        print(mean_confidence_interval(list(itertools.chain(*ci_training))))
        print('Fixed+Inter')
        print(mean_confidence_interval(list(itertools.chain(*fi_training))))
        print('Cont+Fixed')
        print(mean_confidence_interval(list(itertools.chain(*cf_training))))
        print('All')
        print(mean_confidence_interval(list(itertools.chain(*a_training))))
        
        return True
    except Exception as e:
        print('Exception in Prediction', e)
        PrintException()
        return None    