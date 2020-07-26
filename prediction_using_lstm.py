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

def make_lstm_visualize(gd):
    try:
        print('--------inside make_lstm')
        final_df = pd.DataFrame(columns=gd.columns)
        ids = gd.uhid.unique()
        #print('------inside make lstm---unique uhid count =',len(ids))
        shuffle(ids)
        for i in ids:
            x = gd[gd['uhid']==i]
            x = x[range_finder(len(x)):len(x)]
            final_df = final_df.append(x,ignore_index=True)
        final_df.fillna(-999,inplace=True)
        train = final_df[:split_70(len(final_df))]
        test = final_df[split_70(len(final_df)):]

        y_train = train['dischargestatus']
        X_train = train.drop('dischargestatus',axis=1)
        X_train = X_train.drop('uhid',axis=1)
        #X_train = X_train.drop('visittime',axis=1)

        y_test = test['dischargestatus']

        print('train uhid length=',len(train.uhid.unique()), ' UHID =', train.uhid.unique())
        print('test uhid length=',len(test.uhid.unique()), ' UHID =', test.uhid.unique())

        #train.to_csv('train_hdp.csv')
        #test.to_csv('test_hdp.csv')
        #how many rows of machine data is present for death and discharge cases in test and train
        deathTrain = train[train['dischargestatus']==1]
        dischargeTrain = train[train['dischargestatus']==0]
        print('Train death case  total length =', len(deathTrain))
        print('Train discharge case  total length =', len(dischargeTrain))
        #and test['spo2']!=-999
        deathTrain = deathTrain[deathTrain['spo2']!=-999]
        dischargeTrain = dischargeTrain[dischargeTrain['spo2']!=-999]
        print('Train death case  machine correct length =', len(deathTrain))
        print('Train discharge case  machine correct length =', len(dischargeTrain))

        deathTest = test[test['dischargestatus']==1]
        dischargeTest = test[test['dischargestatus']==0]
        print('Test death case  total length =', len(deathTest))
        print('Test discharge case  total length =', len(dischargeTest))
        deathTest = deathTest[deathTest['spo2']!=-999]
        dischargeTest = dischargeTest[dischargeTest['spo2']!=-999]

        print('Test death case  machine correct length =', len(deathTest))
        print('Test discharge case  machine correct length =', len(dischargeTest))

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
        return Xtrain,Xtest,ytrain1,ytest1,test
    except Exception as e:
            print('Error in make_lstm method',e)
            PrintException()
            return None


def visualizeLSTMOutput(xTestWithUHID,hdpPlotdict):
    try:
        currentFigure = None
        path = os.getcwd()       
        for i in xTestWithUHID.uhid.unique():            
            print('Inside visualizeLSTMOutput = ',i)
            hdpAX = hdpPlotdict.get(i)
            #print('hdpAX = ',hdpAX)
            x = xTestWithUHID[xTestWithUHID['uhid']==i]
            deathOrDischargeCase = 'Death'
            print(x.columns)
            #print('case is',x.dischargestatus.head(1))      
            if (x.iloc[0].dischargestatus == 1):
                deathOrDischargeCase = 'Death_Cases'
            elif (x.iloc[0].dischargestatus == 0):
                deathOrDischargeCase = 'Discharge_Cases'
            y_pred = np.array(x['y_pred']).flatten()
            y_df = y_pred
            rcParams['figure.figsize'] = 20, 6
            if not(hdpAX is None) :
                currentFigure = hdpAX.get_figure()
                #print('currentFigure=',currentFigure)
                axes = currentFigure.gca()
                sns.set(font_scale = 2)
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
                currentFigure.savefig(path+'/'+deathOrDischargeCase+'/'+str(i)+'.png',dpi = 300)
        return True
    except Exception as e:
        print ('Exception',e)
        PrintException()
        return None

#LSTM model
def lstm_model(n,gd,hdpPlotdict):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,restore_best_weights=True,patience=3)
    auc_roc_inter = []
    val_a = []
    train_a = []
    print('-----------Inside lstm model-------------',n,len(gd))
    for i in range(5):
        try:
            print('-----------Iteration No-------------=',i)
            print('-----------gd.uhid-------------=',gd.uhid.unique())
            Xtrain,Xtest,ytrain1,ytest1,xTestWithUHID = make_lstm_visualize(gd)
            #Building the LSTM model
            X = Input(shape=(None, n), name='X')
            mX = Masking()(X)
            # X-> BidirectionalLSTM -> MASKING LAYER -> LSTM -> Dense -> y 
            lstm = Bidirectional(LSTM(units=512,activation='tanh',return_sequences=True,recurrent_dropout=0.5,dropout=0.3))
            #for timeseries data of varying length - in case one patient has 15 pulse rate values vs the second patient has 10
            # masking will handle this mismatch of data
            mX = lstm(mX)
            L = LSTM(units=64,activation='tanh',return_sequences=False)(mX)
            y = Dense(1, activation="sigmoid")(L)
            outputs = [y]
            inputs = [X]
            model = Model(inputs,outputs)
            model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
            #validation accuracy
            v_a = []
            #training accuracy
            t_a = []
            #fitting the model
            model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=1,callbacks=[es])
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
            continue
        except Exception as e:
            print('Exception inside lstm_model',i,e)
            PrintException()
            continue
    print('-----------Done with lstm model-------------')
    #so for each iteration we have also captured the validation and training accuracy, this is returned along with auc, roc values
    return auc_roc_inter,list(itertools.chain(*val_a)),list(itertools.chain(*train_a))

def convert_date(x):
    print(x)
    x = str(x)
    return pd.to_datetime(x)

def predictLSTM(gw, fixed, cont, inter,hdpPlotdict):
    try:
        print('Inside predictLSTM column count=',gw.columns)
        #defining the early stopping criteria
        f_a = []
        i_a = []
        c_a = []
        ci_a = []
        cf_a = []
        fi_a = []
        a = []
        #reduced 2 for uhid and dischargestatus, 1 extra for hour_series - temporary testing
        lengthOfFixed = len(fixed) - 2
        #reduced 2 for uhid and dischargestatus
        lengthOfIntermittent = len(inter) - 2
        #reduced 2 for uhid and dischargestatus
        lengthOfContinuous = len(cont) - 2
        """
        gd = gw[fixed]
        print('total length of gd=',len(gd),'gd count',gd.count())
        an = lstm_model(lengthOfFixed,gd,hdpPlotdict)
        f_a.append(an[0])
        print('-----AN-------',an)
        print('---------fixed----------',f_a)
        print(mean_confidence_interval(an[0]))
        gd = gw[inter]
        an = lstm_model(lengthOfIntermittent,gd,lstm_model)
        i_a.append(an[0])
        print('inter',i_a)
        print(mean_confidence_interval(an[0]))
        """
        gd = gw[cont]
        #this will remove all indexes where continuous data is not present
        gd = gd[gd["spo2"] != -999]
        print('---------------AFTER CHECK of SPO2 =-9999--------------')
        an = lstm_model(lengthOfContinuous,gd,hdpPlotdict)
        c_a.append(an[0])
        print('----------c_a----------->',c_a)
        visualizeDataFrameDataset(gd,'cont')    
        print(mean_confidence_interval(an[0]))
        """

        #---------------CONT+INTER------------------
        cont_inter = list(set(cont+inter))
        gd = gw[cont_inter]
         #this will remove all indexes where continuous data is not present
        gd = gd[gd["spo2"] != -999]       
        an = lstm_model(lengthOfIntermittent+lengthOfContinuous,gd,hdpPlotdict)
        ci_a.append(an[0])
        print('cont_inter',ci_a)
        print(mean_confidence_interval(an[0]))
        #---------------FIXED+INTER------------------
        fixed_inter = list(set(fixed+inter))
        gd = gw[fixed_inter]
        an = lstm_model(lengthOfFixed+lengthOfIntermittent,gd,hdpPlotdict)
        fi_a.append(an[0])
        print('fixed_inter',fi_a)
        print(mean_confidence_interval(an[0]))
        #---------------CONT+FIXED------------------
        cont_fixed = list(set(cont+fixed))
        gd = gw[cont_fixed]
        #this will remove all indexes where continuous data is not present
        gd = gd[gd["spo2"] != -999]        
        an = lstm_model(lengthOfFixed+lengthOfContinuous,gd,hdpPlotdict)
        cf_a.append(an[0])
        print('cont_fixed',cf_a)
        print(mean_confidence_interval(an[0]))
        #---------------CONT+FIXED+INTER------------------
        all_cols = list(set(cont+inter+fixed))
        gd = gw[all_cols]
        #this will remove all indexes where continuous data is not present
        gd = gd[gd["spo2"] != -999]        
        an = lstm_model(lengthOfFixed+lengthOfIntermittent+lengthOfContinuous,gd,hdpPlotdict)
        a.append(an[0])
        print('all_cols',a)
        print(mean_confidence_interval(an[0]))

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
        """
        return True
    except Exception as e:
        print('Exception in Prediction', e)
        PrintException()
        return None    