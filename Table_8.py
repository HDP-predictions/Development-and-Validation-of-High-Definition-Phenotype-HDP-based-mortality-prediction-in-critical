import pandas as pd
import numpy as np
from keras.layers import Activation, Dense, Dropout, SpatialDropout1D,Input,Masking,Bidirectional, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
from keras.models import Sequential, Model
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from random import seed
#seed(1)
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import math
from sklearn.metrics import roc_auc_score
import itertools
from sklearn.metrics import classification_report, confusion_matrix
from prettytable import PrettyTable

#defining the early stopping criteria
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,restore_best_weights=True,patience=3)

def split_70(x):
    return int((round((x/15)*0.7))*15)

def range_finder(p):
    length = p
    fractional = (p/15.0) - math.floor(p/15.0)
    return int(round(fractional*15))

def randomize(u):
    
    f_df = pd.DataFrame(columns=u.columns)
    for i in u.uhid.unique():
        x = u[u['uhid']==i]
        x = x[range_finder(len(x)):len(x)]

        f_df = f_df.append(x,ignore_index=True)
        
    return f_df
    
def make_lstm(gd):


    final_df = randomize(gd)
    


    final_df.fillna(-999,inplace=True)


    # In[ ]:

    death_cases = final_df[final_df.dischargestatus == 1]
    discharge_cases = final_df[final_df.dischargestatus == 0]
    
    train_death = death_cases[:split_70(len(death_cases))]
    test_death = death_cases[split_70(len(death_cases)):]
    
    train_discharge = discharge_cases[:split_70(len(discharge_cases))]
    test_discharge = discharge_cases[split_70(len(discharge_cases)):]
    
    train = pd.concat([train_death,train_discharge])
    test = pd.concat([test_death,test_discharge])
    


    # In[ ]:


    y_train = train['dischargestatus']
    X_train = train.drop('dischargestatus',axis=1)
    X_train = X_train.drop('uhid',axis=1)
    #X_train = X_train.drop('visittime',axis=1)

    y_test = test['dischargestatus']
    X_test = test.drop('dischargestatus',axis=1)
    X_test = X_test.drop('uhid',axis=1)
    #X_test = X_test.drop('startdate',axis=1)


    # In[ ]:

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

    return Xtrain,Xtest,ytrain1,ytest1

def lstm_model(Xtrain,Xtest,ytrain1,ytest1):
    auc_roc_inter = []
    val_a = []
    train_a = []
    for i in range(2):
        #Building the LSTM model
        X = Input(shape=(None, 1), name='X')
        mX = Masking()(X)
        lstm = Bidirectional(LSTM(units=512,activation='tanh',return_sequences=True,recurrent_dropout=0.5,dropout=0.3))
        mX = lstm(mX)
        L = LSTM(units=64,activation='tanh',return_sequences=False)(mX)
        y = Dense(1, activation="sigmoid")(L)
        outputs = [y]
        inputs = [X]
        model = Model(inputs,outputs)
        model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

        v_a = []
        t_a = []
        #fitting the model
        model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
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

        def acc(x):
            if x>0.5:
                return 1
            else:
                return 0

        y_model=[]
        for i in y_pred:
            y_model.append(acc(i))
        y_answer=[]
        for j in y_test:
            y_answer.append(acc(j))
            
        val_a.append(v_a)
        train_a.append(t_a)
        auc_roc_inter.append(roc_auc_score(y_answer,y_pred))
        continue
    
    
        
    return auc_roc_inter,y_model,y_answer





gs = pd.read_csv('lstm_analysis.csv')

gs.drop('Unnamed: 0',axis=1,inplace=True)
gs.drop('Unnamed: 0.1',axis=1,inplace=True)

cols_to_use = ['uhid','pulserate', 'dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model(Xtrain,Xtest,ytrain1,ytest1)
death = gs[gs['dischargestatus']==1]
dis = gs[gs['dischargestatus']==0]
auc_pulserate = an[0]
pulserate_dea = (float(len(death[death['pulserate']==-999]))/len(death))*100
pulserate_dis = (float(len(dis[dis['pulserate']==-999]))/len(dis))*100
print("pulse rate")
print(auc_pulserate)
print(pulserate_dea)
print(pulserate_dis)


cols_to_use = ['uhid','heartrate', 'dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model(Xtrain,Xtest,ytrain1,ytest1)
auc_heartrate = an[0]
heartrate_dea = (float(len(death[death['heartrate']==-999]))/len(death))*100
heartrate_dis = (float(len(dis[dis['heartrate']==-999]))/len(dis))*100
print("heart rate")
print(auc_heartrate)
print(heartrate_dea)
print(heartrate_dis)

cols_to_use = ['uhid','ecg_resprate', 'dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model(Xtrain,Xtest,ytrain1,ytest1)
auc_rr = an[0]
rr_dea = (float(len(death[death['ecg_resprate']==-999]))/len(death))*100
rr_dis = (float(len(dis[dis['ecg_resprate']==-999]))/len(dis))*100
print("RR")
print(auc_rr)
print(rr_dea)
print(rr_dis)

cols_to_use = ['uhid','spo2', 'dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model(Xtrain,Xtest,ytrain1,ytest1)
auc_spo2 = an[0]
spo2_dea = (float(len(death[death['spo2']==-999]))/len(death))*100
spo2_dis = (float(len(dis[dis['spo2']==-999]))/len(dis))*100
print("spo2")
print(auc_spo2)
print(spo2_dea)
print(spo2_dis)