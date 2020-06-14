#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import scipy.stats
from prettytable import PrettyTable

#defining the early stopping criteria
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,restore_best_weights=True,patience=3)


# In[ ]:


gs = pd.read_csv('data_8th_june.csv')
gs.drop('Unnamed: 0',axis=1,inplace=True)
gs.drop('Unnamed: 0.1',axis=1,inplace=True)


# In[ ]:


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m,3), round(m-h,3), round(m+h,3)


# In[ ]:


def range_finder(x):
    length = x
    fractional = (x/15.0) - math.floor(x/15.0)
    return int(round(fractional*15))


# In[ ]:


cont = ['pulserate',
       'ecg_resprate', 'spo2', 'heartrate', 'dischargestatus', 'uhid']


# In[ ]:


gd = gs[cont]


# In[ ]:


import math
final_df = pd.DataFrame(columns=gd.columns)
for i in gd.uhid.unique():
    x = gd[gd['uhid']==i]
    x = x[range_finder(len(x)):len(x)]
    
    final_df = final_df.append(x,ignore_index=True)


# In[ ]:


final_df.fillna(-999,inplace=True)


# In[ ]:


train = final_df[:515340]
test = final_df[515340:]


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


# In[ ]:


for i in range(25):
    #Building the LSTM model
    X = Input(shape=(None, 4), name='X')
    mX = Masking()(X)
    lstm = Bidirectional(LSTM(units=512,activation='tanh',return_sequences=True,recurrent_dropout=0.5,dropout=0.3))
    mX = lstm(mX)
    L = LSTM(units=64,activation='tanh',return_sequences=False)(mX)
    y = Dense(1, activation="sigmoid")(L)
    outputs = [y]
    inputs = [X]
    model = Model(inputs,outputs)
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    
    #fitting the model
    model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    #history = model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    for i in range(len(model.history.history['val_acc'])):
        val_a.append(model.history.history['val_acc'][i])
        train_a.append(model.history.history['acc'][i])
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
    
    auc_roc_inter.append(roc_auc_score(y_answer,y_pred))


# In[ ]:


c_a = mean_confidence_interval(auc_roc_inter)


# In[ ]:


c_b = mean_confidence_interval(val_a)


# In[ ]:


c_c = mean_confidence_interval(train_a)


# In[ ]:


#Fixed


# In[ ]:


fixed = ['dischargestatus',  'gender', 'birthweight',
       'birthlength', 'birthheadcircumference', 'inout_patient_status',
       'gestationweekbylmp', 'gestationdaysbylmp',
       'baby_type', 'central_temp', 'apgar_onemin', 'apgar_fivemin',
       'apgar_tenmin', 'motherage', 'conception_type', 'mode_of_delivery',
       'steroidname', 'numberofdose', 'gestation','uhid']


# In[ ]:


gd = gs[fixed]


# In[ ]:


import math
final_df = pd.DataFrame(columns=gd.columns)
for i in gd.uhid.unique():
    x = gd[gd['uhid']==i]
    x = x[range_finder(len(x)):len(x)]
    
    final_df = final_df.append(x,ignore_index=True)
    


# In[ ]:


final_df.fillna(-999,inplace=True)


# In[ ]:


train = final_df[:515340]
test = final_df[515340:]


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


# In[ ]:


for i in range(25):
    #Building the LSTM model
    X = Input(shape=(None, 18), name='X')
    mX = Masking()(X)
    lstm = Bidirectional(LSTM(units=512,activation='tanh',return_sequences=True,recurrent_dropout=0.5,dropout=0.3))
    mX = lstm(mX)
    L = LSTM(units=64,activation='tanh',return_sequences=False)(mX)
    y = Dense(1, activation="sigmoid")(L)
    outputs = [y]
    inputs = [X]
    model = Model(inputs,outputs)
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    
    #fitting the model
    model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    #history = model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    for i in range(len(model.history.history['val_acc'])):
        val_a.append(model.history.history['val_acc'][i])
        train_a.append(model.history.history['acc'][i])
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
    
    auc_roc_inter.append(roc_auc_score(y_answer,y_pred))


# In[ ]:


f_a = mean_confidence_interval(auc_roc_inter)
f_b = mean_confidence_interval(val_a)
f_c = mean_confidence_interval(train_a)


# In[ ]:


#Intermittent


# In[ ]:


inter = ['dischargestatus', 'mean_bp',
       'sys_bp', 'dia_bp', 'peep', 'pip', 'map', 'tidalvol',
       'minvol', 'ti', 'fio2',
       'abd_difference_y',
       'abdomen_girth_y', 'currentdateheight', 'currentdateweight',
       
       'new_ph', 'rbs',
       'stool_day_total', 'temp',
       'total_intake', 'totalparenteralvolume',
       'tpn-tfl', 'typevalue_Antibiotics', 'typevalue_Inotropes',
       'urine', 'urine_per_hour',
       'urine_per_kg_hour', 'uhid']


# In[ ]:


gd = gs[inter]


# In[ ]:


import math
final_df = pd.DataFrame(columns=gd.columns)
for i in gd.uhid.unique():
    x = gd[gd['uhid']==i]
    x = x[range_finder(len(x)):len(x)]
    
    final_df = final_df.append(x,ignore_index=True)
    


# In[ ]:


final_df.fillna(-999,inplace=True)


# In[ ]:


train = final_df[:515340]
test = final_df[515340:]


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


# In[ ]:


for i in range(25):
    #Building the LSTM model
    X = Input(shape=(None, 26), name='X')
    mX = Masking()(X)
    lstm = Bidirectional(LSTM(units=512,activation='tanh',return_sequences=True,recurrent_dropout=0.5,dropout=0.3))
    mX = lstm(mX)
    L = LSTM(units=64,activation='tanh',return_sequences=False)(mX)
    y = Dense(1, activation="sigmoid")(L)
    outputs = [y]
    inputs = [X]
    model = Model(inputs,outputs)
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    
    #fitting the model
    model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    #history = model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    for i in range(len(model.history.history['val_acc'])):
        val_a.append(model.history.history['val_acc'][i])
        train_a.append(model.history.history['acc'][i])
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
    
    auc_roc_inter.append(roc_auc_score(y_answer,y_pred))


# In[ ]:


i_a = mean_confidence_interval(auc_roc_inter)
i_b = mean_confidence_interval(val_a)
i_c = mean_confidence_interval(train_a)


# In[ ]:


cont_inter = list(set(cont+inter))


# In[ ]:


gd = gs[cont_inter]


# In[ ]:


import math
final_df = pd.DataFrame(columns=gd.columns)
for i in gd.uhid.unique():
    x = gd[gd['uhid']==i]
    x = x[range_finder(len(x)):len(x)]
    
    final_df = final_df.append(x,ignore_index=True)
    


# In[ ]:


final_df.fillna(-999,inplace=True)


# In[ ]:


train = final_df[:515340]
test = final_df[515340:]


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


# In[ ]:


auc_roc_inter = []
from sklearn.metrics import roc_auc_score
val_a = []
train_a = []


# In[ ]:


for i in range(25):
    #Building the LSTM model
    X = Input(shape=(None, 30), name='X')
    mX = Masking()(X)
    lstm = Bidirectional(LSTM(units=512,activation='tanh',return_sequences=True,recurrent_dropout=0.5,dropout=0.3))
    mX = lstm(mX)
    L = LSTM(units=64,activation='tanh',return_sequences=False)(mX)
    y = Dense(1, activation="sigmoid")(L)
    outputs = [y]
    inputs = [X]
    model = Model(inputs,outputs)
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    
    #fitting the model
    model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    #history = model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    for i in range(len(model.history.history['val_acc'])):
        val_a.append(model.history.history['val_acc'][i])
        train_a.append(model.history.history['acc'][i])
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
    
    auc_roc_inter.append(roc_auc_score(y_answer,y_pred))


# In[ ]:


ci_a = mean_confidence_interval(auc_roc_inter)
ci_b = mean_confidence_interval(val_a)
ci_c = mean_confidence_interval(train_a)


# In[ ]:


#Fixed+Intermittent


# In[ ]:


fixed_inter = list(set(fixed+inter))


# In[ ]:


gd = gs[fixed_inter]


# In[ ]:


import math
final_df = pd.DataFrame(columns=gd.columns)
for i in gd.uhid.unique():
    x = gd[gd['uhid']==i]
    x = x[range_finder(len(x)):len(x)]
    
    final_df = final_df.append(x,ignore_index=True)


# In[ ]:


final_df.fillna(-999,inplace=True)


# In[ ]:


train = final_df[:515340]
test = final_df[515340:]


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


# In[ ]:


auc_roc_inter = []
from sklearn.metrics import roc_auc_score
val_a = []
train_a = []


# In[ ]:


for i in range(25):
    #Building the LSTM model
    X = Input(shape=(None, 44), name='X')
    mX = Masking()(X)
    lstm = Bidirectional(LSTM(units=512,activation='tanh',return_sequences=True,recurrent_dropout=0.5,dropout=0.3))
    mX = lstm(mX)
    L = LSTM(units=64,activation='tanh',return_sequences=False)(mX)
    y = Dense(1, activation="sigmoid")(L)
    outputs = [y]
    inputs = [X]
    model = Model(inputs,outputs)
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    
    #fitting the model
    model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    #history = model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    for i in range(len(model.history.history['val_acc'])):
        val_a.append(model.history.history['val_acc'][i])
        train_a.append(model.history.history['acc'][i])
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
    
    auc_roc_inter.append(roc_auc_score(y_answer,y_pred))


# In[ ]:


fi_a = mean_confidence_interval(auc_roc_inter)
fi_b = mean_confidence_interval(val_a)
fi_c = mean_confidence_interval(train_a)


# In[ ]:


#Cont+Fixed


# In[ ]:


cont_fixed = list(set(cont+fixed))


# In[ ]:


gd = gs[cont_fixed]


# In[ ]:


import math
final_df = pd.DataFrame(columns=gd.columns)
for i in gd.uhid.unique():
    x = gd[gd['uhid']==i]
    x = x[range_finder(len(x)):len(x)]
    
    final_df = final_df.append(x,ignore_index=True)


# In[ ]:


final_df.fillna(-999,inplace=True)


# In[ ]:


train = final_df[:515340]
test = final_df[515340:]


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


# In[ ]:


auc_roc_inter = []
from sklearn.metrics import roc_auc_score
val_a = []
train_a = []


# In[ ]:


for i in range(25):
    #Building the LSTM model
    X = Input(shape=(None, 22), name='X')
    mX = Masking()(X)
    lstm = Bidirectional(LSTM(units=512,activation='tanh',return_sequences=True,recurrent_dropout=0.5,dropout=0.3))
    mX = lstm(mX)
    L = LSTM(units=64,activation='tanh',return_sequences=False)(mX)
    y = Dense(1, activation="sigmoid")(L)
    outputs = [y]
    inputs = [X]
    model = Model(inputs,outputs)
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    
    #fitting the model
    model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    #history = model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    for i in range(len(model.history.history['val_acc'])):
        val_a.append(model.history.history['val_acc'][i])
        train_a.append(model.history.history['acc'][i])
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
    
    auc_roc_inter.append(roc_auc_score(y_answer,y_pred))


# In[ ]:


cf_a = mean_confidence_interval(auc_roc_inter)
cf_b = mean_confidence_interval(val_a)
cf_c = mean_confidence_interval(train_a)


# In[ ]:


all_cols = list(set(cont+inter+fixed))


# In[ ]:


gd = gs[all_cols]


# In[ ]:


import math
final_df = pd.DataFrame(columns=gd.columns)
for i in gd.uhid.unique():
    x = gd[gd['uhid']==i]
    x = x[range_finder(len(x)):len(x)]
    
    final_df = final_df.append(x,ignore_index=True)
    


# In[ ]:


final_df.fillna(-999,inplace=True)


# In[ ]:


train = final_df[:515340]
test = final_df[515340:]


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


# In[ ]:


auc_roc_inter = []
from sklearn.metrics import roc_auc_score
val_a = []
train_a = []


# In[ ]:


for i in range(25):
    #Building the LSTM model
    X = Input(shape=(None, 48), name='X')
    mX = Masking()(X)
    lstm = Bidirectional(LSTM(units=512,activation='tanh',return_sequences=True,recurrent_dropout=0.5,dropout=0.3))
    mX = lstm(mX)
    L = LSTM(units=64,activation='tanh',return_sequences=False)(mX)
    y = Dense(1, activation="sigmoid")(L)
    outputs = [y]
    inputs = [X]
    model = Model(inputs,outputs)
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    
    #fitting the model
    model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    #history = model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=38,callbacks=[es])
    for i in range(len(model.history.history['val_acc'])):
        val_a.append(model.history.history['val_acc'][i])
        train_a.append(model.history.history['acc'][i])
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
    
    auc_roc_inter.append(roc_auc_score(y_answer,y_pred))


# In[ ]:


a = mean_confidence_interval(auc_roc_inter)
b = mean_confidence_interval(val_a)
c = mean_confidence_interval(train_a)


# In[ ]:


l = [["Fixed" ,f_c, f_a],["Inter ", i_c, i_a],["Cont", c_c, c_a],["Fixed + Inter", fi_c, fi_a],["Fixed + Cont", cf_c, fc_a],["Inter + Cont", ci_c, ci_a],["All", c, a]]

table = PrettyTable(['Parameter', 'Training (Mean Lower Upper)', 'Testing (Mean Lower Upper)'])

for rec in l:
    table.add_row(rec)
    
print(table)


# In[ ]:





# In[ ]:




