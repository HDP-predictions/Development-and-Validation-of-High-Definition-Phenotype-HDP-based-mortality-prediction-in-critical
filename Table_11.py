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
from sklearn.metrics import roc_auc_score
#seed(1)
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import scipy.stats
from prettytable import PrettyTable
import math
import itertools
from random import shuffle


#defining the early stopping criteria
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,restore_best_weights=True,patience=3)

gs = pd.read_csv('lstm_analysis.csv')
gs['urine_per_kg_hour'] = gs['urine_per_hour']/gs['currentdateweight']

gs.drop('Unnamed: 0',axis=1,inplace=True)
gs.drop('Unnamed: 0.1',axis=1,inplace=True)

def split_70(x):
    return int((round((x/15)*0.7))*15)

def range_finder(x):
    length = x
    fractional = (x/15.0) - math.floor(x/15.0)
    return int(round(fractional*15))

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m,3), round(m-h,3), round(m+h,3)

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

def lstm_model(n,Xtrain,Xtest,ytrain1,ytest1):
    auc_roc_inter = []
    val_a = []
    train_a = []
    for i in range(2):
        #Building the LSTM model
        X = Input(shape=(None, n), name='X')
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


cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','new_ph','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_a = mean_confidence_interval(an)
print("PH")
print(a_a)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','rbs','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_b = mean_confidence_interval(an)
print("RBS")
print(a_b)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','temp','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_c = mean_confidence_interval(an)
print("Temp")
print(a_c)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','currentdateweight','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_d = mean_confidence_interval(an)
print("Weight")
print(a_d)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','currentdateheight','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_e = mean_confidence_interval(an)
print("Height")
print(a_e)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','urine_per_kg_hour','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_f = mean_confidence_interval(an)
print("Urine")
print(a_f)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','abd_difference_y','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_g = mean_confidence_interval(an)
print("Abdominal")
print(a_g)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','tpn-tfl','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_h = mean_confidence_interval(an)
print("TPN")
print(a_h)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','mean_bp','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_i = mean_confidence_interval(an)
print("Mean BP")
print(a_i)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','sys_bp','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_j = mean_confidence_interval(an)
print("Sys BP")
print(a_j)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','dia_bp','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_k = mean_confidence_interval(an)
print("Dia BP")
print(a_k)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','peep','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_l = mean_confidence_interval(an)
print("PEEP")
print(a_l)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','pip','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_m = mean_confidence_interval(an)
print("PIP")
print(a_m)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','map','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_n = mean_confidence_interval(an)
print("MAP")
print(a_n)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','fio2','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_o = mean_confidence_interval(an)
print("Fio2")
print(a_o)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','tidalvol','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_p = mean_confidence_interval(an)
print("TV")
print(a_p)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','minvol','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_q = mean_confidence_interval(an)
print("MV")
print(a_q)

cols_to_use = ['uhid','pulserate',  'ecg_resprate', 'spo2', 'heartrate','ti','dischargestatus']
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model((len(cols_to_use)-2),Xtrain,Xtest,ytrain1,ytest1)
a_r = mean_confidence_interval(an)
print("TI")
print(a_r)

