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

gs = pd.read_csv('LSTM_1_hour.csv')
gs.drop('Unnamed: 0',axis=1,inplace=True)

cols_to_use = ['uhid', 'ecg_resprate',
       'spo2', 'heartrate', 'mean_bp', 'sys_bp', 'dia_bp',
       'peep', 'pip', 'map', 'tidalvol', 'minvol', 'ti', 'fio2',
       'abd_difference_y',
       'currentdateheight',
       'currentdateweight','dischargestatus', 
       'new_ph', 
       'rbs',  'stool_day_total', 
       'temp', 'total_intake', 'totalparenteralvolume',
       'tpn-tfl', 'typevalue_Antibiotics', 'typevalue_Inotropes',
       'urine','gender', 'birthweight',
       'birthlength', 'birthheadcircumference', 'inout_patient_status',
       'gestationweekbylmp', 'gestationdaysbylmp',
       'baby_type', 'central_temp', 'apgar_onemin', 'apgar_fivemin',
       'apgar_tenmin', 'motherage', 'conception_type', 'mode_of_delivery',
       'steroidname', 'numberofdose', 'gestation']

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

gd = gs[cols_to_use]

final_df = gd.copy()


final_df.fillna(-999,inplace=True)

def make_lstm(gd):


    final_df = gd.copy()
    



    final_df.fillna(-999,inplace=True)


    # In[ ]:

    train = final_df[:split_70(len(final_df))]
    test = final_df[split_70(len(final_df)):]
#     train = final_df[:2520]
#     test = final_df[2520:]


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
    y_answer_final = []
    y_model_final = []
    for i in range(2):
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
        y_answer_final.append(y_answer)
        y_model_final.append(y_model)
        print("roc_auc_score")
        #print(y_answer)
        #print(y_pred)
        #print(y_test)
        auc_roc_inter.append(roc_auc_score(y_answer,y_pred))
        continue
    
    
    #print(y_model_final)
    return auc_roc_inter,list(itertools.chain(*y_model_final)),list(itertools.chain(*y_answer_final))


Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)

import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

an,y_model,y_answer = lstm_model(Xtrain,Xtest,ytrain1,ytest1)
#print(y_model)
c_a = mean_confidence_interval(an)

cm = confusion_matrix(y_answer,y_model)
ppv_1 = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
npv_1 = round(float(cm[1][1])/(cm[1][0]+cm[1][1]),2)

se_ppv = math.sqrt((ppv_1*(1-ppv_1))/(cm[0][0]+cm[0][1]))
ci_ppv_upper_1 = ppv_1 + (1.96*se_ppv)
ci_ppv_lower_1 = ppv_1 - (1.96*se_ppv)


se_npv = math.sqrt((npv_1*(1-npv_1))/(cm[1][0]+cm[1][1]))
ci_npv_upper_1 = npv_1 + (1.96*se_npv)
ci_npv_lower_1 = npv_1 - (1.96*se_npv)

l = [["1 Hour" ,c_a, ppv_1,ci_ppv_upper_1,ci_ppv_lower_1,npv_1,ci_npv_upper_1,ci_npv_lower_1]]

table = PrettyTable(['Parameter', 'AUC-ROC', 'PPV','PPV(Upper)','PPV(Lower)','NPV','NPV(Upper)','NPV(Lower)'])

for rec in l:
    table.add_row(rec)

print(table)

gs = pd.read_csv('LSTM_6_hour.csv')
gs.drop('Unnamed: 0',axis=1,inplace=True)

gd = gs[cols_to_use]

Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)


an,y_model,y_answer = lstm_model(Xtrain,Xtest,ytrain1,ytest1)

c_6 = mean_confidence_interval(an)

cm = confusion_matrix(y_answer,y_model)
ppv_6 = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
npv_6 = round(float(cm[1][1])/(cm[1][0]+cm[1][1]),2)

se_ppv = math.sqrt((ppv_6*(1-ppv_6))/(cm[0][0]+cm[0][1]))
ci_ppv_upper_6 = ppv_6 + (1.96*se_ppv)
ci_ppv_lower_6 = ppv_6 - (1.96*se_ppv)


se_npv = math.sqrt((npv_6*(1-npv_6))/(cm[1][0]+cm[1][1]))
ci_npv_upper_6 = npv_6 + (1.96*se_npv)
ci_npv_lower_6 = npv_6 - (1.96*se_npv)

gs = pd.read_csv('LSTM_12_hour.csv')
gs.drop('Unnamed: 0',axis=1,inplace=True)
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model(Xtrain,Xtest,ytrain1,ytest1)

c_12 = mean_confidence_interval(an)
cm = confusion_matrix(y_answer,y_model)
ppv_12 = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
npv_12 = round(float(cm[1][1])/(cm[1][0]+cm[1][1]),2)


se_ppv = math.sqrt((ppv_12*(1-ppv_12))/(cm[0][0]+cm[0][1]))
ci_ppv_upper_12 = ppv_12 + (1.96*se_ppv)
ci_ppv_lower_12 = ppv_12 - (1.96*se_ppv)


se_npv = math.sqrt((npv_12*(1-npv_12))/(cm[1][0]+cm[1][1]))
ci_npv_upper_12 = npv_12 + (1.96*se_npv)
ci_npv_lower_12 = npv_12 - (1.96*se_npv)


gs = pd.read_csv('LSTM_48_hour.csv')
gs.drop('Unnamed: 0',axis=1,inplace=True)
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model(Xtrain,Xtest,ytrain1,ytest1)

c_48 = mean_confidence_interval(an)
cm = confusion_matrix(y_answer,y_model)
ppv_48 = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
npv_48 = round(float(cm[1][1])/(cm[1][0]+cm[1][1]),2)

se_ppv = math.sqrt((ppv_48*(1-ppv_48))/(cm[0][0]+cm[0][1]))
ci_ppv_upper_48 = ppv_48 + (1.96*se_ppv)
ci_ppv_lower_48 = ppv_48 - (1.96*se_ppv)


se_npv = math.sqrt((npv_48*(1-npv_48))/(cm[1][0]+cm[1][1]))
ci_npv_upper_48 = npv_48 + (1.96*se_npv)
ci_npv_lower_48 = npv_48 - (1.96*se_npv)

l = [["1 Hour" ,c_a, ppv_1,ci_ppv_upper_1,ci_ppv_lower_1,npv_1,ci_npv_upper_1,ci_npv_lower_1],["6 Hour" ,c_6, ppv_6,ci_ppv_upper_6,ci_ppv_lower_6,npv_6,ci_npv_upper_6,ci_npv_lower_6],["12 Hour" ,c_12, ppv_12,ci_ppv_upper_12,ci_ppv_lower_12,npv_12,ci_npv_upper_12,ci_npv_lower_12],["48 Hour" ,c_48, ppv_48,ci_ppv_upper_48,ci_ppv_lower_48,npv_48,ci_npv_upper_48,ci_npv_lower_48]]
table = PrettyTable(['Parameter', 'AUC-ROC', 'PPV','PPV(Upper)','PPV(Lower)','NPV','NPV(Upper)','NPV(Lower)'])
for rec in l:
    table.add_row(rec)

print(table)


gs = pd.read_csv('LSTM_1_week.csv')
gs.drop('Unnamed: 0',axis=1,inplace=True)
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model(Xtrain,Xtest,ytrain1,ytest1)

c_1w = mean_confidence_interval(an)
cm = confusion_matrix(y_answer,y_model)
ppv_1w = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
npv_1w = round(float(cm[1][1])/(cm[1][0]+cm[1][1]),2)

se_ppv = math.sqrt((ppv_1w*(1-ppv_1w))/(cm[0][0]+cm[0][1]))
ci_ppv_upper_1w = ppv_1w + (1.96*se_ppv)
ci_ppv_lower_1w = ppv_1w - (1.96*se_ppv)


se_npv = math.sqrt((npv_1w*(1-npv_1w))/(cm[1][0]+cm[1][1]))
ci_npv_upper_1w = npv_1w + (1.96*se_npv)
ci_npv_lower_1w = npv_1w - (1.96*se_npv)


gs = pd.read_csv('LSTM_2_week.csv')
gs.drop('Unnamed: 0',axis=1,inplace=True)
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model(Xtrain,Xtest,ytrain1,ytest1)

c_2w = mean_confidence_interval(an)
cm = confusion_matrix(y_answer,y_model)
ppv_2w = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
npv_2w = round(float(cm[1][1])/(cm[1][0]+cm[1][1]),2)

se_ppv = math.sqrt((ppv_2w*(1-ppv_2w))/(cm[0][0]+cm[0][1]))
ci_ppv_upper_2w = ppv_2w + (1.96*se_ppv)
ci_ppv_lower_2w = ppv_2w - (1.96*se_ppv)


se_npv = math.sqrt((npv_2w*(1-npv_2w))/(cm[1][0]+cm[1][1]))
ci_npv_upper_2w = npv_2w + (1.96*se_npv)
ci_npv_lower_2w = npv_2w - (1.96*se_npv)


gs = pd.read_csv('LSTM_3_week.csv')
gs.drop('Unnamed: 0',axis=1,inplace=True)
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model(Xtrain,Xtest,ytrain1,ytest1)

c_3w = mean_confidence_interval(an)
cm = confusion_matrix(y_answer,y_model)
ppv_3w = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
npv_3w = round(float(cm[1][1])/(cm[1][0]+cm[1][1]),2)

se_ppv = math.sqrt((ppv_3w*(1-ppv_3w))/(cm[0][0]+cm[0][1]))
ci_ppv_upper_3w = ppv_3w + (1.96*se_ppv)
ci_ppv_lower_3w = ppv_3w - (1.96*se_ppv)


se_npv = math.sqrt((npv_3w*(1-npv_3w))/(cm[1][0]+cm[1][1]))
ci_npv_upper_3w = npv_3w + (1.96*se_npv)
ci_npv_lower_3w = npv_3w - (1.96*se_npv)


gs = pd.read_csv('LSTM_4_week.csv')
gs.drop('Unnamed: 0',axis=1,inplace=True)
gd = gs[cols_to_use]
Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
an,y_model,y_answer = lstm_model(Xtrain,Xtest,ytrain1,ytest1)

c_4w = mean_confidence_interval(an)
cm = confusion_matrix(y_answer,y_model)
ppv_4w = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
npv_4w = round(float(cm[1][1])/(cm[1][0]+cm[1][1]),2)


se_ppv = math.sqrt((ppv_4w*(1-ppv_4w))/(cm[0][0]+cm[0][1]))
ci_ppv_upper_4w = ppv_4w + (1.96*se_ppv)
ci_ppv_lower_4w = ppv_4w - (1.96*se_ppv)


se_npv = math.sqrt((npv_4w*(1-npv_4w))/(cm[1][0]+cm[1][1]))
ci_npv_upper_4w = npv_4w + (1.96*se_npv)
ci_npv_lower_4w = npv_4w - (1.96*se_npv)


l = [["1 week" ,c_1w, ppv_1w,ci_ppv_upper_1w,ci_ppv_lower_1w,npv_1w,ci_npv_upper_1w,ci_npv_lower_1w],["2 Week" ,c_2w, ppv_2w,ci_ppv_upper_2w,ci_ppv_lower_2w,npv_2w,ci_npv_upper_2w,ci_npv_lower_2w],["3 Week" ,c_3w, ppv_3w,ci_ppv_upper_3w,ci_ppv_lower_3w,npv_3w,ci_npv_upper_3w,ci_npv_lower_3w],["4 Week" ,c_4w, ppv_4w,ci_ppv_upper_4w,ci_ppv_lower_4w,npv_4w,ci_npv_upper_4w,ci_npv_lower_4w]]


table = PrettyTable(['Parameter', 'AUC-ROC', 'PPV','PPV(Upper)','PPV(Lower)','NPV','NPV(Upper)','NPV(Lower)'])
for rec in l:
    table.add_row(rec)

print(table)
