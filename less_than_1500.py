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
from sklearn.metrics import roc_auc_score
#seed(1)
print("second")
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
import scipy.stats
from prettytable import PrettyTable
import math
import itertools
from random import shuffle
print("first")
import psycopg2
from random import shuffle



con = psycopg2.connect (user = 'postgres',
                password = 'postgres',
                port = '5432',
                host = 'localhost',
                database = 'inicudb')

cur  = con.cursor()

print("second")

cur10 = con.cursor()
cur10.execute("SELECT DISTINCT(uhid) FROM apollo.baby_detail WHERE dateofadmission >= '2018-07-01' AND dateofadmission <= '2020-05-31' and UHID IN  (select distinct(uhid) from apollo.babyfeed_detail where uhid in  ( select distinct(uhid) from apollo.baby_visit where uhid in (select  distinct(uhid) from apollo.nursing_vitalparameters where uhid in ( select distinct(uhid) from apollo.device_monitor_detail UNION select distinct(uhid) from apollo.device_monitor_detail_dump ))))and (dischargestatus = 'Death'  OR dischargestatus = 'Discharge') and isreadmitted is not true and gestationweekbylmp is not null and birthweight is not null;")
cols10 = list(map(lambda x: x[0], cur10.description))
fixed = pd.DataFrame(cur10.fetchall(),columns=cols10)

#defining the early stopping criteria
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,restore_best_weights=True,patience=3)


# In[ ]:

print("2222")
gq = pd.read_csv('data_23_june.csv')
#gq.drop('Unnamed: 0',axis=1,inplace=True)
t = gq.groupby('uhid').count()
t.reset_index(inplace=True)
t = t[['uhid','Unnamed: 0']]
gq.drop('Unnamed: 0',axis=1,inplace=True)
gs = pd.merge(gq,t,on=['uhid'],how='left')

#gs.drop('Unnamed: 0.1',axis=1,inplace=True)
print("111")
ids = gs.uhid.unique()

print(gs.columns)

# In[ ]:


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


# Function to select random patients

def randomize(dis,dea):
    #Unnamed: 0 is used as a proxy for LOS in minutes
    df = pd.DataFrame(columns=dis.columns)
    dis_less_26 = dis[dis['gestation']<=26]
    dea_less_26 = dea[dea['gestation']<=26]
    dis_less_26.sort_values(['Unnamed: 0','uhid'],ascending=False)
    dea_less_26.sort_values(['Unnamed: 0','uhid'],ascending=False)
    dis_less_26_uhid = dis_less_26.uhid.unique()
    dea_less_26_uhid = dea_less_26.uhid.unique()
    taken = []
    tk = set()
    for i in dea_less_26.uhid.unique():
        x_dea = dea_less_26[dea_less_26['uhid']==i]
        l = x_dea['Unnamed: 0'].iloc[0]
        x_dis = dis_less_26[dis_less_26['Unnamed: 0']>=l]
        ids_dis = set(x_dis.uhid.unique())
        ids = list(ids_dis - ids_dis.intersection(tk))

        shuffle(ids)
        x = dis_less_26[dis_less_26['uhid']==ids[0]]
        x = x[:len(x_dea)]
        y = pd.concat([x,x_dea])
        taken.append(ids[0])
        tk = set(taken)
        df = df.append(y,ignore_index=True)
    taken = []
    tk = set()
    dis_less_28 = dis[(dis['gestation']>26)&(dis['gestation']<=28)]
    dea_less_28 = dea[(dea['gestation']>26)&(dea['gestation']<=28)]
    dis_less_28.sort_values(['Unnamed: 0','uhid'],ascending=False)
    dea_less_28.sort_values(['Unnamed: 0','uhid'],ascending=False)
    dis_less_28_uhid = dis_less_28.uhid.unique()
    dea_less_28_uhid = dea_less_28.uhid.unique()
    for i in dea_less_28.uhid.unique():
        x_dea = dea_less_28[dea_less_28['uhid']==i]
        l = x_dea['Unnamed: 0'].iloc[0]
        x_dis = dis_less_28[dis_less_28['Unnamed: 0']>l]
        ids_dis = set(x_dis.uhid.unique())
        ids = list(ids_dis - ids_dis.intersection(tk))
        shuffle(ids)
        x = dis_less_28[dis_less_28['uhid']==ids[0]]
        x = x[:len(x_dea)]
        y = pd.concat([x,x_dea])
        taken.append(ids[0])
        tk = set(taken)
        df = df.append(y,ignore_index=True)
    taken = []
    tk = set()
    dis_less_32 = dis[(dis['gestation']>28)&(dis['gestation']<=32)]
    dea_less_32 = dea[(dea['gestation']>28)&(dea['gestation']<=32)]
    dis_less_32.sort_values(['Unnamed: 0','uhid'],ascending=False)
    dea_less_32.sort_values(['Unnamed: 0','uhid'],ascending=False)
    dis_less_32_uhid = dis_less_32.uhid.unique()
    dea_less_32_uhid = dea_less_32.uhid.unique()
    for i in dea_less_32.uhid.unique():
        x_dea = dea_less_32[dea_less_32['uhid']==i]
        l = x_dea['Unnamed: 0'].iloc[0]
        x_dis = dis_less_32[dis_less_32['Unnamed: 0']>l]
        ids_dis = set(x_dis.uhid.unique())
        ids = list(ids_dis - ids_dis.intersection(tk))
        shuffle(ids)
        x = dis_less_32[dis_less_32['uhid']==ids[0]]
        x = x[:len(x_dea)]
        y = pd.concat([x,x_dea])
        taken.append(ids[0])
        tk = set(taken)
        df = df.append(y,ignore_index=True)
    taken = []
    tk = set()
    dis_32 = dis[dis['gestation']>32]
    dea_32 = dea[dea['gestation']>32]
    dis_32.sort_values(['Unnamed: 0','uhid'],ascending=False)
    dea_32.sort_values(['Unnamed: 0','uhid'],ascending=False)
    dis_32_uhid = dis_32.uhid.unique()
    dea_32_uhid = dea_32.uhid.unique()
    for i in dea_34.uhid.unique():
        x_dea = dea_32[dea_32['uhid']==i]
        l = x_dea['Unnamed: 0'].iloc[0]
        x_dis = dis_32[dis_32['Unnamed: 0']>l]
        ids_dis = set(x_dis.uhid.unique())
        ids = list(ids_dis - ids_dis.intersection(tk))
        shuffle(ids)
        x = dis_32[dis_32['uhid']==ids[0]]
        x = x[:len(x_dea)]
        y = pd.concat([x,x_dea])
        taken.append(ids[0])
        tk = set(taken)
        df = df.append(y,ignore_index=True)








    #df = pd.DataFrame(columns=dt.columns)

    #shuffle(ids)
    #uhid = ids[:50]
    #for i in uhid:
    #    x = dt[dt['uhid']==i]
    #    df = df.append(x,ignore_index=True)

    return df

#Function for converting data to multiples of 15
def range_finder(x):
    length = x
    fractional = (x/15.0) - math.floor(x/15.0)
    return int(round(fractional*15))


cont = ['pulserate',
       'ecg_resprate', 'spo2', 'heartrate', 'dischargestatus', 'uhid']

#gd = gs[cont]

def make_lstm(gs):
    #gs1 = gs[gs['birthweight']<=1500]
    #gs2 = gs[(gs['birthweight']>2500)&(gs['birthweight']<3000)]

    #balancing the dataset
    #death1 = gs1[gs1['dischargestatus']==1]
    #death_l1 = death1.uhid.unique()
    #death2 = gs2[gs2['dischargestatus']==1]
    #death_l2 = death2.uhid.unique()
    #dis1 = gs1[gs1['dischargestatus']==0]
    #dis_l1 = dis1.uhid.unique()
    #dis2 = gs2[gs2['dischargestatus']==0]
    #dis_l2 = dis2.uhid.unique()
    #print(len(dis1))
    #final = pd.DataFrame()
    final_df = pd.DataFrame(columns=gs.columns)
    #d1 = randomize(dis1,death1)
    #d2 = randomize(dis2,death2)
    #print(len(d1))
    #d1 = d1[:len(death1)]
    #d2 = d2[:len(death2)]
    #gd = pd.concat([d1,d2])
    ids = gs.uhid.unique()
    print(len(ids))
    shuffle(ids)
    for i in ids:
        x = gd[gd['uhid']==i]
        x = x[range_finder(len(x)):len(x)]

        final_df = final_df.append(x,ignore_index=True)



    final_df.fillna(-999,inplace=True)


    # In[ ]:


    train = final_df[:split_70(len(final_df))]
    test = final_df[split_70(len(final_df)):]


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


#LSTM model
def lstm_model(n,gd):
    auc_roc_inter = []
    val_a = []
    train_a = []


    for i in range(25):
        try:
            Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)
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


            print(i)

            val_a.append(v_a)
            train_a.append(t_a)
            auc_roc_inter.append(roc_auc_score(y_answer,y_pred))
            continue
        except:
            continue
    return auc_roc_inter,list(itertools.chain(*val_a)),list(itertools.chain(*train_a))
f_a = []
i_a = []
c_a = []
ci_a = []
cf_a = []
fi_a = []
a = []


for i in range(5):

    gs1 = gs[gs['birthweight']<=1500]
    gs2 = gs[(gs['birthweight']>2500)&(gs['birthweight']<3000)]

    #balancing the dataset
    death1 = gs1[gs1['dischargestatus']==1]
    death_l1 = death1.uhid.unique()
    death2 = gs2[gs2['dischargestatus']==1]
    death_l2 = death2.uhid.unique()
    dis1 = gs1[gs1['dischargestatus']==0]
    dis_l1 = dis1.uhid.unique()
    dis2 = gs2[gs2['dischargestatus']==0]
    dis_l2 = dis2.uhid.unique()
    print(len(dis1))
    final = pd.DataFrame()
    final_df = pd.DataFrame(columns=gs.columns)
    d1 = randomize(dis1,death1)
    d2 = randomize(dis2,death2)
    gw = pd.concat([d1,d2])



    fixed = ['dischargestatus',  'gender', 'birthweight',
           'birthlength', 'birthheadcircumference', 'inout_patient_status',
           'gestationweekbylmp', 'gestationdaysbylmp',
           'baby_type', 'central_temp', 'apgar_onemin', 'apgar_fivemin',
           'apgar_tenmin', 'motherage', 'conception_type', 'mode_of_delivery',
           'steroidname', 'numberofdose', 'gestation','uhid']


    # In[ ]:

    gd = gw[fixed]



    # In[ ]:



    an = lstm_model(18,gd)

    f_a.append(an[0])


    #f_b = mean_confidence_interval(an[1])



    #f_c = mean_confidence_interval(an[2])

    #print('Fixed')
    #print(f_a,f_b,f_c)
    #print(an[0])




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


    gd = gw[inter]


    an = lstm_model(26,gd)

    i_a.append(an[0])
    #i_b = mean_confidence_interval(an[1])
    #i_c = mean_confidence_interval(an[2])

    #print('Inter')
    #print(i_a,i_b,i_c)
    #print(an[0])
    cont = ['pulserate',
           'ecg_resprate', 'spo2', 'heartrate', 'dischargestatus', 'uhid']


    gd = gw[cont]

    an = lstm_model(4,gd)

    c_a.append(an[0])


    #c_b = mean_confidence_interval(an[1])



    #c_c = mean_confidence_interval(an[2])

    #print('Cont')
    #print(c_a,c_b,c_c)
    #print(an[0])
    cont_inter = list(set(cont+inter))
    gd = gw[cont_inter]

    an = lstm_model(30,gd)

    ci_a.append(an[0])


    #ci_b = mean_confidence_interval(an[1])



    #ci_c = mean_confidence_interval(an[2])

    #print('Cont_Inter')
    #print(ci_a,ci_b,ci_c)
    #print(an[0])


    fixed_inter = list(set(fixed+inter))


    gd = gw[fixed_inter]
    an = lstm_model(44,gd)

    fi_a.append(an[0])


    #fi_b = mean_confidence_interval(an[1])



    #fi_c = mean_confidence_interval(an[2])

    #print('Fixed_Inter')
    #print(fi_a,fi_b,fi_c)
    #print(an[0])

    cont_fixed = list(set(cont+fixed))
    gd = patients[cont_fixed]
    an = lstm_model(22,gd)

    cf_a.append(an[0])


    #cf_b = mean_confidence_interval(an[1])



    #cf_c = mean_confidence_interval(an[2])

    #print('Fixed_Cont')
    #print(cf_a,cf_b,cf_c)
    #print(an[0])

    all_cols = list(set(cont+inter+fixed))
    gd = patients[all_cols]

    an = lstm_model(48,gd)
    a.append(an[0])
    #b = mean_confidence_interval(an[1])
    #c = mean_confidence_interval(an[2])





# In[ ]:

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





# In[ ]:
