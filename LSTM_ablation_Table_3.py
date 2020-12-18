
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
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from random import shuffle

#defining the early stopping criteria
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,restore_best_weights=True,patience=3)


# In[ ]:


gs_full = pd.read_csv('/part1/data_23_june.csv')
gs_full.drop('Unnamed: 0',axis=1,inplace=True)
gs_full.drop('Unnamed: 0.1',axis=1,inplace=True)
print("111")

# In[ ]:

#70:30 split
def split_70(x):
    return int((round((x/15)*0.7))*15)


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


cont = ['pulserate',
       'ecg_resprate', 'spo2', 'heartrate', 'dischargestatus', 'uhid']

gs = pd.DataFrame()
death_babies = gs_full[gs_full.dischargestatus == 1]
discharge_babies = gs_full[gs_full.dischargestatus == 0]

ids_discharge = discharge_babies.uhid.unique()
shuffle(ids_discharge)

gs = gs.append(death_babies)

for k in range (0,15):
    id_uhid = ids_discharge[k]
    discharge_set = discharge_babies[discharge_babies.uhid == id_uhid]
    gs = gs.append(discharge_set)
    print(len(discharge_set))

print(gs.uhid.unique())

gd = gs[cont]

def make_lstm(gd):


    
    final_df = gd.copy()



    final_df.fillna(-999,inplace=True)


    # In[ ]:


    train = final_df[:split_70(len(final_df))]
    test = final_df[range_finder(len(final_df)):]


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
    print(X_train.shape)
    print(X_test.shape)
    Xtrain = np.reshape(X_train, (-1, 15, X_train.shape[1]))
    Xtest = np.reshape(X_test, (-1, 15, X_test.shape[1]))

    return Xtrain,Xtest,ytrain1,ytest1



def lstm_model(n,xtrain,xtest,ytrain1,ytest1):
    auc_roc_inter = []
    val_a = []
    train_a = []
    y_answer_final = []
    y_model_final = []

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
        model.fit(Xtrain, ytrain1, batch_size=60 ,validation_split=0.15,epochs=2,callbacks=[es])
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
        y_model_final.append(y_model)
        y_answer_final.append(y_answer)

        print(i)

        val_a.append(v_a)
        train_a.append(t_a)
        auc_roc_inter.append(roc_auc_score(y_answer,y_pred))
        continue
    return auc_roc_inter,list(itertools.chain(*val_a)),list(itertools.chain(*train_a)),list(itertools.chain(*y_answer_final)),list(itertools.chain(*y_model_final))


gd = gs[cont]

Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)

an = lstm_model(4,Xtrain,Xtest,ytrain1,ytest1)

c_a = mean_confidence_interval(an[0])


c_b = mean_confidence_interval(an[1])



c_c = mean_confidence_interval(an[2])
cm = confusion_matrix(an[3],an[4])
Precision = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
Recall = round(float(cm[0][0])/(cm[1][0]+cm[0][0]),2)
f1_score_cont = 2*((Precision*Recall)/(Precision+Recall))
auprc_cont = average_precision_score(an[3], an[4])   

print('continuous result',c_a,c_b,c_c)


fixed = ['dischargestatus',  'gender', 'birthweight',
       'birthlength', 'birthheadcircumference', 'inout_patient_status',
       'gestationweekbylmp', 'gestationdaysbylmp',
       'baby_type', 'central_temp', 'apgar_onemin', 'apgar_fivemin',
       'apgar_tenmin', 'motherage', 'conception_type', 'mode_of_delivery',
       'steroidname', 'numberofdose', 'gestation','uhid']


# In[ ]:


gd = gs[fixed]

Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)

# In[ ]:



an = lstm_model(18,Xtrain,Xtest,ytrain1,ytest1)

f_a = mean_confidence_interval(an[0])


f_b = mean_confidence_interval(an[1])



f_c = mean_confidence_interval(an[2])



cm = confusion_matrix(an[3],an[4])
Precision = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
Recall = round(float(cm[0][0])/(cm[1][0]+cm[0][0]),2)
f1_score_fixed = 2*((Precision*Recall)/(Precision+Recall))
auprc_fixed = average_precision_score(an[3], an[4])    
print(f1_score_fixed)
print(auprc_fixed) 
print('fixed result',f_a,f_b,f_c)

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


Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)


an = lstm_model(26,Xtrain,Xtest,ytrain1,ytest1)

i_a = mean_confidence_interval(an[0])
i_b = mean_confidence_interval(an[1])
i_c = mean_confidence_interval(an[2])
cm = confusion_matrix(an[3],an[4])
Precision = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
Recall = round(float(cm[0][0])/(cm[1][0]+cm[0][0]),2)
f1_score_inter = 2*((Precision*Recall)/(Precision+Recall))
auprc_inter = average_precision_score(an[3], an[4])

print('intermittent result', i_a,i_b,i_c)

cont_inter = list(set(cont+inter))


    # In[ ]:


gd = gs[cont_inter]


Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)


an = lstm_model(30,Xtrain,Xtest,ytrain1,ytest1)

ci_a = mean_confidence_interval(an[0])
ci_b = mean_confidence_interval(an[1])
ci_c = mean_confidence_interval(an[2])


cm = confusion_matrix(an[3],an[4])
Precision = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
Recall = round(float(cm[0][0])/(cm[1][0]+cm[0][0]),2)
f1_score_continter = 2*((Precision*Recall)/(Precision+Recall))
auprc_continter = average_precision_score(an[3], an[4])


print('continuous internittent result',ci_a,ci_b,ci_c)
fixed_inter = list(set(fixed+inter))


# In[ ]:


gd = gs[fixed_inter]


Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)


an = lstm_model(44,Xtrain,Xtest,ytrain1,ytest1)

fi_a = mean_confidence_interval(an[0])
fi_b = mean_confidence_interval(an[1])
fi_c = mean_confidence_interval(an[2])


cm = confusion_matrix(an[3],an[4])
Precision = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
Recall = round(float(cm[0][0])/(cm[1][0]+cm[0][0]),2)
f1_score_fixedinter = 2*((Precision*Recall)/(Precision+Recall))
auprc_fixedinter = average_precision_score(an[3], an[4])



print('fixed intermittent result',fi_a,fi_b,fi_c)

cont_fixed = list(set(cont+fixed))


gd = gs[cont_fixed]


Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)




an = lstm_model(22,Xtrain,Xtest,ytrain1,ytest1)

cf_a = mean_confidence_interval(an[0])
cf_b = mean_confidence_interval(an[1])
cf_c = mean_confidence_interval(an[2])

cm = confusion_matrix(an[3],an[4])
Precision = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
Recall = round(float(cm[0][0])/(cm[1][0]+cm[0][0]),2)
f1_score_contfixed = 2*((Precision*Recall)/(Precision+Recall))
auprc_contfixed = average_precision_score(an[3], an[4])




print('continuous fixed',cf_a,cf_b,cf_c)





all_cols = list(set(cont+inter+fixed))


gd = gs[all_cols]


Xtrain,Xtest,ytrain1,ytest1 = make_lstm(gd)

an = lstm_model(48,Xtrain,Xtest,ytrain1,ytest1)

a = mean_confidence_interval(an[0])
b = mean_confidence_interval(an[1])
c = mean_confidence_interval(an[2])



cm = confusion_matrix(an[3],an[4])
Precision = round(float(cm[0][0])/(cm[0][0]+cm[0][1]),2)
Recall = round(float(cm[0][0])/(cm[1][0]+cm[0][0]),2)
f1_score_all = 2*((Precision*Recall)/(Precision+Recall))
auprc_all = average_precision_score(an[3], an[4])


print('all result',a,b,c)
l = [["Fixed" ,f_c, f_b,f_a],["Inter ", i_c, i_b,i_a],["Cont", c_c,c_b, c_a],["Fixed + Inter", fi_c,fi_b, fi_a],["Fixed + Cont", cf_c, cf_b,cf_a],["Inter + Cont", ci_c, ci_b,ci_a],["All", c, b,a]]

table = PrettyTable(['Parameter', 'Training (Mean Lower Upper)', 'Validation (Mean Lower Upper)','AUC-ROC (Mean Lower Upper)'])
for rec in l:
    table.add_row(rec)

print(table)


print('F1 score and AUPRC score')
print(f1_score_fixed,auprc_fixed)
print(f1_score_inter,auprc_inter)
print(f1_score_cont,auprc_cont)
print(f1_score_fixedinter,auprc_fixedinter)
#print(f1_score_continter,auprc_continter)
print(f1_score_contfixed,auprc_contfixed)
print(f1_score_continter,auprc_continter)
print(f1_score_all,auprc_all)
