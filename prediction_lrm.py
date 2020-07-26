import math
from entropy import sample_entropy
import nolds
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
import scipy.stats


def range_finder(x):
    length = x
    fractional = (x/15.0) - math.floor(x/15.0)
    return int(round(fractional*15))

def conception(x):
    if x == 'ivf':
        return 1
    else:
        return 0

def baby_type(x):
    if x == 'Twins' or x == 'Triplets':
        return 1
    else:
        return 0

def gender(x):
    if x == 'Male':
        return 1
    else:
        return 0
    
def mod(x):
    if x == 'LSCS':
        return 1
    else:
        return 0
    
def steroid(x):
    if x == 'beta':
        return 1
    else:
        return 0

def training(X,y):
    auc_roc = []
    training = []

    for i in range(25):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.3 
                                                            )
            # SMOTE
            sm = SMOTE(k_neighbors = 2)
            X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)
            y_smote = logistic(X_train_sm, X_test, y_train_sm)
            y_smote_1 = logistic(X_train_sm,X_train,y_train_sm)
            auc_roc.append(roc_auc_score(y_test,y_smote))
            training.append(roc_auc_score(y_train,y_smote_1))
            continue
        except Exception as e:
            print(e,"error")
            continue
    return auc_roc,training

def logistic(X_train, X_test, y_train):
    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test) 
    return y_pred

def in_out(x):
    if x == 'Out Born':
        return 1
    else:
        return 0
        
def discharge(x):
    if x == 'Discharge':
        return 0
    else:
        return 1

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m,3), round(m-h,3), round(m+h,3)

def prepareLRMData(data):

    dg = pd.DataFrame(columns=['uhid','dischargestatus','pulserate_SE','pulserate_DFA','pulserate_ADF','pulserate_Mean','pulserate_Var','ecg_resprate_SE','ecg_resprate_DFA','ecg_resprate_ADF','ecg_resprate_Mean','ecg_resprate_Var',
                          'spo2_SE','spo2_DFA','spo2_ADF','spo2_Mean','spo2_Var','heartrate_SE','heartrate_DFA','heartrate_ADF','heartrate_Mean','heartrate_Var','peep_SE','peep_DFA','peep_ADF','peep_Mean','peep_Var',
                          'pip_SE','pip_DFA','pip_ADF','pip_Mean','pip_Var','map_SE','map_DFA','map_ADF','map_Mean','map_Var','tidalvol_SE','tidalvol_DFA','tidalvol_ADF','tidalvol_Mean','tidalvol_Var',
                          'minvol_SE','minvol_DFA','minvol_ADF','minvol_Mean','minvol_Var','ti_SE','ti_DFA','ti_ADF','ti_Mean','ti_Var','fio2_SE','fio2_DFA','fio2_ADF','fio2_Mean','fio2_Var','abdomen_girth',
       'urine', 'totalparenteralvolume', 'new_ph',
       'gender', 'birthweight', 'birthlength', 'birthheadcircumference',
       'inout_patient_status', 'gestation',
       'baby_type', 'central_temp', 'apgar_onemin', 'apgar_fivemin',
       'apgar_tenmin', 'motherage', 'conception_type', 'mode_of_delivery',
       'steroidname', 'numberofdose', 'rbs', 'temp', 
       'currentdateweight', 'currentdateheight', 'tpn-tfl','mean_bp', 'sys_bp',
       'dia_bp','abd_difference','stool_day_total','total_intake','typevalue_Antibiotics', 'typevalue_Inotropes'])

    for i in data.uhid.unique():
        try:
            print(i)
            t = data[data['uhid']==i]
            t.fillna(t.mean(),inplace=True)
            t.fillna(0,inplace=True)
            #t = t.apply(pd.to_numeric, errors='coerce')
            dg = dg.append({'uhid':i,'dischargestatus':t['dischargestatus'].iloc[0],'pulserate_SE':sample_entropy(t.pulserate, order=2, metric='chebyshev'),'pulserate_DFA':nolds.dfa(t.pulserate),'pulserate_ADF':adfuller(t.pulserate)[0],'pulserate_Mean':np.nanmean(t.pulserate),'pulserate_Var':np.nanvar(t.pulserate),'ecg_resprate_SE':sample_entropy(t.ecg_resprate, order=2, metric='chebyshev'),'ecg_resprate_DFA':nolds.dfa(t.ecg_resprate),'ecg_resprate_ADF':adfuller(t.ecg_resprate)[0],'ecg_resprate_Mean':np.mean(t.ecg_resprate),'ecg_resprate_Var':np.var(t.ecg_resprate),
                                'spo2_SE':sample_entropy(t.spo2, order=2, metric='chebyshev'),'spo2_DFA':nolds.dfa(t.spo2),'spo2_ADF':adfuller(t.spo2)[0],'spo2_Mean':np.mean(t.spo2),'spo2_Var':np.var(t.spo2),'heartrate_SE':sample_entropy(t.heartrate, order=2, metric='chebyshev'),'heartrate_DFA':nolds.dfa(t.heartrate),'heartrate_ADF':adfuller(t.heartrate)[0],'heartrate_Mean':np.mean(t.heartrate),'heartrate_Var':np.var(t.heartrate),'peep_SE':sample_entropy(t.peep, order=2, metric='chebyshev'),'peep_DFA':nolds.dfa(t.peep),'peep_ADF':adfuller(t.peep)[0],'peep_Mean':np.mean(t.peep),'peep_Var':np.var(t.peep),
                                'pip_SE':sample_entropy(t.pip, order=2, metric='chebyshev'),'pip_DFA':nolds.dfa(t.pip),'pip_ADF':adfuller(t.pip)[0],'pip_Mean':np.mean(t.pip),'pip_Var':np.var(t.pip),'map_SE':sample_entropy(t.map, order=2, metric='chebyshev'),'map_DFA':nolds.dfa(t.map),'map_ADF':adfuller(t.map)[0],'map_Mean':np.mean(t.map),'map_Var':np.var(t.map),'tidalvol_SE':sample_entropy(t.tidalvol, order=2, metric='chebyshev'),'tidalvol_DFA':nolds.dfa(t.tidalvol),'tidalvol_ADF':adfuller(t.tidalvol)[0],'tidalvol_Mean':np.mean(t.tidalvol),'tidalvol_Var':np.var(t.tidalvol),
                                'minvol_SE':sample_entropy(t.minvol, order=2, metric='chebyshev'),'minvol_DFA':nolds.dfa(t.minvol),'minvol_ADF':adfuller(t.minvol)[0],'minvol_Mean':np.mean(t.minvol),'minvol_Var':np.var(t.minvol),'ti_SE':sample_entropy(t.ti, order=2, metric='chebyshev'),'ti_DFA':nolds.dfa(t.ti),'ti_ADF':adfuller(t.ti)[0],'ti_Mean':np.mean(t.ti),'ti_Var':np.var(t.ti),'fio2_SE':sample_entropy(t.fio2, order=2, metric='chebyshev'),'fio2_DFA':nolds.dfa(t.fio2),'fio2_ADF':adfuller(t.fio2)[0],'fio2_Mean':np.mean(t.fio2),'fio2_Var':np.var(t.fio2),'abdomen_girth':np.mean(t.abdomen_girth),
            'urine':np.nansum(t.urine), 'totalparenteralvolume':np.nansum(t.totalparenteralvolume),'total_intake':np.nansum(t.total_intake), 'new_ph':np.mean(t.new_ph),
            'gender':t['gender'].iloc[0], 'birthweight':t['birthweight'].iloc[0], 'birthlength':t['birthlength'].iloc[0], 'birthheadcircumference':t['birthheadcircumference'].iloc[0],
            'inout_patient_status':t['inout_patient_status'].iloc[0], 'gestation':t['gestation'].iloc[0], 
            'baby_type':t['baby_type'].iloc[0], 'central_temp':np.nanmean(t.central_temp), 'apgar_onemin':t['apgar_onemin'].iloc[0], 'apgar_fivemin':t['apgar_fivemin'].iloc[0],
            'apgar_tenmin':t['apgar_tenmin'].iloc[0], 'motherage':t['motherage'].iloc[0], 'conception_type':t['conception_type'].iloc[0], 'mode_of_delivery':t['mode_of_delivery'].iloc[0],
            'numberofdose':np.nansum(t.numberofdose), 'rbs':np.nanmean(t.rbs), 'temp':np.nanmean(t.temp),
            'currentdateweight':np.nanmean(t.currentdateweight), 'currentdateheight':np.nanmean(t.currentdateheight),
            'mean_bp':np.nanmean(t.mean_bp),'dia_bp':np.nanmean(t.dia_bp),'sys_bp':np.nanmean(t.sys_bp),'stool_day_total':np.nanmean(t.stool_day_total),
            'tpn-tfl':np.nansum(t['tpn-tfl']), 'typevalue_Inotropes':np.nansum(t.typevalue_Inotropes),
            'typevalue_Antibiotics':np.nansum(t.typevalue_Antibiotics),'steroidname':np.nansum(t.steroidname),
            },ignore_index=True)
            
            
        except Exception as e:
            print(e,"error")

    return dg

def predictionUsingLRM(dg):
    dg = dg.replace([np.inf, -np.inf], np.nan)

    cont = ['ecg_resprate_ADF', 'ecg_resprate_DFA', 'ecg_resprate_Mean',
        'ecg_resprate_SE', 'ecg_resprate_Var', 'fio2_ADF', 'fio2_DFA',
        'fio2_Mean', 'fio2_SE', 'fio2_Var', 'heartrate_ADF', 'heartrate_DFA',
        'heartrate_Mean', 'heartrate_SE', 'heartrate_Var', 
        'pulserate_ADF', 'pulserate_DFA', 'pulserate_Mean', 'pulserate_SE',
        'pulserate_Var', 'spo2_ADF', 'spo2_DFA', 'spo2_Mean', 'spo2_SE',
        'spo2_Var']

    fixed = ['gender', 'birthweight',
        'birthlength', 'birthheadcircumference', 'inout_patient_status', 'baby_type', 'central_temp',
        'apgar_onemin', 'apgar_fivemin', 'apgar_tenmin', 'motherage',
        'conception_type', 'mode_of_delivery', 'steroidname', 'numberofdose',
        'gestation']

    inter = [ 'central_temp','currentdateheight', 'currentdateweight', 'new_ph', 'rbs', 'stool_day_total','temp', 'total_intake', 'totalparenteralvolume',
    'peep_DFA','peep_Mean','peep_SE','peep_Var','pip_ADF','pip_DFA','pip_Mean','pip_SE','pip_Var','map_ADF',
    'map_DFA','map_Mean','map_SE','map_Var','tidalvol_ADF','tidalvol_DFA','tidalvol_Mean','tidalvol_SE','tidalvol_Var','minvol_ADF','minvol_DFA','minvol_Mean','minvol_SE','minvol_Var','ti_ADF','ti_DFA','ti_Mean','ti_SE','ti_Var','fio2_ADF','fio2_DFA','fio2_Mean','fio2_SE','fio2_Var','mean_bp','sys_bp','dia_bp']

    #Continuous
    X = dg[cont]
    y = dg['dischargestatus']
    X = X.fillna(X.mean())
    an = training(X,y)
    c_a = mean_confidence_interval(an[0])
    c_t = mean_confidence_interval(an[1])


    #Fixed
    X = dg[fixed]
    y = dg['dischargestatus']
    X = X.fillna(X.mean())
    an = training(X,y)
    f_a = mean_confidence_interval(an[0])
    f_t = mean_confidence_interval(an[1])


    #Intermittent
    X = dg[inter]
    y = dg['dischargestatus']
    X = X.fillna(X.mean())
    an = training(X,y)
    i_a = mean_confidence_interval(an[0])
    i_t = mean_confidence_interval(an[1])

    #Fixed + Continuous
    fixed_cont = list(set(fixed+cont))
    X = dg[fixed_cont]
    y = dg['dischargestatus']
    X = X.fillna(X.mean())
    an = training(X,y)
    fc_a = mean_confidence_interval(an[0])
    fc_t = mean_confidence_interval(an[1])

    #Fixed + Intermittent
    fixed_inter = list(set(fixed+inter))
    X = dg[fixed_inter]
    y = dg['dischargestatus']
    X = X.fillna(X.mean())
    an = training(X,y)
    fi_a = mean_confidence_interval(an[0])
    fi_t = mean_confidence_interval(an[1])

    #Continuous + Intermittent
    cont_inter = list(set(cont+inter))
    X = dg[cont_inter]
    y = dg['dischargestatus']
    X = X.fillna(X.mean())
    an = training(X,y)
    ci_a = mean_confidence_interval(an[0])
    ci_t = mean_confidence_interval(an[1])

    #ALL
    all_cols = list(set(cont+inter+fixed))
    X = dg[all_cols]
    y = dg['dischargestatus']
    X = X.fillna(X.mean())
    an = training(X,y)
    a = mean_confidence_interval(an[0])
    t = mean_confidence_interval(an[1])

    from prettytable import PrettyTable
    l = [["Fixed" ,f_t, f_a],["Inter ", i_t, i_a],["Cont", c_t, c_a],["Fixed + Inter", fi_t, fi_a],["Fixed + Cont", fc_t, fc_a],["Inter + Cont", ci_t, ci_a],["All", t, a]]

    table = PrettyTable(['Parameter', 'Training (Mean Lower Upper)', 'Testing (Mean Lower Upper)'])

    for rec in l:
        table.add_row(rec)
        
    print(table)
    

def forwardFillData(data): 

    cols_to_use =  ['uhid','pulserate', 'ecg_resprate',
        'spo2', 'heartrate', 'mean_bp', 'sys_bp', 'dia_bp',
        'peep', 'pip', 'map', 'tidalvol', 'minvol', 'ti', 'fio2',
        'abd_difference_y',
        'abdomen_girth','currentdateheight',
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



    df = data[cols_to_use]
    
    uhid = df.uhid.unique()
    ds = pd.DataFrame(columns=df.columns)
    for i in uhid:
        x = df[df['uhid']==i]
        x.currentdateweight.fillna(method='ffill',inplace=True)
        x.currentdateheight.fillna(method='ffill',inplace=True)
        x.central_temp.fillna(method='ffill',inplace=True)
        
        x.abdomen_girth.fillna(method='ffill',inplace=True)
        x.heartrate.fillna(method='ffill',limit=5,inplace=True)
        x.pulserate.fillna(method='ffill',limit=5,inplace=True)
        x.ecg_resprate.fillna(method='ffill',limit=5,inplace=True)
        x.spo2.fillna(method='ffill',limit=5,inplace=True)
        x.mean_bp.fillna(method='ffill',limit=5,inplace=True)
        x.sys_bp.fillna(method='ffill',limit=5,inplace=True)
        x.dia_bp.fillna(method='ffill',limit=5,inplace=True)
        x.minvol.fillna(method='ffill',limit=5,inplace=True)
        x.ti.fillna(method='ffill',limit=5,inplace=True)
        x.peep.fillna(method='ffill',limit=5,inplace=True)
        x.pip.fillna(method='ffill',limit=5,inplace=True)
        x.map.fillna(method='ffill',limit=5,inplace=True)
        x.tidalvol.fillna(method='ffill',limit=5,inplace=True)
        x.fio2.fillna(method='ffill',limit=5,inplace=True)
        x.rbs.fillna(method='ffill',limit=5,inplace=True)
        x.new_ph.fillna(method='ffill',limit=5,inplace=True)
        
        
        ds = ds.append(x,ignore_index=True)
        

    dt = ds.fillna(-999)
    final_df = pd.DataFrame(columns=dt.columns)
    for i in uhid:
        x = dt[dt['uhid']==i]
        x = x[range_finder(len(x)):len(x)]
        
        final_df = final_df.append(x,ignore_index=True)

    return final_df

def predictLRM(data):

    #Forward Filling and Categorical to Binary
    final_df = forwardFillData(data)

    #Comment this if LRM data is already prepared
    #dg = prepareLRMData(final_df)
    #Comment this if LRM data is to be prepared
    lrm_data = pd.read_csv('LRM_all_data.csv')

    #Predicting using LRM
    predictionUsingLRM(lrm_data)







