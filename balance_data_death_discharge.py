
import pandas as pd
import numpy as np
# from keras.layers import Activation, Dense, Dropout, SpatialDropout1D,Input,Masking,Bidirectional, TimeDistributed
# from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM,GRU
# from keras.models import Sequential, Model
# from keras.preprocessing import sequence
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping
from random import seed
# from sklearn.metrics import roc_auc_score
#seed(1)
#print("second")
# import tensorflow as tf
# from keras.layers import Input, Dense
# from keras.models import Model
import scipy.stats
from prettytable import PrettyTable
import math
import itertools
from random import shuffle
#print("first")
import psycopg2
from random import shuffle
from datetime import timedelta

# Function to select random patients
# For given set of death patients, select similar profile discharge patients

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

def actual_birthdate(x,y):
    try:
        
        return pd.to_datetime(x) + timedelta(seconds=y)

    except Exception as e:
        print("error", e)
        pass

def calculateLOS(x,y):
    try:
        
        return math.ceil((x - y).total_seconds()/60)

    except Exception as e:
        print("error", e)
        pass

def second_addition(x):
    try:
        if x.split(",")[2] == 'PM':
            if x.split(",")[0] != '12':
                return (12 + float(x.split(",")[0]) + float(x.split(",")[1])/60)*3600
            else:
                return (float(x.split(",")[0]) + float(x.split(",")[1])/60)*3600
        else:
            return (float(x.split(",")[0]) + float(x.split(",")[1])/60)*3600
    except:
        
        if x.split(" ")[1] == 'PM':
            if (x.split(" ")[0]).split(":")[0] != '12':
                return (12 + float((x.split(" ")[0]).split(":")[0]) + float((x.split(" ")[0]).split(":")[1])/60)*3600
            else:
                return (float((x.split(" ")[0]).split(":")[0]) + float((x.split(" ")[0]).split(":")[1])/60)*3600
        else:
            return (float((x.split(" ")[0]).split(":")[0]) + float((x.split(" ")[0]).split(":")[1])/60)*3600

def randomize(dis,dea):
    #Unnamed: 0 is used as a proxy for LOS in minutes
    df = pd.DataFrame(columns=dis.columns)
    #the gestation categories are <26, 26-28, 28-32, and > 32
    #find out death and discharge case for gestation less than 26 weeks
    dis_less_26 = dis[dis['gestation']<=26]
    dea_less_26 = dea[dea['gestation']<=26]
    dis_less_26.sort_values(['los','uhid'],ascending=False)
    dea_less_26.sort_values(['los','uhid'],ascending=False)
    #find unirque patient ids for selected gestation
    dis_less_26_uhid = dis_less_26.uhid.unique()
    dea_less_26_uhid = dea_less_26.uhid.unique()

    #print(dea_less_26.uhid.unique())
    taken = []
    tk = set()
    #iterated over all death cases meeting the gestation category condition
    for i in dea_less_26.uhid.unique():
        x_dea = dea_less_26[dea_less_26['uhid']==i]
        # o = dk[dk['uhid']==i]
        # dis_date = o.dischargeddate.iloc[0]
        # x_dea = x_dea[x_dea['hour_series']<=dis_date]
        x_dea['dischargestatus'] = 1
        
        #Unnamed: 0 denotes number of rows per minute for given UHID
        minutesDataDeathHDPPatient = x_dea['los'].iloc[0]
        #Find all discharge cases that meet the condition > = Unnamed: 0
        #print(type(minutesDataDeathHDPPatient))
        #print(dis_less_26)
        x_dis = dis_less_26[dis_less_26['los']>=minutesDataDeathHDPPatient]
        #remove the selected UHID from the available selection candidates
        ids_dis = set(x_dis.uhid.unique())
        #print(ids_dis,"ids_dis")
        ids = list(ids_dis - ids_dis.intersection(tk))
        #print(ids,"ids")
        shuffle(ids)
        x = dis_less_26[dis_less_26['uhid']==ids[0]]
        # o1 = dk[dk['uhid']==ids[0]]
        # dis_date1 = o1.dischargeddate.iloc[0]
        # x = x[x['hour_series']<=dis_date1]
        x['dischargestatus'] = 0
        #pick the data from discharge patient for same number of minutes as in death case
        x = x[:len(x_dea)]
        y = pd.concat([x,x_dea])
        taken.append(ids[0])
        #remove the selected UHID from the available selection candidates by inserting it in tk
        tk = set(taken)
        df = df.append(y,ignore_index=True)
    taken = []
    tk = set()
    dis_less_28 = dis[(dis['gestation']>26)&(dis['gestation']<=28)]
    dea_less_28 = dea[(dea['gestation']>26)&(dea['gestation']<=28)]
    dis_less_28.sort_values(['los','uhid'],ascending=False)
    dea_less_28.sort_values(['los','uhid'],ascending=False)
    dis_less_28_uhid = dis_less_28.uhid.unique()
    dea_less_28_uhid = dea_less_28.uhid.unique()
    for i in dea_less_28.uhid.unique():
        x_dea = dea_less_28[dea_less_28['uhid']==i]
        # o = dk[dk['uhid']==i]
        # dis_date = o.dischargeddate.iloc[0]
        # x_dea = x_dea[x_dea['hour_series']<=dis_date]
        x_dea['dischargestatus'] = 1
        l = x_dea['los'].iloc[0]
        x_dis = dis_less_28[dis_less_28['los']>l]
        ids_dis = set(x_dis.uhid.unique())
        ids = list(ids_dis - ids_dis.intersection(tk))
        shuffle(ids)
        x = dis_less_28[dis_less_28['uhid']==ids[0]]
        # o1 = dk[dk['uhid']==ids[0]]
        # dis_date1 = o1.dischargeddate.iloc[0]
        # x = x[x['hour_series']<=dis_date1]
        x['dischargestatus'] = 0
        x = x[:len(x_dea)]
        y = pd.concat([x,x_dea])
        taken.append(ids[0])
        tk = set(taken)
        df = df.append(y,ignore_index=True)
    taken = []
    tk = set()
    dis_less_32 = dis[(dis['gestation']>28)&(dis['gestation']<=32)]
    dea_less_32 = dea[(dea['gestation']>28)&(dea['gestation']<=32)]
    dis_less_32.sort_values(['los','uhid'],ascending=False)
    dea_less_32.sort_values(['los','uhid'],ascending=False)
    dis_less_32_uhid = dis_less_32.uhid.unique()
    dea_less_32_uhid = dea_less_32.uhid.unique()
    for i in dea_less_32.uhid.unique():
        x_dea = dea_less_32[dea_less_32['uhid']==i]
        # o = dk[dk['uhid']==i]
        # dis_date = o.dischargeddate.iloc[0]
        # x_dea = x_dea[x_dea['hour_series']<=dis_date]
        x_dea['dischargestatus'] = 1
        l = x_dea['los'].iloc[0]
        x_dis = dis_less_32[dis_less_32['los']>l]
        ids_dis = set(x_dis.uhid.unique())
        ids = list(ids_dis - ids_dis.intersection(tk))
        shuffle(ids)
        x = dis_less_32[dis_less_32['uhid']==ids[0]]
        # o1 = dk[dk['uhid']==ids[0]]
        # dis_date1 = o1.dischargeddate.iloc[0]
        # x = x[x['hour_series']<=dis_date1]
        x['dischargestatus'] = 0
        x = x[:len(x_dea)]
        y = pd.concat([x,x_dea])
        taken.append(ids[0])
        tk = set(taken)
        df = df.append(y,ignore_index=True)
    taken = []
    tk = set()
    dis_32 = dis[dis['gestation']>32]
    dea_32 = dea[dea['gestation']>32]
    dis_32.sort_values(['los','uhid'],ascending=False)
    dea_32.sort_values(['los','uhid'],ascending=False)
    dis_32_uhid = dis_32.uhid.unique()
    dea_32_uhid = dea_32.uhid.unique()
    for i in dea_32.uhid.unique():
        x_dea = dea_32[dea_32['uhid']==i]
        # o = dk[dk['uhid']==i]
        # dis_date = o.dischargeddate.iloc[0]
        # x_dea = x_dea[x_dea['hour_series']<=dis_date]
        x_dea['dischargestatus'] = 1
        l = x_dea['los'].iloc[0]
        x_dis = dis_32[dis_32['los']>l]
        ids_dis = set(x_dis.uhid.unique())
        ids = list(ids_dis - ids_dis.intersection(tk))
        shuffle(ids)
        x = dis_32[dis_32['uhid']==ids[0]]
        # o1 = dk[dk['uhid']==ids[0]]
        # dis_date1 = o1.dischargeddate.iloc[0]
        # x = x[x['hour_series']<=dis_date1]
        x['dischargestatus'] = 0
        x = x[:len(x_dea)]
        y = pd.concat([x,x_dea])
        taken.append(ids[0])
        tk = set(taken)
        df = df.append(y,ignore_index=True)

    return df


def fetchingDischargeDeathset(con, flag):
    cur  = con.cursor()
    #print("connected to database")

    #Firstly taking only Death cases
    cur10 = con.cursor()
    cur10.execute("SELECT DISTINCT(uhid) as uhid,birthweight,dischargestatus, round( CAST((gestationweekbylmp + gestationdaysbylmp/7::float) as numeric),2) as gestation, dischargeddate, dateofadmission, timeofadmission FROM apollo.baby_detail WHERE  dateofadmission >= '2018-07-01' AND dateofadmission <= '2020-05-31' and UHID IN  (select distinct(uhid) from apollo.babyfeed_detail where uhid in  ( select distinct(uhid) from apollo.baby_visit where uhid in (select  distinct(uhid) from apollo.nursing_vitalparameters where uhid in ( select distinct(t1.uhid) from apollo.device_monitor_detail as t1 LEFT JOIN apollo.baby_detail AS t2 ON t1.uhid=t2.uhid WHERE t1.starttime <t2.dischargeddate and t2.isreadmitted is not true UNION select distinct(t1.uhid) from apollo.device_monitor_detail_dump as t1 LEFT JOIN apollo.baby_detail AS t2 ON t1.uhid=t2.uhid  WHERE t1.starttime < t2.dischargeddate and t2.isreadmitted is not true))))and (dischargestatus = 'Death') and isreadmitted is not true and gestationweekbylmp is not null and birthweight is not null;")
    cols10 = list(map(lambda x: x[0], cur10.description))
    death_cases = pd.DataFrame(cur10.fetchall(),columns=cols10)

    #calculating count of spo2 value in each death patient
    maxSpo2Count = 0
    spo2CountList = []
    cur11 = con.cursor()

    for uhid in  death_cases.uhid.unique():
        cur11.execute("select t1.starttime from apollo.device_monitor_detail as t1 WHERE t1.uhid ='" + uhid + "' and t1.spo2 is not null  UNION select t2.starttime from apollo.device_monitor_detail_dump as t2  WHERE t2.uhid = '" + uhid + "' and t2.spo2 is not null")
        cols11 = list(map(lambda x: x[0], cur11.description))
        uhid_data = pd.DataFrame(cur11.fetchall(),columns=cols11)
        print(uhid)
        # if(len(uhid_data)) > maxSpo2Count:
        #     maxSpo2Count = len(uhid_data)
        print(len(uhid_data))
        spo2CountList.append(len(uhid_data))

    #Sorted in Descending order
    spo2CountList.sort(reverse = True)

    #Now taking up only the discharge cases
    cur10 = con.cursor()
    cur10.execute("SELECT DISTINCT(uhid) as uhid,birthweight,dischargestatus, round( CAST((gestationweekbylmp + gestationdaysbylmp/7::float) as numeric),2) as gestation, dischargeddate, dateofadmission, timeofadmission FROM apollo.baby_detail WHERE  dateofadmission >= '2018-07-01' AND dateofadmission <= '2020-05-31' and UHID IN  (select distinct(uhid) from apollo.babyfeed_detail where uhid in  ( select distinct(uhid) from apollo.baby_visit where uhid in (select  distinct(uhid) from apollo.nursing_vitalparameters where uhid in ( select distinct(t1.uhid) from apollo.device_monitor_detail as t1 LEFT JOIN apollo.baby_detail AS t2 ON t1.uhid=t2.uhid  WHERE t1.spo2 is not null and t1.starttime <t2.dischargeddate and t2.isreadmitted is not true UNION select distinct(t1.uhid) from apollo.device_monitor_detail_dump as t1 LEFT JOIN apollo.baby_detail AS t2 ON t1.uhid=t2.uhid  WHERE t1.spo2 is not null and t1.starttime < t2.dischargeddate and t2.isreadmitted is not true)))) and (dischargestatus = 'Discharge') and isreadmitted is not true and gestationweekbylmp is not null and birthweight is not null;")
    cols10 = list(map(lambda x: x[0], cur10.description))
    discharge_cases = pd.DataFrame(cur10.fetchall(),columns=cols10)
    discharge_cases_uhid = discharge_cases.uhid.unique()

    #Prepare the string of all discharge cases to be given as an input in next query
    dischargeUhidString = ""
    counter = 0
    for uhid in discharge_cases_uhid:
        if counter == 0:
            dischargeUhidString = "'" + str(uhid) + "'"
            counter = counter + 1
        else :
            dischargeUhidString = dischargeUhidString + ",'" + uhid + "'"

    #For every death case picking up the discharge case according to the count of spo2 of death patient
    finalDischargeCases = pd.DataFrame()
    dischargeCaseDataframe = pd.DataFrame()
    for count in spo2CountList:
        cur11 = con.cursor()
        cur11.execute("((select distinct(uhid) as uhid ,count(*) as sum from apollo.device_monitor_detail where spo2 is not null group by uhid having count(*) > " + str(count) + " and uhid IN ( " + dischargeUhidString +")) UNION  (select distinct(uhid) as uhid ,count(*) as sum from apollo.device_monitor_detail_dump where spo2 is not null group by uhid having  count(*) > " + str(count) + " and uhid IN ( " + dischargeUhidString +"))) order by sum limit 1")
        cols11 = list(map(lambda x: x[0], cur11.description))
        device_data = pd.DataFrame(cur11.fetchall(),columns=cols11)
        print(count , "death count")
        print(device_data.sum , "discharge count")
        print(device_data.uhid , "discharge uhid")

        #Fetching the chosen uhid and then append to the discharge List
        dischargeCaseDataframe = discharge_cases[discharge_cases.uhid == device_data.uhid[0]]
        finalDischargeCases = finalDischargeCases.append(dischargeCaseDataframe)
        
    #Concatenate Discharge and Death cases
    gs = pd.concat([finalDischargeCases,death_cases])

    print(len(gs))

    #print(len(gs))
    #gs.set_index('uhid',inplace=True)
    #gs.reset_index(inplace=True)
    dates_detail = gs.copy()
    dates_detail['add_seconds_admission'] = dates_detail['timeofadmission'].apply(second_addition)
    dates_detail['actual_DOA'] = dates_detail.apply(lambda x: actual_birthdate(x['dateofadmission'], x['add_seconds_admission']), axis=1)
    dates_detail['los'] = dates_detail.apply(lambda x: calculateLOS(x['dischargeddate'], x['actual_DOA']), axis=1)

    if flag == True:
        print("-------------Randomization used on gestation or weight condition--------")       
        gs1 = dates_detail[dates_detail['birthweight']<=1500]
        gs2 = dates_detail[(dates_detail['birthweight']>2500)&(dates_detail['birthweight']<3000)]
        #balancing the dataset
        death1 = gs1[gs1['dischargestatus']=="Death"]
        death_l1 = death1.uhid.unique()
        death2 = gs2[gs2['dischargestatus']=="Death"]
        death_l2 = death2.uhid.unique()
        dis1 = gs1[gs1['dischargestatus']=="Discharge"]
        dis_l1 = dis1.uhid.unique()
        dis2 = gs2[gs2['dischargestatus']=="Discharge"]
        dis_l2 = dis2.uhid.unique()
        #print(len(dis1))
        final = pd.DataFrame()
        final_df = pd.DataFrame(columns=dates_detail.columns)
        d1 = randomize(dis1,death1)
        d2 = randomize(dis2,death2)
        gw = pd.concat([d1,d2])
    else:

        print("-------------no randomization based on gestation or weight - equal number of babies selected from discharge without any condition--------")
        gs1 = dates_detail[dates_detail['dischargestatus']=="Death"]
        gs2 = dates_detail[dates_detail['dischargestatus']=="Discharge"]

        death_length = len(gs1)
        print('Length of death ==>',death_length)
        ids = gs2.uhid.unique()
        shuffle(ids)
        discharge_cases = ids[0:death_length]
        print('Length of discharge_cases ==>',len(discharge_cases))
        discharge = pd.DataFrame()
        for i in discharge_cases:
            print(i,"selected ID")
            x = gs2[gs2.uhid == i]
            discharge = discharge.append(x,ignore_index = True)
        gs1['dischargestatus'] = 1
        discharge['dischargestatus'] = 0
        #discharge cases may or may not have data -- can we check for continuous data??
        gw = pd.concat([gs1,discharge])
    #print(gw.uhid.unique())
    gw.to_csv("HDP_Analysis.csv")
    return gw

#balanceDataset()
