import os
import sys
import linecache
import pandas as pd
import numpy as np
import psycopg2
import math
from datetime import timedelta

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

def urine_check(x):
    try:
        if x > 200:
            return x/10
        else:
            return x
    except:
        pass

def to_float(x):
    if x == '' or x == None:
        return np.nan
    else:
        return float(x)

def split_date(x):
    try:
        t=str(x)
        return t.split(":")[0] + ":" + t.split(":")[1]
    except:
        pass

def split_hour(x):
    try:
        t=str(x)
        return t.split(":")[0]
    except:
        pass
    
def split_date_1(x):
    try:
        t=str(x)
        return str(t.split(" ")[0])
    except:
        pass

def actual_birthdate(x,y):
    try:
        return pd.to_datetime(x) + timedelta(seconds=y)
    except:
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

def day(x):
    return x.total_seconds()/86400

def ph_func(x):
    try:
        p = float(x)
        if p>=6.80 and p<=7.8:
            return p
        else:
            return None
    except:
        pass

def temp_a(x,y):
    if not(x is None) and (x>0):
        return x
    else:
        return y
def stamp_1(x,y):
    return str(x) +" "+ str(y).split(":")[0]

def to_str(x):
    return (str(x))

def abd(x):
    try:   
        if not(x is None) and x>200:
            return x/10
        else:
            return x
    except:
        pass

def abd_2(x):
    if x < 0:
        return ((-1)*x)/10
    elif x > 200:
        return x/10
    else:
        return x


def abd_3(x):
    if not(x is None) and x < 15:
        return np.nan
    elif not(x is None) and x > 50:
        return np.nan
    else:
        return x


def upm(x):
    if not(x is None) and x > 100:
        return np.nan
    else:
        return x


def temp_f_c(x):
    if  x != None and x > 90:
        return (x - 32)*(5/9)
    else:
        return x
    

def weight_correct(x):
    if not(x is None) and x<10:
        return x*100
    elif not(x is None) and x>5000:
        return x/10
    else:
        return x

def weight_correct_2(x):
    if not(x is None) and x<300:
        return x*10
    elif x>5000:
        return x/10
    else:
        return x
    
def to_date(x):
    return pd.to_datetime(x)

def height_correct(x):
    if not(x is None) and x < 10:
        return np.nan
    else:
        return x
    
def rbs_correct(x):
    if not(x is None) and x > 700:
        return np.nan
    else:
        return x

def con_time(x):
    return str(x).split("+")[0].split(":")[0] + ":" +str(x).split("+")[0].split(":")[1]

def con_time_2(x):
    return str(x).split(":")[0] +":"+str(x).split(":")[1]

def conception(x):
    print(x);
    if 'ivf' in x:
        return 1
    else:
        return 0
    
def mod(x):
    print(x);
    if 'LSCS' in x:
        return 1
    else:
        return 0
    
def steroid(x):
    print(x);
    if 'beta' in x:
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

def range_finder(x):
    length = x
    fractional = (x/15.0) - math.floor(x/15.0)
    return int(round(fractional*15))

def in_out(x):
    if x == 'Out Born':
        return 1
    else:
        return 0


def fetchingFromDatabase(typeQuery,schemaName,patientCaseUHID,cur,conditionCase,caseType):
    query = "";

    if typeQuery == "fixed":
        query = "SELECT t1.uhid,t1.dateofbirth,t1.timeofbirth,t1.timeofadmission,t1.dateofadmission,t1.gender,t1.birthweight,t1.birthlength,t1.birthheadcircumference,t1.inout_patient_status,round( CAST((t1.gestationweekbylmp + t1.gestationdaysbylmp/7::float) as numeric),2) as gestation, t1.gestationweekbylmp,t1.gestationdaysbylmp,t1.baby_type,t1.central_temp,t1.dischargeddate,t2.apgar_onemin,t2.apgar_fivemin,t2.apgar_tenmin,t3.motherage,t4.conception_type,t4.mode_of_delivery,t5.steroidname,t5.numberofdose,t1.dischargestatus  FROM "+schemaName+".baby_detail AS t1 LEFT JOIN "+schemaName+".birth_to_nicu AS t2 ON t1.uhid=t2.uhid LEFT JOIN "+schemaName+".parent_detail AS t3 ON t1.uhid=t3.uhid LEFT JOIN "+schemaName+".antenatal_history_detail AS t4 ON t1.uhid=t4.uhid LEFT JOIN "+schemaName+".antenatal_steroid_detail AS t5 ON t1.uhid=t5.uhid where t1.timeofadmission is not null and t1.timeofbirth is not null and t1.uhid = '"+patientCaseUHID+"';"
    elif typeQuery == "filtering":
        query = "SELECT DISTINCT(uhid) FROM "+schemaName+".baby_detail WHERE dateofadmission >= '2018-07-01' AND dateofadmission <= '2021-06-25' and UHID IN  (select distinct(uhid) from "+schemaName+".babyfeed_detail where uhid in  ( select distinct(uhid) from "+schemaName+".baby_visit where uhid in (select  distinct(uhid) from "+schemaName+".nursing_vitalparameters where uhid in ( select distinct(uhid) from "+schemaName+".device_monitor_detail UNION select distinct(uhid) from "+schemaName+".device_monitor_detail_dump))))  and isreadmitted is not true and gestationweekbylmp is not null and birthweight is not null and uhid = '"+patientCaseUHID+"';"

    elif typeQuery == "antenatal":
        query = "select distinct(b.uhid),l.conception_type, b.gender, b.dateofadmission, b.dischargestatus,b.birthweight,b.weight_galevel,b.weight_centile,b.birthlength,b.birthheadcircumference, b.inout_patient_status, b.gestationweekbylmp, b.gestationdaysbylmp,round( CAST((b.gestationweekbylmp + b.gestationdaysbylmp/7::float) as numeric),2) as Gestation,b.dischargeddate, b.admissionweight, b.baby_type,b.baby_number, b.branchname ,DATE_PART('day',b.dischargeddate - b.dateofadmission) as LOS,c.apgar_onemin, c.apgar_fivemin,c.apgar_tenmin, c.resuscitation,d.isantenatalsteroidgiven, d.mode_of_delivery,z.motherage,e.jaundicestatus as JAUNDICE,  f.eventstatus as SEPSIS,f.progressnotes, g.eventstatus as RDS, g.progressnotes,y.eventstatus as ASPHYXIA,y.progressnotes from "+schemaName+".baby_detail as b left join "+schemaName+".birth_to_nicu as c on b.uhid = c.uhid and b.episodeid = c.episodeid left join "+schemaName+".parent_detail as z on b.uhid = z.uhid and b.episodeid = z.episodeid left join "+schemaName+".antenatal_history_detail as d on b.uhid = d.uhid and b.episodeid = d.episodeid left join "+schemaName+".sa_jaundice AS e ON b.uhid = e.uhid and e.jaundicestatus = 'Yes' and e.phototherapyvalue='Start' left join "+schemaName+".sa_infection_sepsis AS f ON b.uhid = f.uhid and f.eventstatus = 'yes' and f.episode_number = 1 left join "+schemaName+".sa_cns_asphyxia AS y ON b.uhid = y.uhid and y.eventstatus = 'yes' and y.episode_number = 1 left join "+schemaName+".antenatal_history_detail as l on b.uhid=l.uhid left join "+schemaName+".sa_resp_rds AS g ON b.uhid = g.uhid and g.eventstatus = 'Yes' and g.episode_number = 1 and g.uhid IN (select distinct(h.uhid) from "+schemaName+".respsupport AS h where h.eventname='Respiratory Distress' and (h.rs_vent_type ='Mechanical Ventilation' OR h.rs_vent_type ='HFO') UNION select distinct(i.uhid) from "+schemaName+".sa_resp_rds AS i where i.sufactantname is not null) and b.uhid = '"+patientCaseUHID+ "' order by b.dateofadmission"
    
    elif typeQuery == "output":
        query = "SELECT t1.uhid,t1.abdomen_girth,t1.urine,t1.stool,t1.stool_passed, t1.entry_timestamp FROM "+schemaName+".nursing_intake_output AS t1 where t1.uhid = '"+patientCaseUHID+"'";

    elif typeQuery == "bloodgas":
        query = "SELECT uhid,creationtime,modificationtime,ph,entrydate FROM "+schemaName+".nursing_bloodgas where uhid ='"+ patientCaseUHID+"'"

    elif typeQuery == "vitals":
        query = "SELECT t1.uhid,t1.entrydate,t1.rbs,t1.skintemp,t1.centraltemp  FROM "+schemaName+".nursing_vitalparameters AS t1 where t1.uhid = '"+patientCaseUHID+"'"

    elif typeQuery == "anthropometry":
        query = "SELECT t1.uhid,t1.visitdate,t1.visittime,t1.currentdateweight, t1.currentdateheight  FROM "+schemaName+".baby_visit AS t1 where t1.uhid = '"+patientCaseUHID+"'"
    
    elif typeQuery == "medication":
        query = "SELECT t1.uhid,t1.startdate,t1.medicineorderdate,t1.medicinename,t1.medicationtype,t2.typevalue FROM "+schemaName+".baby_prescription AS t1 LEFT JOIN "+schemaName+".ref_medtype AS t2 ON t1.medicationtype = t2.typeid WHERE (t1.medicationtype = 'TYPE0001' OR t1.medicationtype = 'TYPE0009' OR t1.medicationtype = 'TYPE0004') and (t1.uhid ='"+patientCaseUHID+"')"

    elif typeQuery == "nutrition":
        query = "SELECT t1.uhid,t1.entrydatetime,t1.totalparenteralvolume,t1.total_intake FROM "+schemaName+".babyfeed_detail AS t1 where t1.uhid = '"+patientCaseUHID+"'"

    elif typeQuery == "monitor":
        query = "SELECT t11.uhid,t11.starttime,t11.pulserate, t11.ecg_resprate, t11.spo2, t11.heartrate, t21.dischargestatus,t11.mean_bp,t11.sys_bp,t11.dia_bp  FROM "+schemaName+".device_monitor_detail AS t11 LEFT JOIN "+schemaName+".baby_detail AS t21 ON t11.uhid=t21.uhid WHERE (t21.dischargestatus = '"+caseType+"' AND t21.dateofadmission > '2018-07-01' and t21.uhid = '"+ patientCaseUHID+"' AND t11.starttime < t21.dischargeddate)  UNION" + " SELECT t12.uhid,t12.starttime,t12.pulserate, t12.ecg_resprate, t12.spo2, t12.heartrate, t22.dischargestatus,t12.mean_bp,t12.sys_bp,t12.dia_bp  FROM "+schemaName+".device_monitor_detail_dump AS t12 LEFT JOIN "+schemaName+".baby_detail AS t22 ON t12.uhid=t22.uhid WHERE (t22.dischargestatus = '"+caseType+"' AND t22.dateofadmission > '2018-07-01' and t22.uhid = '"+ patientCaseUHID+"' AND t12.starttime < t22.dischargeddate);"

    elif typeQuery == "ventilator":
        query = "SELECT t1.uhid,t1.start_time,t1.creationtime,t1.peep, t1.pip,t1.map ,t1.tidalvol, t1.minvol,t1.ti,t1.fio2 FROM "+schemaName+".device_ventilator_detail_dump AS t1 LEFT JOIN "+schemaName+".baby_detail AS t2 ON t1.uhid=t2.uhid WHERE (t2.dischargestatus = '"+caseType+"' OR t2.dischargestatus = '"+caseType+"') and t2.uhid = '"+patientCaseUHID+"';"

    cur.execute(query)
    cols = list(map(lambda x: x[0], cur.description))
    result = pd.DataFrame(cur.fetchall(),columns=cols)
    return result;

try:      
    con = psycopg2.connect (user = 'postgres',
                password = 'postgres',
                port = '5432',
                host = 'localhost',                
                database = 'inicudb')
    patientCaseUHID = 'RSHI.0000026579'
    caseType = "Discharge"
    cur  = con.cursor()
    path = os.getcwd()
    conditionCase = ""
    folderName = "/Discharge_Cases/"

    #caseType = 'Discharge'
    seperator = '_'
    #patientCaseUHID = ["RNEH.0000008375", "RNEH.0000011301", "RNEH.0000012581", "RNEH.0000013713", "RSHI.0000012088", "RSHI.0000013287", "RSHI.0000014720"
    #, "RSHI.0000015178", "RSHI.0000015211", "RSHI.0000015691", "RSHI.0000016373", "RSHI.0000017471", "RSHI.0000017472", "RSHI.0000019707"
    #, "RSHI.0000021953", "RSHI.0000023451"]
    schemaName = 'zz_apollo'
    #patientCaseUHID = ["RNEH.0000012581"]
    filePath = path+folderName+patientCaseUHID+"/"
    fileName = filePath+caseType+seperator+patientCaseUHID+seperator+'intermediate_checkpoint_new_5.csv'
    print ('data preparation started for',fileName)    

    #Fixed Parameters
    fixed = fetchingFromDatabase("fixed",schemaName,patientCaseUHID,cur,conditionCase,caseType)
    s1_fixed = set(fixed.uhid.unique())
    #Filtering Data
    uhids = fetchingFromDatabase("filtering",schemaName,patientCaseUHID,cur,conditionCase,caseType)
    s2_uhids = set(uhids.uhid.unique())

    #Removing Duplicates
    l = list(s1_fixed - s1_fixed.intersection(s2_uhids))
    fixed.set_index('uhid',inplace=True)
    fixed = fixed.drop(l)

    fixed.reset_index(inplace=True)
    fixed_final = fixed.copy()
    #Calculating TOA and TOB in seconds
    fixed_final['add_seconds'] = fixed_final['timeofbirth'].apply(second_addition)
    fixed_final['add_seconds_admission'] = fixed_final['timeofadmission'].apply(second_addition)

    #Combining Date and Time(Admission and Birth)
    print(fixed_final)
    fixed_final['actual_DOB'] = fixed_final.apply(lambda x: actual_birthdate(x['dateofbirth'], x['add_seconds']), axis=1)
    fixed_final['actual_DOA'] = fixed_final.apply(lambda x: actual_birthdate(x['dateofadmission'], x['add_seconds_admission']), axis=1)
    fixed_final = fixed_final[['actual_DOB','uhid','dischargeddate','actual_DOA','gender','birthweight','birthlength','birthheadcircumference','inout_patient_status'
    ,'gestation','gestationweekbylmp','gestationdaysbylmp','baby_type','central_temp','apgar_onemin',
    'apgar_fivemin','apgar_tenmin','motherage','conception_type','mode_of_delivery','steroidname',
    'numberofdose','dischargestatus','birthlength','birthheadcircumference','inout_patient_status'
    ,'gestationweekbylmp','gestationdaysbylmp','baby_type','central_temp','apgar_onemin',
    'apgar_fivemin','apgar_tenmin','motherage','conception_type','mode_of_delivery','steroidname',
    'numberofdose']]

    fixed_final.dropna(subset=['dischargeddate'],inplace=True)
    fixed_final['los'] = fixed_final['dischargeddate'] - fixed_final['actual_DOA']
    fixed_final['day_1'] = fixed_final['los'].apply(day)
    fixed_final = fixed_final[fixed_final['day_1']>=0]
    fixed_final.sort_values(by = ['actual_DOA'],inplace=True)
    fixed_final.drop_duplicates(subset=['uhid'],keep='first',inplace=True)
    print('Total number of columns in  data prepare 1='+str(len(fixed_final.columns)))

    #Repeat data and add hour series column representing minute wise data
    fixed_hdp = pd.DataFrame()
    for i in fixed_final.uhid.unique():
        x = fixed_final[fixed_final['uhid']==i]
        time_s = []
        #n = math.ceil((x.dischargeddate.iloc[0] - x.actual_DOA.iloc[0]).total_seconds()/60)
        n = math.ceil((x.dischargeddate.iloc[0] - x.actual_DOA.iloc[0]).total_seconds())
        print(n)
        try:
            print(x)
            x = pd.concat([x]*int(n))
            print(len(x))

            for inner in range(0,len(x)):
                #print(i)
                #time_s.append(x['actual_DOA'].iloc[inner] + timedelta(seconds = inner*60))
                time_s.append(x['actual_DOA'].iloc[inner] + timedelta(seconds = inner))
                #t['hour_series'].loc[i] = t['actual_DOA'].iloc[i] + timedelta(hours = i)

            x['hour_series'] = time_s
        except Exception as e:
            print(i,n, e)
            continue
        fixed_hdp = fixed_hdp.append(x,ignore_index=True)

    fixed_hdp.drop_duplicates(subset=['uhid','hour_series'],inplace=True)
    print('Total number of columns in  data prepare hour_series added 2='+str(len(fixed_hdp.columns)))



    antenatal = fetchingFromDatabase("antenatal",schemaName,patientCaseUHID,cur,conditionCase,caseType)

    data = antenatal.copy()
    ids = fixed_hdp.uhid.unique()
    data.drop_duplicates('uhid',keep='first',inplace=True)
    df = pd.DataFrame(columns=data.columns)
    for i in ids:
        x = data[data['uhid']==i]
        df = df.append(x,ignore_index=True)
    pd.set_option('display.max_columns',100)
    antenatal = pd.DataFrame()
    antenatal = fixed_hdp.copy()
    uhid = antenatal.uhid.unique()
    fixed_hdp_final = pd.DataFrame()
    for i in uhid:
        x = fixed_hdp[fixed_hdp['uhid']==i]
        time_s = []
        #n = math.ceil((x.dischargeddate.iloc[0] - x.actual_DOA.iloc[0]).total_seconds()/60)
        n = math.ceil((x.dischargeddate.iloc[0] - x.actual_DOA.iloc[0]).total_seconds())
        try:
            x = pd.concat([x]*int(n))
            for inner in range(0,len(x)):
                #time_s.append(x['actual_DOA'].iloc[inner] + timedelta(seconds = inner*60))
                time_s.append(x['actual_DOA'].iloc[inner] + timedelta(seconds = inner))
                #t['hour_series'].loc[i] = t['actual_DOA'].iloc[i] + timedelta(hours = i)

            x['hour_series'] = time_s
        except Exception as inst:
            print(inst)
            continue
        fixed_hdp_final = fixed_hdp_final.append(x,ignore_index=True)

    fixed_hdp_final['hour_series'].apply(split_hour)

    fixed_hdp_final['hour'] = fixed_hdp_final['hour_series'].apply(split_hour)
    fixed_hdp_final['day'] = fixed_hdp_final['hour_series'].apply(split_date_1)
    print ('data preparation intake output started')    
    print('Total number of columns in  data prepare hour_series added 3='+str(len(fixed_hdp_final.columns)))        
    
    #Intake Output
    output = fetchingFromDatabase("output",schemaName,patientCaseUHID,cur,conditionCase,caseType)
    output_d = pd.DataFrame()

    for i in uhid:
        x = output[output['uhid']==i]
        output_d = output_d.append(x,ignore_index=True)

    output_d.entry_timestamp = output_d.entry_timestamp + timedelta(seconds=19800)
    output_d['hour'] = output_d.entry_timestamp.apply(split_hour)
    output_d['day'] = output_d.entry_timestamp.apply(split_date_1)

    #Merging Intake-Output and Fixed/Antenatal
    output_d.sort_values(by=['uhid','entry_timestamp'],inplace=True)
    output_d = output_d.drop_duplicates(subset=['uhid','hour'],keep='last')
    fixed_hdp_output_final = pd.merge(fixed_hdp_final,output_d,on=['uhid','hour'],how='left')

    print('Total number of columns in  data prepare hour_series added 4='+str(len(fixed_hdp_output_final.columns)))        

    fixed_hdp_output_final.sort_values('hour_series')
    fixed_hdp_output_final_urine = pd.DataFrame()

    for i in uhid:
        x = fixed_hdp_output_final[fixed_hdp_output_final['uhid']==i]
        x['time_divide'] = 0
        #print(len(x))
        startTime = None
        for j in range(len(x)):
            try:
                if j == 0:
                    x.loc[j,'time_divide'] = 0.0                
                    startTime = x.actual_DOA.iloc[j]
                else:
                    if(x.urine.iloc[j] != None):
                        urine_value = float(x.urine.iloc[j])
                        if(~math.isnan(urine_value) and (urine_value >0)):
                            x.loc[j,'time_divide'] = (pd.to_datetime(str(x.entry_timestamp.iloc[j]).split("+")[0]) - startTime).total_seconds()
                            startTime = pd.to_datetime(str(x.entry_timestamp.iloc[j]).split("+")[0])
                #print(x.loc[j,'time_divide'])
            except Exception as e:
                print(j,'Exception is',e)
                continue
        fixed_hdp_output_final_urine = fixed_hdp_output_final_urine.append(x,ignore_index=True)  
    fixed_hdp_output_final_urine = fixed_hdp_output_final_urine.sort_values('hour_series')
    fixed_hdp_output_final_urine['series'] = fixed_hdp_output_final_urine.hour_series
    fixed_hdp_output_final_urine['urine'] = fixed_hdp_output_final_urine['urine'].apply(to_float)
    fixed_hdp_output_final_urine['urine'] = fixed_hdp_output_final_urine['urine'].apply(urine_check)
    fixed_hdp_output_final_urine['urine_per_hour'] = fixed_hdp_output_final_urine['urine']/((fixed_hdp_output_final_urine['time_divide'])/(3600))
    print('Total number of columns in  data prepare hour_series added 5='+str(len(fixed_hdp_output_final_urine.columns)))        
    fixed_hdp_output_final_urine = fixed_hdp_output_final_urine.sort_values(by = ['uhid','hour_series'])
    fixed_hdp_output_final_urine_hour = pd.DataFrame()
    for i in fixed_hdp_output_final_urine.uhid.unique():
        x = fixed_hdp_output_final_urine[fixed_hdp_output_final_urine['uhid']==i]
        x['urine_per_hour'].fillna(method='bfill',inplace=True)
        fixed_hdp_output_final_urine_hour = fixed_hdp_output_final_urine_hour.append(x,ignore_index=True)
    
    #Blood Gas
    ph = fetchingFromDatabase("bloodgas",schemaName,patientCaseUHID,cur,conditionCase,caseType)

    ph['new_ph'] = ph.ph.apply(ph_func)
    ph.dropna(subset=['new_ph'],inplace=True)
    
    ph_hdp = pd.DataFrame()
    for i in ph.uhid.unique():
        x = ph[ph['uhid']==i]
        ph_hdp = ph_hdp.append(x,ignore_index=True)
    if len(ph_hdp.columns)>0:
        ph_hdp['hour'] = ph_hdp.entrydate.apply(split_hour)
        ph_hdp.drop_duplicates(subset=['uhid','hour'],keep='last',inplace=True)
    else:
        x = pd.DataFrame(data=[patientCaseUHID],columns={'uhid'})
        print(x)
        ph_hdp_1 = pd.DataFrame(columns=ph.columns)
        ph_hdp_1 = ph_hdp_1.append(x,ignore_index=True)
    fixed_hdp_output_final_urine_hour['uhid'] = fixed_hdp_output_final_urine_hour['uhid'].astype(str)

    if len(ph_hdp.columns)>0:
        fixed_hdp_output_final_urine_ph = pd.merge(fixed_hdp_output_final_urine_hour,ph_hdp,on=['uhid','hour'],how='left')
        fixed_hdp_output_final_urine_ph = fixed_hdp_output_final_urine_ph.drop_duplicates(subset=['uhid','hour_series'],keep='first')
    else:
        fixed_hdp_output_final_urine_ph = pd.merge(fixed_hdp_output_final_urine_hour,ph_hdp_1,on=['uhid'],how='left')
        fixed_hdp_output_final_urine_ph = fixed_hdp_output_final_urine_ph.drop_duplicates(subset=['uhid','hour_series'],keep='first')

    print('Total number of columns in  data prepare hour_series added 6='+str(len(fixed_hdp_output_final_urine_ph.columns)))        

    #Vitals
    print ('data preparation vitals - continuous data started')    
    vitals = fetchingFromDatabase("vitals",schemaName,patientCaseUHID,cur,conditionCase,caseType)

    vitals.drop_duplicates(subset=['uhid','entrydate'],keep='first',inplace=True)
    vitals['hour'] = vitals.entrydate.apply(split_hour)
    fixed_hdp_output_final_urine_ph.uhid = fixed_hdp_output_final_urine_ph.uhid.astype(str)
    fixed_hdp_output_final_urine_ph_vitals = pd.merge(fixed_hdp_output_final_urine_ph,vitals,on = ['uhid','hour'], how='left')
    fixed_hdp_output_final_urine_ph_vitals.drop_duplicates(subset=['uhid','hour_series'],keep='first',inplace=True)
    fixed_hdp_output_final_urine_ph_vitals['temp'] = fixed_hdp_output_final_urine_ph_vitals.apply(lambda x: temp_a(x['centraltemp'], x['skintemp']), axis=1)

    print('Total number of columns in  data prepare skin temperature added 7='+str(len(fixed_hdp_output_final_urine_ph_vitals.columns)))        

    #Anthropometry
    anthropometry = fetchingFromDatabase("anthropometry",schemaName,patientCaseUHID,cur,conditionCase,caseType)

    anthropometry.sort_values('visitdate',inplace=True)
    anthropometry['hour'] = anthropometry.apply(lambda x: stamp_1(x['visitdate'], x['visittime']), axis=1)
    anthropometry['day'] = anthropometry['visitdate'].apply(to_str)
    anthropometry.drop_duplicates(subset=['uhid','visitdate'],keep='last',inplace=True)
    fixed_hdp_output_final_urine_ph_vitals['day'] = fixed_hdp_output_final_urine_ph_vitals['hour_series'].apply(split_date_1)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry = pd.merge(fixed_hdp_output_final_urine_ph_vitals,anthropometry,on = ['uhid','day'], how='left')

    print('Total number of columns in  data prepare Anthropometry added 7='+str(len(fixed_hdp_output_final_urine_ph_vitals_anthropometry.columns)))        

    #Medications
    meds = fetchingFromDatabase("medication",schemaName,patientCaseUHID,cur,conditionCase,caseType)

    meds_dummies = pd.get_dummies(meds,columns=['typevalue'])
    meds_dummies['hour'] = meds_dummies.startdate.apply(split_hour)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry.drop(['hour_x','hour_y'],axis=1,inplace=True)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry['hour'] = fixed_hdp_output_final_urine_ph_vitals_anthropometry.hour_series.apply(split_hour)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication = pd.merge(fixed_hdp_output_final_urine_ph_vitals_anthropometry,meds_dummies,on = ['uhid','hour'], how='left')
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication['day'] = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication['hour_series'].apply(split_date_1)

    print('Total number of columns in  data prepare Medications added 7='+str(len(fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication.columns)))        

    #Nutrition
    pn = fetchingFromDatabase("nutrition",schemaName,patientCaseUHID,cur,conditionCase,caseType)

    pn['hour'] = pn.entrydatetime.apply(split_hour)
    pn['total_intake'] = pn['total_intake'].astype(float)
    pn['tpn-tfl'] = pn['totalparenteralvolume']/pn['total_intake']
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition = pd.merge(fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication,pn,on = ['uhid','hour'], how='left')
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.drop_duplicates(subset=['uhid','hour_series'],inplace=True)

    print('Total number of columns in  data prepare nutrition added 8='+str(len(fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.columns)))        
    
    #filling NA for abdomen girth, height and nutrition
    q_dummy = pd.DataFrame()
    for i in fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.uhid.unique():
        x = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition[fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['uhid']==i]
        x['currentdateheight'].fillna(method='ffill',inplace=True)
        x['currentdateweight'].fillna(method='ffill',inplace=True)
        q_dummy = q_dummy.append(x,ignore_index=True)
    q_dummy['abdomen_girth'] = pd.to_numeric(q_dummy['abdomen_girth'], errors='coerce')
    q_dummy['abdomen_girth'] = q_dummy['abdomen_girth'].apply(abd)

    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition = pd.DataFrame()
    print('Total number of columns in  data prepare q added 9='+str(len(fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.columns)))        
    
    for i in q_dummy.uhid.unique():
        x = q_dummy[q_dummy['uhid']==i]
        #for j in y['day'].unique():
        #x = y[y['day']==j]
        x['abdomen_girth'].fillna(method='bfill',inplace=True)
        x['currentdateweight'].fillna(method='bfill',inplace=True)
        x['currentdateheight'].fillna(method='bfill',inplace=True)
        x['totalparenteralvolume'].fillna(method='ffill',inplace=True)
        x['total_intake'].fillna(method='ffill',inplace=True)
        x['tpn-tfl'].fillna(method='ffill',inplace=True)
        fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.append(x,ignore_index=True)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.replace([np.inf, -np.inf], np.nan)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['abdomen_girth'] = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['abdomen_girth'].apply(abd_2)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['abdomen_girth'] = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['abdomen_girth'].apply(abd_3)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['urine_per_hour'] = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['urine_per_hour'].apply(upm)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['temp'] = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['temp'].apply(temp_f_c)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.currentdate = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.currentdateweight.apply(weight_correct)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.currentdate = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.currentdateweight.apply(weight_correct_2)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['currentdateheight'] = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['currentdateheight'].apply(height_correct)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['rbs'] = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['rbs'].apply(rbs_correct)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['day'] = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['hour_series'].apply(split_date_1)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['hour_series'] = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['hour_series'].apply(to_date)

    print('Total number of columns in  data prepare rbs added 10='+str(len(fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.columns)))        
    
    #Calculating Stool Total
    q_dummy = pd.DataFrame()
    for i in fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.uhid.unique():
        x = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition[fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['uhid']==i]
        x.rbs.fillna(method='ffill',inplace=True)
        x.currentdateweight.fillna(method='ffill',inplace=True)
        x.currentdateheight.fillna(method='ffill',inplace=True)
        n = math.ceil(len(x)/24)+1
        start_date = pd.to_datetime(x['day'].iloc[0]+" " + "08:00:00") - timedelta(hours=24)
        for i in range(int(n)):          
            y = x[(x['hour_series']>=start_date + timedelta(hours=24*i)) & (x['hour_series']<=start_date + timedelta(hours=24*(i+1)))]
            y['stool_day_total'] = (y['stool_passed'].sum())/60
            q_dummy =q_dummy.append(y,ignore_index=True)

    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition = q_dummy.copy()
    q_dummy = pd.DataFrame()
    for i in fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.uhid.unique():
        x = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition[fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['uhid']==i]
        x.abdomen_girth.fillna(method='ffill',inplace=True)
        q_dummy = q_dummy.append(x, ignore_index=True)

    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition = pd.DataFrame()
    for i in q_dummy.uhid.unique():
        x = q_dummy[q_dummy['uhid']==i]
        x['abd_difference_y'] = x['abdomen_girth'] - x['abdomen_girth'].shift(periods=1)
        fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.append(x,ignore_index=True)

    #Device Monitor
    monitor = fetchingFromDatabase("monitor",schemaName,patientCaseUHID,cur,conditionCase,caseType)
    monitor.sort_values('starttime',inplace=True)
    test = monitor.drop_duplicates(subset=['uhid','starttime'],keep='first')
    print('Total number of columns in test ='+str(len(fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.columns)))             

    #Device Ventilator
    ventilator_cont = fetchingFromDatabase("ventilator",schemaName,patientCaseUHID,cur,conditionCase,caseType)

    ventilator_cont = ventilator_cont.drop_duplicates(subset=['uhid','start_time'],keep='first')
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['date'] = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.starttime.apply(split_date)
    ventilator_cont['date'] = ventilator_cont.start_time.apply(split_date)
    cont_data = pd.merge(fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition,ventilator_cont,on=['uhid','date'],how='left',copy=False)

    print('Total number of columns in cont_data ='+str(len(cont_data.columns)))  

    
    
                
    
    test_cont = cont_data.drop_duplicates(subset=['uhid','starttime','heartrate'],keep='first')
    test_cont['hour_series'] = test_cont['date'].apply(split_hour)
    test_cont['ref_hour'] = test_cont['hour_series'].apply(to_str)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['ref_hour'] = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['hour_series'].apply(to_str)
    test_cont['cont_time'] = test_cont.starttime.apply(con_time)
    fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition['cont_time'] = fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.ref_hour.apply(con_time_2)

    print('Total number of columns in final hour_series added ='+str(len(fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition.columns))+'in test_cont added ='+str(len(test_cont.columns)))    

    final_hdp = pd.merge(fixed_hdp_output_final_urine_ph_vitals_anthropometry_medication_nutrition,test_cont, on=['uhid','cont_time'],how='left')
    final_hdp['gender'] = final_hdp['gender'].apply(gender)
    final_hdp['inout_patient_status'] = final_hdp['inout_patient_status'].apply(in_out)
    final_hdp['baby_type'] = final_hdp['baby_type'].apply(baby_type)
    final_hdp['conception_type'] = final_hdp['conception_type'].apply(conception)
    final_hdp['mode_of_delivery'] = final_hdp['mode_of_delivery'].apply(mod)
    final_hdp['steroidname'] = final_hdp['steroidname'].apply(steroid)


    

    # Next is data fill step
    print ('data preparation forward filling started')    
    final_hdp['pulserate'].fillna(method='ffill',limit=5)
    final_hdp['heartrate'].fillna(method='ffill',limit=5)
    final_hdp['ecg_resprate'].fillna(method='ffill',limit=5)
    final_hdp['spo2'].fillna(method='ffill',limit=5)
    final_hdp['mean_bp'].fillna(method='ffill')
    final_hdp['sys_bp'].fillna(method='ffill')
    final_hdp['dia_bp'].fillna(method='ffill')
    final_hdp['mean_bp'].fillna(method='ffill')
    final_hdp['peep'].fillna(method='ffill')
    final_hdp['pip'].fillna(method='ffill')
    final_hdp['map'].fillna(method='ffill')
    final_hdp['tidalvol'].fillna(method='ffill')
    final_hdp['minvol'].fillna(method='ffill')
    final_hdp['ti'].fillna(method='ffill')
    final_hdp['fio2'].fillna(method='ffill')
    final_hdp['abdomen_girth'].fillna(method='ffill')
    final_hdp['urine'].fillna(method='ffill')
    final_hdp['stool_day_total'].fillna(method='ffill')
    final_hdp['new_ph'].fillna(method='ffill')
    final_hdp['rbs'].fillna(method='ffill')
    final_hdp['skintemp'].fillna(method='ffill')
    final_hdp['centraltemp'].fillna(method='ffill')
    final_hdp['temp'].fillna(method='ffill')
    final_hdp['currentdateweight'].fillna(method='ffill')
    final_hdp['currentdateheight'].fillna(method='ffill')
    final_hdp['totalparenteralvolume'].fillna(method='ffill')
    final_hdp['total_intake'].fillna(method='ffill')
    final_hdp['tpn-tfl'].fillna(method='ffill')
    if caseType == "Death":
        final_hdp['dischargestatus'] = 1
    elif  caseType == "Discharge":
        final_hdp['dischargestatus'] = 0     
    if not os.path.exists(filePath):
        os.makedirs(filePath)


    uhid = final_hdp.uhid.unique()
    apnea_data = pd.DataFrame()

    for i in uhid:
        x = final_hdp[final_hdp['uhid']==i]
        durationApnea = 0;
        lastEventTimeApnea = None;
        minHeartRateApnea = 999;
        minPulseRateApnea = 999;
        minSpo2Apnea = 999;
        apneaGot = False;
        firstApnea = 0;



        for j in range(len(x)):
            try:
                if((x.spo2.iloc[j] != None and int(x.spo2.iloc[j]) < 85) and ((x.heartrate.iloc[j] != None and int(x.heartrate.iloc[j]) < 100) or (x.pulserate.iloc[j] != None and int(x.pulserate.iloc[j]) < 100))):
                    #Apnea Code
                    if(firstApnea == 0):
                        if(x.heartrate.iloc[j] != None):
                            minHeartRateApnea = int(x.heartrate.iloc[j]);
                        if(x.pulserate.iloc[j] != None):
                            minPulseRateApnea = int(x.pulserate.iloc[j]);
                        if(x.spo2.iloc[j] != None):
                            minSpo2Apnea = int(x.spo2.iloc[j]);

                        firstApnea = 1;
                        lastEventTimeApnea = x.starttime.iloc[j];
                        x.loc[j,'apnea'] = 1

                        

                    elif(((x.starttime.iloc[j] - lastEventTimeApnea).total_seconds() <= 2)):
                        if(x.heartrate.iloc[j] != None and minHeartRateApnea > int(x.heartrate.iloc[j])):
                            minHeartRateApnea = int(x.heartrate.iloc[j]);
                        

                        if(x.pulserate.iloc[j] != None and minPulseRateApnea > int(x.pulserate.iloc[j])):
                            minPulseRateApnea = int(x.pulserate.iloc[j]);
                        

                        if(x.spo2.iloc[j] != None and minSpo2Apnea > int(x.spo2.iloc[j])):
                            minSpo2Apnea = int(x.spo2.iloc[j]);
                        

                        diffDuration = (x.starttime.iloc[j] - lastEventTimeApnea).total_seconds();
                        durationApnea = durationApnea + diffDuration;
                        lastEventTimeApnea = x.starttime.iloc[j];
                        apneaGot = True;
                        x.loc[j,'apnea'] = 1


                    elif(apneaGot == True and durationApnea >= 15): 
                        if(durationApnea == 0):
                            durationApnea = 1
                    
                        apneaGot = False;
                        lastEventTimeApnea = x.starttime.iloc[j];
                        durationApnea = 0;
                        firstApnea = 0;
                        minHeartRateApnea = 999;
                        minPulseRateApnea = 999;
                        minSpo2Apnea = 999;

                        if(x.heartrate.iloc[j] != None):
                            minHeartRateApnea = int(x.heartrate.iloc[j]);
                        if(x.pulserate.iloc[j] != None):
                            minPulseRateApnea = int(x.pulserate.iloc[j]);
                        if(x.spo2.iloc[j] != None):
                            minSpo2Apnea = int(x.spo2.iloc[j]);

                        firstApnea = 1;
                        lastEventTimeApnea = x.starttime.iloc[j];
                        x.loc[j,'apnea'] = 0                        
                    else:
                        x.loc[j,'apnea'] = 1
                        if(x.heartrate.iloc[j] != None and minHeartRateApnea > int(x.heartrate.iloc[j])):
                            minHeartRateApnea = int(x.heartrate.iloc[j]);
                        

                        if(x.pulserate.iloc[j] != None and minHeartRateApnea > int(x.pulserate.iloc[j])):
                            minPulseRateApnea = int(x.pulserate.iloc[j]);
                        

                        if(x.spo2.iloc[j] != None and minHeartRateApnea > int(x.spo2.iloc[j])):
                            minSpo2Apnea = int(x.spo2.iloc[j]);
                        

                        apneaGot = False;
                        lastEventTimeApnea = x.starttime.iloc[j];
                        durationApnea = 0;


                else:
                    if(apneaGot == True and durationApnea >= 15):
                        
                        if(durationApnea == 0):
                            durationApnea = 1
                        
                        x.loc[j,'apnea'] = 0
                        apneaGot = False;
                        lastEventTimeApnea = x.starttime.iloc[j];
                        durationApnea = 0;
                        firstApnea = 0;
                        minHeartRateApnea = 999;
                        minPulseRateApnea = 999;
                        minSpo2Apnea = 999;
                    elif(apneaGot == True and durationApnea < 15):
                        apneaGot = False;
                        durationApnea = 0;
                        firstApnea = 0;
                        minHeartRateApnea = 999;
                        minPulseRateApnea = 999;
                        minSpo2Apnea = 999;
                        x.loc[j,'apnea'] = 0
                    else:
                        x.loc[j,'apnea'] = 0
                
            except Exception as e:
                print(j,'Exception is',e)
        apnea_data = apnea_data.append(x,ignore_index=True)


    print('Total number of columns in final final_hdp  ='+str(len(apnea_data.columns)))             
    apnea_data.to_csv(fileName)
except Exception as e:
    print('Exception in data preparation', e)
    PrintException()
    