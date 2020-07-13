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
def read_prepare_data(patientCaseUHID,caseType,conditionCase,folderName):
    try:    
        path = os.getcwd()
        seperator = '_'
        filePath = path+folderName+patientCaseUHID+"/"
        fileName = filePath+caseType+seperator+patientCaseUHID+seperator+'intermediate_checkpoint_new_5.csv'
        preparedData = pd.read_csv(fileName)
        return fileName, preparedData
    except Exception as e:
        print('Exception in data preparation', e)
        PrintException()
        return None
def prepare_data(con,patientCaseUHID,caseType,conditionCase,folderName):
    try:      
        cur  = con.cursor()
        path = os.getcwd()
        #caseType = 'Discharge'
        seperator = '_'
        #patientCaseUHID = ["RNEH.0000008375", "RNEH.0000011301", "RNEH.0000012581", "RNEH.0000013713", "RSHI.0000012088", "RSHI.0000013287", "RSHI.0000014720"
        #, "RSHI.0000015178", "RSHI.0000015211", "RSHI.0000015691", "RSHI.0000016373", "RSHI.0000017471", "RSHI.0000017472", "RSHI.0000019707"
        #, "RSHI.0000021953", "RSHI.0000023451"]
        schemaName = 'apollo'
        #patientCaseUHID = ["RNEH.0000012581"]
        filePath = path+folderName+patientCaseUHID+"/"
        fileName = filePath+caseType+seperator+patientCaseUHID+seperator+'intermediate_checkpoint_new_5.csv'
        print ('data preparation started for',fileName)    
        #Fixed Parameters
        cur10 = con.cursor()


        queryStr = "SELECT t1.uhid,t1.dateofbirth,t1.timeofbirth,t1.timeofadmission,t1.dateofadmission,t1.gender,t1.birthweight,t1.birthlength,t1.birthheadcircumference,t1.inout_patient_status,round( CAST((t1.gestationweekbylmp + t1.gestationdaysbylmp/7::float) as numeric),2) as gestation, t1.gestationweekbylmp,t1.gestationdaysbylmp,t1.baby_type,t1.central_temp,t1.dischargeddate,t2.apgar_onemin,t2.apgar_fivemin,t2.apgar_tenmin,t3.motherage,t4.conception_type,t4.mode_of_delivery,t5.steroidname,t5.numberofdose,t1.dischargestatus  FROM "+schemaName+".baby_detail AS t1 LEFT JOIN "+schemaName+".birth_to_nicu AS t2 ON t1.uhid=t2.uhid LEFT JOIN "+schemaName+".parent_detail AS t3 ON t1.uhid=t3.uhid LEFT JOIN "+schemaName+".antenatal_history_detail AS t4 ON t1.uhid=t4.uhid LEFT JOIN "+schemaName+".antenatal_steroid_detail AS t5 ON t1.uhid=t5.uhid where t1.timeofadmission is not null and t1.timeofbirth is not null and t1.uhid = '"+patientCaseUHID+"';"
        cur10.execute(queryStr)
        #print(queryStr)
        cols10 = list(map(lambda x: x[0], cur10.description))
        fixed = pd.DataFrame(cur10.fetchall(),columns=cols10)

        s1 = set(fixed.uhid.unique())
        #print('Distinct UHID 1=',len(s1))

        cur11 = con.cursor()
        queryStr = "SELECT DISTINCT(uhid) FROM "+schemaName+".baby_detail WHERE dateofadmission >= '2018-07-01' AND dateofadmission <= '2020-06-25' and UHID IN  (select distinct(uhid) from "+schemaName+".babyfeed_detail where uhid in  ( select distinct(uhid) from "+schemaName+".baby_visit where uhid in (select  distinct(uhid) from "+schemaName+".nursing_vitalparameters where uhid in ( select distinct(uhid) from "+schemaName+".device_monitor_detail UNION select distinct(uhid) from "+schemaName+".device_monitor_detail_dump)))) and "+conditionCase+" and isreadmitted is not true and gestationweekbylmp is not null and birthweight is not null and uhid = '"+patientCaseUHID+"';"
        cur11.execute(queryStr)
        #print(queryStr)
        cols11 = list(map(lambda x: x[0], cur11.description))
        uhids = pd.DataFrame(cur11.fetchall(),columns=cols11)

        s2 = set(uhids.uhid.unique())
        #print('Distinct UHID 2=',len(s2))
        l = list(s1 - s1.intersection(s2))

        fixed.set_index('uhid',inplace=True)

        df = fixed.drop(l)

        df.reset_index(inplace=True)
        dates_detail = df.copy()
        #print(dates_detail)
        #print('dates_detail')
        dates_detail['add_seconds'] = dates_detail['timeofbirth'].apply(second_addition)
        dates_detail['add_seconds_admission'] = dates_detail['timeofadmission'].apply(second_addition)

        #Combining Date and Time(Admission and Birth)
        dates_detail['actual_DOB'] = dates_detail.apply(lambda x: actual_birthdate(x['dateofbirth'], x['add_seconds']), axis=1)
        dates_detail['actual_DOA'] = dates_detail.apply(lambda x: actual_birthdate(x['dateofadmission'], x['add_seconds_admission']), axis=1)
        dd = dates_detail[['actual_DOB','uhid','dischargeddate','actual_DOA','gender','birthweight','birthlength','birthheadcircumference','inout_patient_status'
        ,'gestation','gestationweekbylmp','gestationdaysbylmp','baby_type','central_temp','apgar_onemin',
        'apgar_fivemin','apgar_tenmin','motherage','conception_type','mode_of_delivery','steroidname',
        'numberofdose','dischargestatus','birthlength','birthheadcircumference','inout_patient_status'
        ,'gestationweekbylmp','gestationdaysbylmp','baby_type','central_temp','apgar_onemin',
        'apgar_fivemin','apgar_tenmin','motherage','conception_type','mode_of_delivery','steroidname',
        'numberofdose']]

        dd.dropna(subset=['dischargeddate'],inplace=True)
        dd['los'] = dates_detail['dischargeddate'] - dates_detail['actual_DOA']

        dd['day_1'] = dd['los'].apply(day)

        dd = dd[dd['day_1']>=0]

        dd.sort_values(by = ['actual_DOA'],inplace=True)
        dd.drop_duplicates(subset=['uhid'],keep='first',inplace=True)
        dt = pd.DataFrame()
        for i in dd.uhid.unique():
            x = dd[dd['uhid']==i]
            time_s = []
            n = math.ceil((x.dischargeddate.iloc[0] - x.actual_DOA.iloc[0]).total_seconds()/60)
            try:
                x = pd.concat([x]*int(n))

                for inner in range(0,len(x)):
                    #print(i)
                    time_s.append(x['actual_DOA'].iloc[inner] + timedelta(seconds = inner*60))
                    #t['hour_series'].loc[i] = t['actual_DOA'].iloc[i] + timedelta(hours = i)

                x['hour_series'] = time_s
            except Exception as e:
                print(i,n, e)
                continue
            dt = dt.append(x,ignore_index=True)

        dt.drop_duplicates(subset=['uhid','hour_series'],inplace=True)



        cur1 = con.cursor()
        cur1.execute("select distinct(b.uhid),l.conception_type, b.gender, b.dateofadmission, b.dischargestatus,b.birthweight,b.weight_galevel,b.weight_centile,b.birthlength,b.birthheadcircumference, b.inout_patient_status, b.gestationweekbylmp, b.gestationdaysbylmp,round( CAST((b.gestationweekbylmp + b.gestationdaysbylmp/7::float) as numeric),2) as Gestation,b.dischargeddate, b.admissionweight, b.baby_type,b.baby_number, b.branchname ,DATE_PART('day',b.dischargeddate - b.dateofadmission) as LOS,c.apgar_onemin, c.apgar_fivemin,c.apgar_tenmin, c.resuscitation,d.isantenatalsteroidgiven, d.mode_of_delivery,z.motherage,e.jaundicestatus as JAUNDICE,  f.eventstatus as SEPSIS,f.progressnotes, g.eventstatus as RDS, g.progressnotes,y.eventstatus as ASPHYXIA,y.progressnotes from "+schemaName+".baby_detail as b left join "+schemaName+".birth_to_nicu as c on b.uhid = c.uhid and b.episodeid = c.episodeid left join "+schemaName+".parent_detail as z on b.uhid = z.uhid and b.episodeid = z.episodeid left join "+schemaName+".antenatal_history_detail as d on b.uhid = d.uhid and b.episodeid = d.episodeid left join "+schemaName+".sa_jaundice AS e ON b.uhid = e.uhid and e.jaundicestatus = 'Yes' and e.phototherapyvalue='Start' left join "+schemaName+".sa_infection_sepsis AS f ON b.uhid = f.uhid and f.eventstatus = 'yes' and f.episode_number = 1 left join "+schemaName+".sa_cns_asphyxia AS y ON b.uhid = y.uhid and y.eventstatus = 'yes' and y.episode_number = 1 left join "+schemaName+".antenatal_history_detail as l on b.uhid=l.uhid left join "+schemaName+".sa_resp_rds AS g ON b.uhid = g.uhid and g.eventstatus = 'Yes' and g.episode_number = 1 and g.uhid IN (select distinct(h.uhid) from "+schemaName+".respsupport AS h where h.eventname='Respiratory Distress' and (h.rs_vent_type ='Mechanical Ventilation' OR h.rs_vent_type ='HFO') UNION select distinct(i.uhid) from "+schemaName+".sa_resp_rds AS i where i.sufactantname is not null) and b.uhid = '"+patientCaseUHID+ "' order by b.dateofadmission")
        cols1 = list(map(lambda x: x[0], cur1.description))
        ds = pd.DataFrame(cur1.fetchall(),columns=cols1)

        data = ds.copy()

        ids = dt.uhid.unique()

        data.drop_duplicates('uhid',keep='first',inplace=True)

        df = pd.DataFrame(columns=data.columns)

        for i in ids:
            x = data[data['uhid']==i]
            df = df.append(x,ignore_index=True)

        pd.set_option('display.max_columns',100)

        ds = pd.DataFrame()

        ds = dt.copy()
        uhid = ds.uhid.unique()
        dt = pd.DataFrame()

        for i in uhid:
            x = dd[dd['uhid']==i]
            time_s = []
            n = math.ceil((x.dischargeddate.iloc[0] - x.actual_DOA.iloc[0]).total_seconds()/60)
            try:
                x = pd.concat([x]*int(n))
                for inner in range(0,len(x)):
                    #print(i)
                    time_s.append(x['actual_DOA'].iloc[inner] + timedelta(seconds = inner*60))
                    #t['hour_series'].loc[i] = t['actual_DOA'].iloc[i] + timedelta(hours = i)

                x['hour_series'] = time_s
            except Exception as inst:
                print(inst)
                continue
            dt = dt.append(x,ignore_index=True)

        dt['hour_series'].apply(split_hour)

        dt['hour'] = dt['hour_series'].apply(split_hour)
        dt['day'] = dt['hour_series'].apply(split_date_1)
        print ('data preparation intake output started')    
        #Intake Output
        cur9= con.cursor()
        cur9.execute("SELECT t1.uhid,t1.abdomen_girth,t1.urine,t1.stool,t1.stool_passed, t1.entry_timestamp FROM "+schemaName+".nursing_intake_output AS t1 where t1.uhid = '"+patientCaseUHID+"'")
        cols9 = list(map(lambda x: x[0], cur9.description))
        output = pd.DataFrame(cur9.fetchall(),columns=cols9)

        output_d = pd.DataFrame()

        for i in uhid:
            x = output[output['uhid']==i]
            output_d = output_d.append(x,ignore_index=True)

        output_d.entry_timestamp = output_d.entry_timestamp + timedelta(seconds=19800)
        output_d['hour'] = output_d.entry_timestamp.apply(split_hour)
        output_d['day'] = output_d.entry_timestamp.apply(split_date_1)

        output_d.sort_values(by=['uhid','entry_timestamp'],inplace=True)
        test = output_d.drop_duplicates(subset=['uhid','hour'],keep='last')
        dh = pd.merge(dt,test,on=['uhid','hour'],how='left')
        dh.sort_values('hour_series')
        dg = pd.DataFrame()

        for i in uhid:
            x = dh[dh['uhid']==i]
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
            dg = dg.append(x,ignore_index=True)  

        dg = dg.sort_values('hour_series')

        data = dg.copy()

        data['series'] = data.hour_series

        dg = data.copy()

        dg['urine'] = dg['urine'].apply(to_float)
        dg['urine'] = dg['urine'].apply(urine_check)
        dg['urine_per_hour'] = dg['urine']/((dg['time_divide'])/(3600))
        dt = dg.sort_values(by = ['uhid','hour_series'])
        ds = pd.DataFrame()

        for i in dt.uhid.unique():
            x = dt[dt['uhid']==i]
            x['urine_per_hour'].fillna(method='bfill',inplace=True)
            ds = ds.append(x,ignore_index=True)

        #Blood Gas
        cur1 = con.cursor()
        cur1.execute("SELECT uhid,creationtime,modificationtime,ph,entrydate FROM "+schemaName+".nursing_bloodgas where uhid ='"+ patientCaseUHID+"'")
        cols1 = list(map(lambda x: x[0],cur1.description))
        ph = pd.DataFrame(cur1.fetchall(),columns=cols1)

        ph['new_ph'] = ph.ph.apply(ph_func)
        ph.dropna(subset=['new_ph'],inplace=True)
        ph_df = pd.DataFrame()
        for i in ph.uhid.unique():
            x = ph[ph['uhid']==i]
            ph_df = ph_df.append(x,ignore_index=True)

        if len(ph_df.columns)>0:
            ph_df['hour'] = ph_df.entrydate.apply(split_hour)
            ph_df.drop_duplicates(subset=['uhid','hour'],keep='last',inplace=True)
        else:
            x = pd.DataFrame(data=[patientCaseUHID],columns={'uhid'})
            print(x)
            ph_df_1 = pd.DataFrame(columns=ph.columns)
            ph_df_1 = ph_df_1.append(x,ignore_index=True)

        ds['uhid'] = ds['uhid'].astype(str)

        if len(ph_df.columns)>0:
            ss = pd.merge(ds,ph_df,on=['uhid','hour'],how='left')
            ss = ss.drop_duplicates(subset=['uhid','hour_series'],keep='first')
        else:
            ss = pd.merge(ds,ph_df_1,on=['uhid'],how='left')
            ss = ss.drop_duplicates(subset=['uhid','hour_series'],keep='first')

        #Vitals
        print ('data preparation vitals - continuous data started')    

        cur3 = con.cursor()
        cur3.execute("SELECT t1.uhid,t1.entrydate,t1.rbs,t1.skintemp,t1.centraltemp  FROM "+schemaName+".nursing_vitalparameters AS t1 where t1.uhid = '"+patientCaseUHID+"'")
        cols3 = list(map(lambda x: x[0], cur3.description))
        nv = pd.DataFrame(cur3.fetchall(),columns=cols3)

        nv.drop_duplicates(subset=['uhid','entrydate'],keep='first',inplace=True)

        nv['hour'] = nv.entrydate.apply(split_hour)

        ss.uhid = ss.uhid.astype(str)

        s1 = pd.merge(ss,nv,on = ['uhid','hour'], how='left')

        s1.drop_duplicates(subset=['uhid','hour_series'],keep='first',inplace=True)

        s1['temp'] = s1.apply(lambda x: temp_a(x['centraltemp'], x['skintemp']), axis=1)

        #Anthropometry
        cur2 = con.cursor()
        cur2.execute("SELECT t1.uhid,t1.visitdate,t1.visittime,t1.currentdateweight, t1.currentdateheight  FROM "+schemaName+".baby_visit AS t1 where t1.uhid = '"+patientCaseUHID+"'")
        cols2 = list(map(lambda x: x[0], cur2.description))
        bv = pd.DataFrame(cur2.fetchall(),columns=cols2)

        bv.sort_values('visitdate',inplace=True)

        bv['hour'] = bv.apply(lambda x: stamp_1(x['visitdate'], x['visittime']), axis=1)

        bv['day'] = bv['visitdate'].apply(to_str)

        bv.drop_duplicates(subset=['uhid','visitdate'],keep='last',inplace=True)

        s1['day'] = s1['hour_series'].apply(split_date_1)

        s2 = pd.merge(s1,bv,on = ['uhid','day'], how='left')

        #Medications
        cur7= con.cursor()
        cur7.execute("SELECT t1.uhid,t1.startdate,t1.medicineorderdate,t1.medicinename,t1.medicationtype,t2.typevalue FROM "+schemaName+".baby_prescription AS t1 LEFT JOIN "+schemaName+".ref_medtype AS t2 ON t1.medicationtype = t2.typeid WHERE (t1.medicationtype = 'TYPE0001' OR t1.medicationtype = 'TYPE0009' OR t1.medicationtype = 'TYPE0004') and (t1.uhid ='"+patientCaseUHID+"')")
        cols7 = list(map(lambda x: x[0], cur7.description))
        meds = pd.DataFrame(cur7.fetchall(),columns=cols7)

        meds_dummies = pd.get_dummies(meds,columns=['typevalue'])
        meds_dummies['hour'] = meds_dummies.startdate.apply(split_hour)
        s2.drop(['hour_x','hour_y'],axis=1,inplace=True)

        s2['hour'] = s2.hour_series.apply(split_hour)

        s3 = pd.merge(s2,meds_dummies,on = ['uhid','hour'], how='left')

        s3['day'] = s3['hour_series'].apply(split_date_1)

        #Nutrition
        cur8= con.cursor()
        cur8.execute("SELECT t1.uhid,t1.entrydatetime,t1.totalparenteralvolume,t1.total_intake FROM "+schemaName+".babyfeed_detail AS t1 where t1.uhid = '"+patientCaseUHID+"'")
        cols8 = list(map(lambda x: x[0], cur8.description))
        pn = pd.DataFrame(cur8.fetchall(),columns=cols8)

        pn['hour'] = pn.entrydatetime.apply(split_hour)
        pn['total_intake'] = pn['total_intake'].astype(float)
        pn['tpn-tfl'] = pn['totalparenteralvolume']/pn['total_intake']

        s4 = pd.merge(s3,pn,on = ['uhid','hour'], how='left')

        s4.drop_duplicates(subset=['uhid','hour_series'],inplace=True)
        q = pd.DataFrame()
        for i in s4.uhid.unique():
            x = s4[s4['uhid']==i]
            x['currentdateheight'].fillna(method='ffill',inplace=True)
            x['currentdateweight'].fillna(method='ffill',inplace=True)
            q = q.append(x,ignore_index=True)

        q['abdomen_girth'] = pd.to_numeric(q['abdomen_girth'], errors='coerce')
        q['abdomen_girth'] = q['abdomen_girth'].apply(abd)

        dt = pd.DataFrame()
        for i in q.uhid.unique():
            x = q[q['uhid']==i]
            #for j in y['day'].unique():
            #x = y[y['day']==j]
            x['abdomen_girth'].fillna(method='bfill',inplace=True)
            x['currentdateweight'].fillna(method='bfill',inplace=True)
            x['currentdateheight'].fillna(method='bfill',inplace=True)
            x['totalparenteralvolume'].fillna(method='ffill',inplace=True)
            x['total_intake'].fillna(method='ffill',inplace=True)
            x['tpn-tfl'].fillna(method='ffill',inplace=True)
            dt = dt.append(x,ignore_index=True)

        a = dt.replace([np.inf, -np.inf], np.nan)

        a['abdomen_girth'] = a['abdomen_girth'].apply(abd_2)
        a['abdomen_girth'] = a['abdomen_girth'].apply(abd_3)
        a['urine_per_hour'] = a['urine_per_hour'].apply(upm)
        a['temp'] = a['temp'].apply(temp_f_c)
        a.currentdate = a.currentdateweight.apply(weight_correct)
        a.currentdate = a.currentdateweight.apply(weight_correct_2)
        a['currentdateheight'] = a['currentdateheight'].apply(height_correct)
        a['rbs'] = a['rbs'].apply(rbs_correct)

        data = a.copy()
        data['day'] = data['hour_series'].apply(split_date_1)

        data['hour_series'] = data['hour_series'].apply(to_date)
        df = pd.DataFrame()
        for i in data.uhid.unique():
            x = data[data['uhid']==i]
            x.rbs.fillna(method='ffill',inplace=True)
            x.currentdateweight.fillna(method='ffill',inplace=True)
            x.currentdateheight.fillna(method='ffill',inplace=True)
            n = math.ceil(len(x)/24)+1
            start_date = pd.to_datetime(x['day'].iloc[0]+" " + "08:00:00") - timedelta(hours=24)
            for i in range(int(n)):
                
                
                y = x[(x['hour_series']>=start_date + timedelta(hours=24*i)) & (x['hour_series']<=start_date + timedelta(hours=24*(i+1)))]
                y['stool_day_total'] = (y['stool_passed'].sum())/60
                df =df.append(y,ignore_index=True)

        test = df.copy()
        dq = pd.DataFrame()
        for i in test.uhid.unique():
            x = test[test['uhid']==i]
            x.abdomen_girth.fillna(method='ffill',inplace=True)
            dq = dq.append(x, ignore_index=True)

        final = pd.DataFrame()
        for i in dq.uhid.unique():
            x = dq[dq['uhid']==i]
            x['abd_difference_y'] = x['abdomen_girth'] - x['abdomen_girth'].shift(periods=1)
            final = final.append(x,ignore_index=True)

        #Vital Parameters
        cur1 = con.cursor()
        #combine data of monitor and monitor dump
        queryStr = "SELECT t11.uhid,t11.starttime,t11.pulserate, t11.ecg_resprate, t11.spo2, t11.heartrate, t21.dischargestatus,t11.mean_bp,t11.sys_bp,t11.dia_bp  FROM "+schemaName+".device_monitor_detail AS t11 LEFT JOIN "+schemaName+".baby_detail AS t21 ON t11.uhid=t21.uhid WHERE (t21.dischargestatus = '"+caseType+"' AND t21.dateofadmission > '2018-07-01' and t21.uhid = '"+ patientCaseUHID+"' AND t11.starttime < t21.dischargeddate)  UNION" + " SELECT t12.uhid,t12.starttime,t12.pulserate, t12.ecg_resprate, t12.spo2, t12.heartrate, t22.dischargestatus,t12.mean_bp,t12.sys_bp,t12.dia_bp  FROM "+schemaName+".device_monitor_detail_dump AS t12 LEFT JOIN "+schemaName+".baby_detail AS t22 ON t12.uhid=t22.uhid WHERE (t22.dischargestatus = '"+caseType+"' AND t22.dateofadmission > '2018-07-01' and t22.uhid = '"+ patientCaseUHID+"' AND t12.starttime < t22.dischargeddate);"
        cur1.execute(queryStr)
        cols1 = list(map(lambda x: x[0], cur1.description))
        ds = pd.DataFrame(cur1.fetchall(),columns=cols1)
        ds.sort_values('starttime',inplace=True)

        test = ds.drop_duplicates(subset=['uhid','starttime'],keep='first')

        #Ventilator Parameters
        cur_vent = con.cursor()
        cur_vent.execute("SELECT t1.uhid,t1.start_time,t1.creationtime,t1.peep, t1.pip,t1.map ,t1.tidalvol, t1.minvol,t1.ti,t1.fio2 FROM "+schemaName+".device_ventilator_detail_dump AS t1 LEFT JOIN "+schemaName+".baby_detail AS t2 ON t1.uhid=t2.uhid WHERE (t2.dischargestatus = '"+caseType+"' OR t2.dischargestatus = '"+caseType+"') and t2.uhid = '"+patientCaseUHID+"';")
        cols_vent = list(map(lambda x: x[0], cur_vent.description))
        ventilator_cont = pd.DataFrame(cur_vent.fetchall(),columns=cols_vent)

        test_vent = ventilator_cont.drop_duplicates(subset=['uhid','start_time'],keep='first')

        test['date'] = test.starttime.apply(split_date)
        test_vent['date'] = test_vent.start_time.apply(split_date)

        cont_data = pd.merge(test,test_vent,on=['uhid','date'],how='left',copy=False)
        test_cont = cont_data.drop_duplicates(subset=['uhid','starttime','heartrate'],keep='first')

        test_cont['hour_series'] = test_cont['date'].apply(split_hour)


        test_cont['ref_hour'] = test_cont['hour_series'].apply(to_str)
        final['ref_hour'] = final['hour_series'].apply(to_str)

        test_cont['cont_time'] = test_cont.starttime.apply(con_time)

        final['cont_time'] = final.ref_hour.apply(con_time_2)
        qw = pd.merge(final,test_cont, on=['uhid','cont_time'],how='left')
        # Next is data fill step
        print ('data preparation forward filling started')    
        qw['pulserate'].fillna(method='ffill',limit=5)
        qw['heartrate'].fillna(method='ffill',limit=5)
        qw['ecg_resprate'].fillna(method='ffill',limit=5)
        qw['spo2'].fillna(method='ffill',limit=5)
        qw['mean_bp'].fillna(method='ffill')
        qw['sys_bp'].fillna(method='ffill')
        qw['dia_bp'].fillna(method='ffill')
        qw['mean_bp'].fillna(method='ffill')
        qw['peep'].fillna(method='ffill')
        qw['pip'].fillna(method='ffill')
        qw['map'].fillna(method='ffill')
        qw['tidalvol'].fillna(method='ffill')
        qw['minvol'].fillna(method='ffill')
        qw['ti'].fillna(method='ffill')
        qw['fio2'].fillna(method='ffill')
        qw['abdomen_girth'].fillna(method='ffill')
        qw['urine'].fillna(method='ffill')
        qw['stool_day_total'].fillna(method='ffill')
        qw['new_ph'].fillna(method='ffill')
        qw['rbs'].fillna(method='ffill')
        qw['skintemp'].fillna(method='ffill')
        qw['centraltemp'].fillna(method='ffill')
        qw['temp'].fillna(method='ffill')
        qw['currentdateweight'].fillna(method='ffill')
        qw['currentdateheight'].fillna(method='ffill')
        qw['totalparenteralvolume'].fillna(method='ffill')
        qw['total_intake'].fillna(method='ffill')
        qw['tpn-tfl'].fillna(method='ffill')
        if caseType == "Death":
            qw['dischargestatus'] = 1
        elif  caseType == "Discharge":
            qw['dischargestatus'] = 0     
        if not os.path.exists(filePath):
            os.makedirs(filePath)   
        qw.to_csv(fileName)
        return fileName, qw
    except Exception as e:
        print('Exception in data preparation', e)
        PrintException()
        return None
        