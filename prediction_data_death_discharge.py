
import os
import sys
import linecache
from balance_data_death_discharge import *
from data_preparation_hdp import *
from data_visualization import *
from prediction_using_lstm import *

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
    return True

try:
    con = psycopg2.connect (user = 'postgres',
                    password = 'postgres',
                    port = '5433',
                    host = 'localhost',                
                    database = 'inicudb')
    preparedData = pd.DataFrame()
    conditionCase = "(dischargestatus = 'Discharge')"
    data = [["RJUB.12475","Discharge"]]
    patientCaseUHIDSet = pd.DataFrame(data,columns=['uhid','typeOfCase'])
    #fileName = prepare_data(patientCaseUHID,typeOfCase,"(dischargestatus = 'Discharge')",folderName)
    for row in patientCaseUHIDSet.itertuples():
        patientCaseUHID = getattr(row, 'uhid')
        typeOfCase = getattr(row, 'typeOfCase')
        if typeOfCase == "Death":
            folderName = "/Death_Cases/"
            conditionCase = "(dischargestatus = 'Death' or dischargestatus = 'LAMA' )"
            typeOfCase = "Death"
        elif typeOfCase == "Discharge":
            folderName = "/Discharge_Cases/"
            conditionCase = "(dischargestatus = 'Discharge')"
            typeOfCase = "Discharge"       
        print('uhid',patientCaseUHID)
        print('typeOfCase',typeOfCase)
        fileName,uhidDataSet = read_prepare_data(con,patientCaseUHID,typeOfCase,conditionCase,folderName)
        preparedData = pd.concat([preparedData,uhidDataSet], axis=0, ignore_index=True)
        #preparedData = preparedData.append(uhidDataSet)
        print('preparedData length=',len(preparedData))
except Exception as e:
    print(e)
    PrintException()
#visualizeDataset(fileName,folderName,patientCaseUHID,typeOfCase)
"""
#Check how many death cases, as per length of data in death cases build equal number of discharge cases 
"""
folderName = ""
typeOfCase = ""
deathCase = 1
dischargeCase = 0
print("---------Balancing Data----------")
con = psycopg2.connect (user = 'postgres',
                password = 'postgres',
                port = '5433',
                host = 'localhost',                
                database = 'inicudb')
#generate new set of death and discharge cases
#balanceDS = balanceDataset(con)
#we can also load previously generated set whose data preparation is already done for faster execution
balanceDS = pd.read_csv('death_discharge_set.csv')
print('Length of balanced dataset',len(balanceDS))
print("---------Preparing Data----------")
preparedData = pd.DataFrame()
visualFlag = False
i = 1
for row in balanceDS.itertuples():
    try:
        print(i,'---->',getattr(row, 'uhid'), getattr(row, 'dischargestatus')) 
        i=i+1

        patientCaseUHID = getattr(row, 'uhid')
        #print('patientCaseUHID', patientCaseUHID)
        caseType = getattr(row, 'dischargestatus')
        #print('caseType',caseType,type(caseType))
        if caseType == deathCase:
            folderName = "/Death_Cases/"
            conditionCase = "(dischargestatus = 'Death' or dischargestatus = 'LAMA' )"
            typeOfCase = "Death"
        elif caseType == dischargeCase:
            folderName = "/Discharge_Cases/"
            conditionCase = "(dischargestatus = 'Discharge')"
            typeOfCase = "Discharge"
        print('patientCaseUHID',patientCaseUHID,'caseType',typeOfCase,'conditionCase',conditionCase,'folderName',folderName)
        # uncomment below to generate data first time
        #fileName,uhidDataSet = prepare_data(con,patientCaseUHID,typeOfCase,conditionCase,folderName)
        # uncomment below in case csv data is already generated and now lstm needs to be executed
        fileName,uhidDataSet = read_prepare_data(con,patientCaseUHID,typeOfCase,conditionCase,folderName)
        visualFlag = visualizeDataset(fileName,folderName,patientCaseUHID,typeOfCase)
        print('UHID',patientCaseUHID,'data visualization done')
        preparedData = pd.concat([preparedData,uhidDataSet], axis=0, ignore_index=True)
        print('UHID',patientCaseUHID,'data preperation done total number of rows added =',len(uhidDataSet), 'number of columns in new frame='+str(len(uhidDataSet.columns)),'number of columns in total frame='+str(len(preparedData.columns)))        
        #preparedData = preparedData.append(uhidDataSet)
        print('preparedData length=',len(preparedData),'  added uhid minutes=',len(uhidDataSet))
    except Exception as e:
        print(e)
        PrintException()
        continue    
print("---------Data Visualization Done----------")
print('Visualization Result=',visualFlag)
print("---------LSTM Analysis Start----------")
fixed = ['dischargestatus',  'gender', 'birthweight',
       'birthlength', 'birthheadcircumference', 'inout_patient_status',
       'gestationweekbylmp', 'gestationdaysbylmp',
       'baby_type', 'central_temp', 'apgar_onemin', 'apgar_fivemin',
       'apgar_tenmin', 'motherage', 'conception_type', 'mode_of_delivery',
       'steroidname', 'numberofdose', 'gestation','uhid']
inter = ['dischargestatus', 'mean_bp',
       'sys_bp', 'dia_bp', 'peep', 'pip', 'map', 'tidalvol',
       'minvol', 'ti', 'fio2',
       'abd_difference_y', 'currentdateheight', 'currentdateweight',
       'new_ph', 'rbs',
       'stool_day_total', 'temp',
       'total_intake', 'totalparenteralvolume',
       'tpn-tfl', 'typevalue_Antibiotics', 'typevalue_Inotropes',
       'urine', 'urine_per_hour', 'uhid']
cont  = ['pulserate','ecg_resprate', 'spo2', 'heartrate', 'dischargestatus', 'uhid']
preparedData = pd.read_csv('lstm_analysis.csv')
print('Total number of columns in new frame='+str(len(preparedData.columns)))
#preparedData.to_csv('lstm_analysis.csv')
predictLSTM(preparedData, fixed, cont, inter)
