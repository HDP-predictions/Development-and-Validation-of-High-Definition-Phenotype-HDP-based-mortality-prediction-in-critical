
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
"""
folderName = "/Discharge_Cases/"
patientCaseUHID = "RSHI.0000012314"
typeOfCase = "Discharge"
fileName = prepare_data(patientCaseUHID,typeOfCase,"(dischargestatus = 'Discharge')",folderName)
visualizeDataset(fileName,folderName,patientCaseUHID,typeOfCase)
"""
#Check how many death cases, as per length of data in death cases build equal number of discharge cases 
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
balanceDS = balanceDataset(con)
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
        fileName,uhidDataSet = read_prepare_data(patientCaseUHID,typeOfCase,conditionCase,folderName)
        print('UHID',patientCaseUHID,'data preperation done total number of colums built=',len(uhidDataSet))
        visualFlag = visualizeDataset(fileName,folderName,patientCaseUHID,typeOfCase)
        print('UHID',patientCaseUHID,'data visualization done')
        preparedData = preparedData.append(uhidDataSet)
        print('preparedData length=',len(preparedData))
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
print('1columns=',preparedData.columns)
predictLSTM(preparedData, fixed, cont, inter)