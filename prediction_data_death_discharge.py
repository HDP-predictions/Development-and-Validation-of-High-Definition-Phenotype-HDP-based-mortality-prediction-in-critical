
import os
import sys
import linecache
from balance_data_death_discharge import *
from data_preparation_hdp import *
from data_visualization import *
from prediction_using_lstm import *
from calculate_imputation import *
from sklearn.model_selection import train_test_split
from random import shuffle
from math import *
#from prediction_lrm import *

def splittingSets(dfCase,final_df):

    #Calculating the count of every UHID
    symbols = dfCase.groupby('uhid')

    trainingSetFrame = pd.DataFrame(columns=final_df.columns)
    testingSetFrame = pd.DataFrame(columns=final_df.columns)

    #Preparing a dictionary to store uhid and its count
    dict = {}
    listValues = []
    for symbol, group in symbols:
        print(symbol)
        print(len(group))
        dict[symbol] = len(group)
        listValues.append(len(group))

    #Sorting according the count - Descending
    sort_orders = sorted(dict.items(), key=lambda x: x[1], reverse=True)

    print(sort_orders)

    #Dividing the count into centiles
    firstQuartileValue = np.quantile(listValues, .75)
    secondQuartileValue = np.quantile(listValues, .50)
    thirdQuartileValue = np.quantile(listValues, .25)

    firstQuartile = []
    secondQuartile = []
    thirdQuartile = []
    fourthQuartile = []

    #According to centiles the patients are divided into 4 categories
    for i in range(0, len(sort_orders)):
        if(sort_orders[i][1] >= firstQuartileValue):
            firstQuartile.append(sort_orders[i][0])
        elif(sort_orders[i][1] < firstQuartileValue and sort_orders[i][1] >= secondQuartileValue):
            secondQuartile.append(sort_orders[i][0])
        elif(sort_orders[i][1] < secondQuartileValue and sort_orders[i][1] >= thirdQuartileValue):
            thirdQuartile.append(sort_orders[i][0])
        else:
            fourthQuartile.append(sort_orders[i][0])

    #Function used to split the data set into training(70%) and testing(30%) from each category
    train,test = splittingQuartiles(firstQuartile,final_df)
    trainingSetFrame = trainingSetFrame.append(train)
    testingSetFrame = testingSetFrame.append(test)

    train,test = splittingQuartiles(secondQuartile,final_df)
    trainingSetFrame = trainingSetFrame.append(train)
    testingSetFrame = testingSetFrame.append(test)

    train,test = splittingQuartiles(thirdQuartile,final_df)
    trainingSetFrame = trainingSetFrame.append(train)
    testingSetFrame = testingSetFrame.append(test)

    train,test = splittingQuartiles(fourthQuartile,final_df)
    trainingSetFrame = trainingSetFrame.append(train)
    testingSetFrame = testingSetFrame.append(test)

    return trainingSetFrame, testingSetFrame, sort_orders

def splittingQuartiles(quartileList,original_df):
    trainingSetDummy = pd.DataFrame(columns=original_df.columns)
    testingSetDummy = pd.DataFrame(columns=original_df.columns)
    shuffle(quartileList)
    trainingLength = round(len(quartileList) * 0.7)
    for i in range(len(quartileList)):
        if i < trainingLength:
            train = original_df[original_df.uhid == quartileList[i]]
            trainingSetDummy = trainingSetDummy.append(train)
        else:
            test = original_df[original_df.uhid == quartileList[i]]
            testingSetDummy = testingSetDummy.append(test)

    return trainingSetDummy,testingSetDummy

def range_finder(x):
    length = x
    fractional = (x/15.0) - math.floor(x/15.0)
    return int(round(fractional*15))

def prepareTrainTestSet(gd):
    try:
        print('--------inside make_lstm')
        final_df = pd.DataFrame(columns=gd.columns)
        ids = gd.uhid.unique()
        #print('------inside make lstm---unique uhid count =',len(ids))
        shuffle(ids)
        for i in ids:
            x = gd[gd['uhid']==i]
            x = x[range_finder(len(x)):len(x)]
            final_df = final_df.append(x,ignore_index=True)

        final_df.fillna(-999,inplace=True)
        #Copy the original file before applying the check of spo2
        original_df = final_df.copy()

        trainingSet = pd.DataFrame(columns=gd.columns)
        testingSet = pd.DataFrame(columns=gd.columns)

        final_df = final_df[final_df["spo2"] != -999]
        
        #Splitting the Data from into discharge and death
        deathCases = final_df[final_df.dischargestatus == 1]
        dischargeCases = final_df[final_df.dischargestatus == 0]

        print('Splitting of Death cases start')
        #Firstly splitting 15 death cases to train and test
        trainDeath, testDeath, sort_orders_death = splittingSets(deathCases,final_df)
        print('Splitting of Death cases end')
        print('Splitting of Discharge cases start')
        #Secondly splitting 15 discharge cases to train and test
        trainDischarge, testDischarge, sort_orders_discharge = splittingSets(dischargeCases,final_df)
        print('Splitting of Discharge cases end')

        print('--------------------TRAINING SET BEFORE BALANCE---------------------')
        print("Death cases" , trainDeath.uhid.unique(), len(trainDeath))
        print("Discharge cases" , trainDischarge.uhid.unique(),len(trainDischarge))
        #testingSet is used for testing (30%)
        #testingSet = ids[train_count:]
        print('--------------------TESTING SET BEFORE BALANCE----------------------')
        print("Death cases" , testDeath.uhid.unique(),len(testDeath))
        print("Discharge cases" , testDischarge.uhid.unique(),len(testDischarge))

        #Balancing the data of Discharge and Death. If discharge count is more 
        #then prune extra count from death case and vice-versa
        for i in range(0, len(sort_orders_discharge)):

            countDeath = sort_orders_death[i][1]
            countDischarge = sort_orders_discharge[i][1]
            if(countDeath > countDischarge):
                
                train = trainDeath[trainDeath.uhid == sort_orders_death[i][0]]
                if(len(train) > 0):
                    trainFinal = train[:(countDischarge)]
                    trainingSet = trainingSet.append(trainFinal)
                else:
                    test = testDeath[testDeath.uhid == sort_orders_death[i][0]]
                    testFinal = test[:(countDischarge)]
                    testingSet = testingSet.append(testFinal)

                train = trainDischarge[trainDischarge.uhid == sort_orders_discharge[i][0]]
                if(len(train) > 0):
                    trainingSet = trainingSet.append(train)
                else:
                    
                    test = testDischarge[testDischarge.uhid == sort_orders_discharge[i][0]] 
                    testingSet = testingSet.append(test)
            else:
                train = trainDischarge[trainDischarge.uhid == sort_orders_discharge[i][0]]
                if(len(train) > 0):
                    trainFinal = train[:(countDeath)]
                    trainingSet = trainingSet.append(trainFinal)
                else:
                    test = testDischarge[testDischarge.uhid == sort_orders_discharge[i][0]]
                    testFinal = test[:(countDeath)]
                    testingSet = testingSet.append(testFinal)

                train = trainDeath[trainDeath.uhid == sort_orders_death[i][0]]
                if(len(train) > 0):
                    trainingSet = trainingSet.append(train)
                else:
                    test = testDeath[testDeath.uhid == sort_orders_death[i][0]]
                    testingSet = testingSet.append(test)
           

        #Calculating Death Count
        #death_count = final_df[final_df.dischargestatus == 1]

        #Getting the 70% percent value for training model
        #train_count = int(0.7 * len(ids))
        #print(train_count)
        #trainingSet is used for training (70%)
        #trainingSet = ids[0:train_count]
        print('--------------------TRAINING SET AFTER BALANCE---------------------')
        print(trainingSet.uhid.unique(), len(trainingSet))
        deathTraining = trainingSet[trainingSet.dischargestatus == 1]
        print("Death cases" , deathTraining.uhid.unique(), len(deathTraining))

        dischargeTraining = trainingSet[trainingSet.dischargestatus == 0]
        print("Discharge cases" , dischargeTraining.uhid.unique(), len(dischargeTraining))
        #testingSet is used for testing (30%)
        #testingSet = ids[train_count:]
        print('--------------------TESTING SET AFTER BALANCE----------------------')
        print(testingSet.uhid.unique(), len(testingSet))
        deathTesting = testingSet[testingSet.dischargestatus == 1]
        print("Death cases" , deathTesting.uhid.unique(), len(deathTesting))

        dischargeTesting = testingSet[testingSet.dischargestatus == 0]
        print("Discharge cases" , dischargeTesting.uhid.unique(), len(dischargeTesting))
  
        return trainingSet,testingSet
    except Exception as e:
            print('Error in make_lstm method',e)
            PrintException()
            return None

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
folderName = ""
typeOfCase = ""
deathCase = 1
dischargeCase = 0
print("---------Balancing Data----------")
con = psycopg2.connect (user = 'postgres',
                password = 'postgres',
                port = '5432',
                host = 'localhost',                
                database = 'inicudb')

#The variable would be True if discharge cases are retrieved on the basis of birth weight and gestation
enablingRandomize = False
#generate new set of death and discharge cases
#balanceDS = balanceDataset(con,enablingRandomize)
#we can also load previously generated set whose data preparation is already done for faster execution
"""
"""
balanceDS = pd.read_csv('death_discharge_set.csv')
print('Length of balanced dataset',len(balanceDS))
print("---------Preparing Data----------")
preparedData = pd.DataFrame()
visualFlag = False
i = 1
hdpPlotdict = {}
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
        visualFlag,hdpAX = visualizeDataset(fileName,folderName,patientCaseUHID,typeOfCase)
        dictEntry = {patientCaseUHID:hdpAX} 
        hdpPlotdict.update(dictEntry)
        #print('hdpPlotdict = ',hdpPlotdict)
        #print('UHID',patientCaseUHID,'data visualization done')
        #calculateDataImputation(uhidDataSet)
        preparedData = pd.concat([preparedData,uhidDataSet], axis=0, ignore_index=True)
        print('UHID',patientCaseUHID,'data preperation done total number of rows added =',len(uhidDataSet), 'number of columns in new frame='+str(len(uhidDataSet.columns)),'number of columns in total frame='+str(len(preparedData.columns)))        
        #preparedData = preparedData.append(uhidDataSet)
        print('preparedData length=',len(preparedData),'  added uhid minutes=',len(uhidDataSet))
    except Exception as e:
        print('Exception in prediction_data_death_discharge', e)
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
#preparedData = pd.read_csv('lstm_analysis.csv')
print('Total number of columns in new frame='+str(len(preparedData.columns)))

for i in range(5):
    trainingSet,testingSet = prepareTrainTestSet(preparedData)
    predictLSTM(preparedData, fixed, cont, inter,hdpPlotdict,trainingSet,testingSet)
#predictLRM(preparedData)