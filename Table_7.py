import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Reading CSV for 30 patients
data = pd.read_csv('lstm_analysis.csv')
#Replacing -999 with NaN values 
data.fillna(-999,inplace=True)
cols = ['uhid','pulserate', 'ecg_resprate',
       'spo2', 'heartrate', 'mean_bp', 'sys_bp', 'dia_bp',
       'peep', 'pip', 'map', 'tidalvol', 'minvol', 'ti', 'fio2',
       'abd_difference_y',
       'currentdateheight',
       'currentdateweight','dischargestatus', 
       'new_ph', 
       'rbs',  'stool_day_total', 
       'temp', 'total_intake', 'totalparenteralvolume',
       'tpn-tfl', 'typevalue_Antibiotics',
       'urine','gender', 'birthweight',
       'birthlength', 'birthheadcircumference', 'inout_patient_status',
       'gestationweekbylmp', 'gestationdaysbylmp',
       'baby_type', 'central_temp', 'apgar_onemin', 'apgar_fivemin',
       'apgar_tenmin', 'motherage', 'conception_type', 'mode_of_delivery',
       'steroidname', 'numberofdose', 'gestation']

#Divide the data set into two sub sets according to discharge and death
death = data[data['dischargestatus']==1]
dis = data[data['dischargestatus']==0]
#Calculating Length
n_dis = len(dis)
n_dea = len(death)

#Plotting Graphs and calculating imputation of each variable for death cases
x_series = []
y_series = []
counter = -1
for i in cols:
    counter = counter + 1
        
    if(counter == 6):
        plt.bar(x_series,y_series,align='center') # A bar chart
        plt.xlabel('Parameters')
        plt.ylabel('Data Missing(percentage)')
        for inner in range(len(y_series)):
            plt.hlines(y_series[inner],0,x_series[inner]) # Here you are drawing the horizontal lines
        plt.show()
        x_series = []
        y_series = []
        counter = 0
    x_series.append(i)
    y_series.append(((len(death[death[i]==-999]))/n_dea)*100)
    print(i,(len(death[death[i]==-999])),((len(death[death[i]==-999]))/n_dea)*100)
    
if(counter > 0):
    plt.bar(x_series,y_series,align='center') # A bar chart
    plt.xlabel('Parameters')
    plt.ylabel('Data Missing(percentage)')
    for inner in range(len(y_series)):
        plt.hlines(y_series[inner],0,x_series[inner]) # Here you are drawing the horizontal lines
    plt.show()
    x_series = []
    y_series = []
    counter = 0
  
#Calculating imputation of each variable for Discharge cases
for i in cols:
    print(i,(len(dis[dis[i]==-999])),((len(dis[dis[i]==-999]))/len(dis))*100)
