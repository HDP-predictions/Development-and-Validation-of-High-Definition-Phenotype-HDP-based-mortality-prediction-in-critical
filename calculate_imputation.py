#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import sys
import linecache
import matplotlib.pyplot as plt


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

def calculateDataImputation(data):

    try:

        print('----Data Imputation Started-------')
        path = os.getcwd()
        folderNameDischarge = "/Imputation_Discharge/"
        folderNameDeath = "/Imputation_Death/"
        plt.rcParams["figure.figsize"] = (20,10)
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
            'tpn-tfl', 'typevalue_Antibiotics' , 'typevalue_Inotropes',
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
        graph_number = 1

        filePath = path+folderNameDeath
        if not os.path.exists(filePath):
            os.makedirs(filePath)

        filePath = path+folderNameDischarge
        if not os.path.exists(filePath):
            os.makedirs(filePath)

        if(n_dea > 0):

            for i in cols:
                counter = counter + 1
                    
                #Plotting only 6 variables at a time on graph
                if(counter == 6):
                    fig, (ax1) = plt.subplots(1, sharex=True)
                    plt.bar(x_series,y_series,align='center') # A bar chart
                    plt.xlabel('Parameters')
                    plt.ylabel('Data Missing(percentage)')
                    for inner in range(len(y_series)):
                        plt.hlines(y_series[inner],0,x_series[inner]) # Here you are drawing the horizontal lines

                    plt.savefig(path+folderNameDeath+'imputation' + str(graph_number)+ '.png')
                    graph_number = graph_number + 1
                    #plt.show()
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
                plt.savefig(path+folderNameDeath+'imputation' + str(graph_number)+ '.png')
                x_series = []
                y_series = []
                counter = 0
        
        #Plotting Graphs and calculating imputation of each variable for Discharge cases
        x_series = []
        y_series = []
        counter = -1
        graph_number = 1
        if(n_dis > 0):
            for i in cols:
                counter = counter + 1
                    
                #Plotting only 6 variables at a time on graph
                if(counter == 6):
                    plt.bar(x_series,y_series,align='center') # A bar chart
                    plt.xlabel('Parameters')
                    plt.ylabel('Data Missing(percentage)')
                    for inner in range(len(y_series)):
                        plt.hlines(y_series[inner],0,x_series[inner]) # Here you are drawing the horizontal lines
                    plt.savefig(path+folderNameDischarge+'imputation' + str(graph_number)+ '.png')
                    graph_number = graph_number + 1
                    #plt.show()
                    x_series = []
                    y_series = []
                    counter = 0
                x_series.append(i)
                y_series.append(((len(dis[dis[i]==-999]))/n_dis)*100)
                print(i,(len(dis[dis[i]==-999])),((len(dis[dis[i]==-999]))/n_dis)*100)
                
            if(counter > 0):
                plt.bar(x_series,y_series,align='center') # A bar chart
                plt.xlabel('Parameters')
                plt.ylabel('Data Missing(percentage)')
                for inner in range(len(y_series)):
                    plt.hlines(y_series[inner],0,x_series[inner]) # Here you are drawing the horizontal lines
                plt.savefig(path+folderNameDischarge+'imputation' + str(graph_number)+ '.png')
                x_series = []
                y_series = []
                counter = 0

            print('----Data Imputation Ended-------')
    
    except Exception as e:
        print('Exception in data Imputation', e)
        PrintException()
        return False   

    