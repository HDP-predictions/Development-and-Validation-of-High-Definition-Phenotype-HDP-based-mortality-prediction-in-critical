#!/usr/bin/env python
# coding: utf-8

import os
import sys
import linecache
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[12]:
def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

def visualizeDataFrameDataset(xTestWithUHID,typeOfData):
    print('----------Inside visualizeDataFrameDataset--------')
    deathOrDischargeCase = None
    try:    
        currentFigure = None
        path = os.getcwd()       
        plt.figure()      
        for i in xTestWithUHID.uhid.unique():   
            data = xTestWithUHID[xTestWithUHID['uhid']==i] 

            if (data.iloc[0].dischargestatus == 1):
                deathOrDischargeCase = 'Death_Cases'
            elif (data.iloc[0].dischargestatus == 0):
                deathOrDischargeCase = 'Discharge_Cases'
            plt.rcParams["figure.figsize"] = (20,10)        
            #print('birthWeight=',birthWeight)
            fig, (ax1, ax2,ax3,ax4,ax5,ax6,hdpAX) = plt.subplots(7, sharex=True)
            fig.suptitle('Distributions for '+ i)
            print('-----DATA---=',data)
            x = range(0,len(data),1)
            print('-----X---=',x)
            if (typeOfData == 'cont'):
                ax1.set(xlabel='Minutes', ylabel='Continuous Signals')
                ax1.plot( x, 'spo2',   'b-', data=data)
                ax1.plot( x, 'pulserate', 'r-',   data=data, label='Heart Rate')
                ax1.legend(loc="upper right")
            elif (typeOfData == 'inter'):
                ax2.set(xlabel='Minutes', ylabel='Intake')
                ax3.set(xlabel='Minutes', ylabel='Output')
                ax4.set(xlabel='Minutes', ylabel='Vitals')
                ax5.set(xlabel='Minutes', ylabel='Ventilator')
                ax6.set(xlabel='Minutes', ylabel='Anthropometry')   
                data.currentdateweight = data.currentdateweight - birthWeight
                #print(data.currentdateweight)
                data.currentdateweight - birthWeight
                ax2.plot(  x, 'total_intake', 'r-', data=data)
                ax2.plot(  x, 'totalparenteralvolume','b-',  data=data)
                ax2.plot(  x, 'tpn-tfl', 'c-', data=data)

                ax3.plot(  x, 'stool_day_total', data=data, marker='', color='deepskyblue', linewidth=2)
                ax3.plot(  x, 'stool_passed', data=data, marker='', color='pink', linewidth=2)
                ax3.plot(  x, 'urine', data=data, marker='', color='slategray', linewidth=2)
                ax3.plot(  x, 'urine_per_hour', data=data, marker='', color='lime', linewidth=2)
                
                
                ax4.plot(  x, 'mean_bp', '-g', data=data, linewidth=2)
                ax4.plot(  x, 'rbs', '-r', data=data, linewidth=2)
                ax4.plot(  x, 'new_ph', '-b', data=data, linewidth=2)

                ax5.plot( x, 'fio2', data=data, marker='', color='darkgoldenrod', linewidth=2)
                ax5.plot( x, 'peep', data=data, marker='', color='royalblue', linewidth=2)
                ax5.plot( x, 'pip', data=data, marker='', color='olive', linewidth=2)

                ax6.plot( x, 'abd_difference_y', data=data, marker='', color='red', linewidth=2)
                ax6.plot( x, 'currentdateheight', data=data, marker='', color='skyblue', linewidth=2)
                ax6.plot( x, 'currentdateweight', data=data, marker='', color='green', linewidth=2,label='Weight diff from BW')
                ax2.legend(loc="upper right")
                ax3.legend(loc="upper right")
                ax4.legend(loc="upper right")    
                ax5.legend(loc="upper right")  
                ax6.legend(loc="upper right") 
            elif (typeOfData == 'fixed'):
                print('fixed data passed for visualization - strange')
            hdpAX.set(xlabel='Minutes', ylabel='HDP Prediction')
            currentFigure = hdpAX.get_figure()
            currentFigure.savefig(path+'/'+deathOrDischargeCase+'/'+str(i)+'_processed.png',dpi = 300)
        return True
    except Exception as e:
        print ('Exception in  visualizeDataFrameDataset--->',e)
        PrintException()
        return None



    except Exception as e:
        print('Exception in data visualization', e)
        PrintException()
        return False 

def visualizeDataset(fileName, folderName, uhid, caseType):
    try:    
        path = os.getcwd()
        #caseType = 'Death'
        #caseType = 'Discharge'
        seperator = '_'
        #deathCaseUHID = ["RNEH.0000008375", "RNEH.0000011301", "RNEH.0000012581", "RNEH.0000013713", "RSHI.0000012088", "RSHI.0000013287", "RSHI.0000014720"
        #, "RSHI.0000015178", "RSHI.0000015211", "RSHI.0000015691", "RSHI.0000016373", "RSHI.0000017471", "RSHI.0000017472", "RSHI.0000019707"
        #, "RSHI.0000021953", "RSHI.0000023451"]
        #deathCaseUHID = ["RSHI.0000023451"]
        data = pd.DataFrame()
        plt.rcParams["figure.figsize"] = (20,10)
        fileName = path+folderName+uhid+"/"+caseType+seperator+uhid+seperator+'intermediate_checkpoint_new_5.csv'
        #print (fileName)
        data = pd.read_csv(fileName)
        birthWeight = data.birthweight[0]
        #print('birthWeight=',birthWeight)
        fig, (ax1, ax2,ax3,ax4,ax5,ax6,hdpAX) = plt.subplots(7, sharex=True)
        fig.suptitle('Distributions for '+caseType+' '+ uhid)
        ax1.set(xlabel='Minutes', ylabel='Continuous Signals')
        ax2.set(xlabel='Minutes', ylabel='Intake')
        ax3.set(xlabel='Minutes', ylabel='Output')
        ax4.set(xlabel='Minutes', ylabel='Vitals')
        ax5.set(xlabel='Minutes', ylabel='Ventilator')
        ax6.set(xlabel='Minutes', ylabel='Anthropometry')   
        hdpAX.set(xlabel='Minutes', ylabel='HDP Prediction')

        x = data['Unnamed: 0']
        data.currentdateweight = data.currentdateweight - birthWeight
        #print(data.currentdateweight)
        data.currentdateweight - birthWeight
        ax1.plot( x, 'spo2',   'b-', data=data)
        ax1.plot( x, 'pulserate', 'r-',   data=data, label='Heart Rate')

        ax2.plot(  x, 'total_intake', 'r-', data=data)
        ax2.plot(  x, 'totalparenteralvolume','b-',  data=data)
        ax2.plot(  x, 'tpn-tfl', 'c-', data=data)

        ax3.plot(  x, 'stool_day_total', data=data, marker='', color='deepskyblue', linewidth=2)
        ax3.plot(  x, 'stool_passed', data=data, marker='', color='pink', linewidth=2)
        ax3.plot(  x, 'urine', data=data, marker='', color='slategray', linewidth=2)
        ax3.plot(  x, 'urine_per_hour', data=data, marker='', color='lime', linewidth=2)
        
        
        ax4.plot(  x, 'mean_bp', '-g', data=data, linewidth=2)
        ax4.plot(  x, 'rbs', '-r', data=data, linewidth=2)
        ax4.plot(  x, 'new_ph', '-b', data=data, linewidth=2)

        ax5.plot( x, 'fio2', data=data, marker='', color='darkgoldenrod', linewidth=2)
        ax5.plot( x, 'peep', data=data, marker='', color='royalblue', linewidth=2)
        ax5.plot( x, 'pip', data=data, marker='', color='olive', linewidth=2)

        ax6.plot( x, 'abd_difference_y', data=data, marker='', color='red', linewidth=2)
        ax6.plot( x, 'currentdateheight', data=data, marker='', color='skyblue', linewidth=2)
        ax6.plot( x, 'currentdateweight', data=data, marker='', color='green', linewidth=2,label='Weight diff from BW')
        
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        ax3.legend(loc="upper right")
        ax4.legend(loc="upper right")    
        ax5.legend(loc="upper right")  
        ax6.legend(loc="upper right") 
        #commented by HS for saving after drawing HDP later in the program
        #data preperation testing can uncomment this if no need to plot HDP
        #plt.savefig(path+folderName+uhid+'.png')
        return True,hdpAX
    except Exception as e:
        print('Exception in data visualization', e)
        PrintException()
        return False    