# Development-and-Validation-of-High-Definition-Phenotype-HDP-based-mortality-prediction-in-critical

Install the latest distribution of Anaconda
Additional Packages:
1.Tensorflow => 1.14.0
2.Plotly

-- Table 2: "Baseline characteristics of the population" is generated from Baseline_Table_2.ipynb using 'baseline_20th_june.csv'

-- Table 3a): "Ablation experiment for contribution of fixed, intermittent and continuous during mortality prediction" LSTM is generated from data_preparation.py,prediction_data_death_discharge.py, prediction_using_lstm.py,prediction_lrm.py using 'lstm_analysis.csv' and 'death_discharge_set.csv'. 

-- Table 3b): "Summary of LRM and LSTM mortality detection performance at different time points" is generated from Table_4.ipynb for LSTM using the files 'LSTM_X1_hour.csv' and 'LSTM_X2_week.csv'(X1 = 1,6,12,48 & X2=1,2,3,4) and Table_4_LRM.ipynb using the files 'LRM_X1_hour.csv' and 'LRM_X2_week.csv'. (X1 = 1,6,12,48 & X2=1,2,3,4) 

-- Supplementary eTable 6: "Comparison of CRIB (12 hours), CRIB II (1 hour) SNAP-II (12 hours), SNAPPE-II (12 hours), LRM and LSTM (48th)hour for predicting death and discharge." is generated from CRIB_eTable_6.ipynb, CRIB_2_eTable_6.ipynb and SNAPPE-2 SNAP-2_eTable_6.ipynb using the files 'crib.csv', 'crib_2.csv' ,'snap.csv' and 'snappe.csv'.

-- Supplementary Table 7: "Imputation distribution in intermittent and continuous parameters" is generated from Table_7.py using the file 'lstm_analysis.csv'

-- Supplementary Table 8: "Performance of individual continuous parameters using LSTM" is generated from Table_8.py using the file 'lstm_analysis.csv'

-- Supplementary Table 9 : "Performance of combination of continuous parameters using LSTM" is generated from Table_9.py using 'lstm_analysis.csv'.

-- Supplementary Table 10 : "eTable 10: Performance of individual intermittent parameters using LSTM" is generated from Table_10.py using 'lstm_analysis.csv'

-- Supplementary Table11: "Performance of individual intermittent parameters with fixed and continuous using LSTM" is generated from Table_11.py using 'lstm_analysis.csv'.

