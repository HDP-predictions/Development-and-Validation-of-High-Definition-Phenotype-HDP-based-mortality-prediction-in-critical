# Development-and-Validation-of-High-Definition-Phenotype-HDP-based-mortality-prediction-in-critical

Install the latest distribution of Anaconda
Additional Packages:
1.Tensorflow => 1.14.0
2.Plotly

-- Table 2: "Baseline characteristics of the population" is generated from Baseline_Table_2.ipynb

-- Table 3: "Ablation experiment for contribution of fixed, intermittent and continuous during mortality prediction" is generated from LSTM_ablation_Table_3.ipynb and LRM_ablation_Table_3.ipynb using files 'data_8th_june.csv' and 'data_stat.csv'. 

-- Table 4: "Summary of NSIS (LRM & LSTM) mortality detection performance at different time points" is generated from X1_hour_LSTM_Table_4.ipynb, X1_hour_LRM_Table_4.ipynb, X2_week_LSTM_Table_4.ipynb and X2_week_LRM_Table_4.ipynb using the files 'LRM_X1_hour.csv', 'LRM_X2_week.csv', 'LSTM_X1_hour.csv' and 'LSTM_X2_week.csv'. (X1 = 1,6,12,48 & X2=1,2,3,4) 

-- Supplementary eTable 6: "Comparison of CRIB (12 hours), CRIB II (1 hour) SNAP-II (12 hours), SNAPPE-II (12 hours), NSIS (LRM &LSTM) (48th)hour for predicting death and discharge." is generated from CRIB_eTable_6.ipynb, CRIB_2_eTable_6.ipynb and SNAPPE-2 SNAP-2_eTable_6.ipynb using the files 'crib.csv', 'crib_2.csv' ,'snap.csv' and 'snappe.csv'.

-- Supplementary eTable 8: "Performance of individual continuous parameters using LSTM" is generated from eTable_8.ipynb using the file 'data_8th_june.csv'

-- Supplementary eTable 9 : "Performance of combination of continuous parameters using LSTM" is generated from eTable_9.csv using 'data_8th_june.csv'.

-- Supplementary eTable11: "Performance of individual intermittent parameters with fixed and continuous using LSTM" is generated from xx_eTable_11.ipynb using 'data_8th_june.csv'.

-- Supplementary Table S2(a,b,c): "Medicines deviation statistics" is generated from Medication tables.R

Steps to successfully run individual scripts: Install and import the required packages - tibble, lsmeans, dplyr, relimp, rstudioapi, readxl, multcomp, qpcR, xquartz, ggplot2, hrbrthemes, viridis, tidyverse

Scripts
Download the Training files of Site 1 and Site 2 for both Neofax and Nutrition (These files are created from JAVA code - EsphaganErrors.java and NeofaxErrors.java)
Define the path in Baseline Table 1.R, Final_table.R and in other R files.
Run the script and respective Tables and Figures will be generated.
