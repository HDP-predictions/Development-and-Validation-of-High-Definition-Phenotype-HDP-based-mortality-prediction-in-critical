# Development-and-Validation-of-High-Definition-Phenotype-HDP-based-mortality-prediction-in-critical

Install the latest distribution of Anaconda
Additional Packages:
1.Tensorflow => 1.14.0
2.Plotly

-- Table 2: "Baseline characteristics of the population" is generated from Baseline_Table_2.ipynb

-- Table 3: "Ablation experiment for contribution of fixed, intermittent and continuous during mortality prediction" is generated from LSTM_ablation_Table_3.ipynb and LRM_ablation_Table_3.ipynb

-- Table 4: "Summary of NSIS (LRM & LSTM) mortality detection performance at different time points" is generated from X1_hour_LSTM_Table_4.ipynb, X1_hour_LRM_Table_4.ipynb, X2_week_LSTM_Table_4.ipynb and X2_week_LRM_Table_4.ipynb. (X1 = 1,6,12,48 & X2=1,2,3,4) 

-- Supplementary eTable 6: "Comparison of CRIB (12 hours), CRIB II (1 hour) SNAP-II (12 hours), SNAPPE-II (12 hours), NSIS (LRM &LSTM) (48th)hour for predicting death and discharge." is generated from CRIB_eTable_6.ipynb, CRIB_2_eTable_6.ipynb and SNAPPE-2 SNAP-2_eTable_6.ipynb

-- Supplementary Figure S2(a) and S2(b): "Medication deviation across gestation group" is generated from Medication bar chart.R

-- Supplementary Figure S3 : "Clinical diagnosis distribution across gestational age groups" is generated from Morbidity stacks.R

-- Supplementary Figure S4(a,b,c,d): "Patient frequency vs LOS for various gestation categories" is generated from Supplementary 4.R

-- Supplementary Table S2(a,b,c): "Medicines deviation statistics" is generated from Medication tables.R

Steps to successfully run individual scripts: Install and import the required packages - tibble, lsmeans, dplyr, relimp, rstudioapi, readxl, multcomp, qpcR, xquartz, ggplot2, hrbrthemes, viridis, tidyverse

Scripts
Download the Training files of Site 1 and Site 2 for both Neofax and Nutrition (These files are created from JAVA code - EsphaganErrors.java and NeofaxErrors.java)
Define the path in Baseline Table 1.R, Final_table.R and in other R files.
Run the script and respective Tables and Figures will be generated.
