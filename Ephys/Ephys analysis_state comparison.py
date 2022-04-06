# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:19:45 2021

@author: Fu-Ptacek Lab
"""
##############################################
##############################################

import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as st
from statannot import add_stat_annotation
import matplotlib


from LocalResources import dataWrangling as dw
from LocalResources import plottingOptions as po
from LocalResources import ephysProcessing as ep

matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
##############################################
data_dir = 'Data/'
score_len = 3 ## sleep score bout length in seconds
gross_filename = '3 mice spikesort combined.xlsx' ## filename with directions to spike and score files
fileName = data_dir + gross_filename
endRecording = 6 ##end of recording in hrs
recordingSleepBoutEnd = endRecording*60*60/score_len ## finds the sleep/wake bout the firing rate file stops
globalFont = 20 ## fontsize for plotting
z_score = True ## whether to z score the data
globalName = '3 mice combined zscore' ## attached to the end of files

data1 = pd.read_excel(fileName)
data2 = data1.values.tolist()
scores = dw.extract_column(data2, 1)
spikes = dw.extract_column(data2, 0)

wake_spikes, NREM_spikes, REM_spikes, all_spikes = ep.combineAllMice(scores, spikes,z_score,score_len,recordingSleepBoutEnd)
wake_spike_avg = []
for i in range(len(wake_spikes)):
    wake_spike_avg.append(np.mean(wake_spikes[i]))
NREM_spike_avg = []
for i in range(len(NREM_spikes)):
    NREM_spike_avg.append(np.mean(NREM_spikes[i]))    
REM_spike_avg = []
for i in range(len(REM_spikes)):
    REM_spike_avg.append(np.mean(REM_spikes[i])) 
    
wake_NREM_ttest = st.ttest_rel(wake_spike_avg,NREM_spike_avg)
wake_NREM_ttest = wake_NREM_ttest[1]
wake_REM_ttest = st.ttest_rel(wake_spike_avg,REM_spike_avg)
wake_REM_ttest = wake_REM_ttest[1]
NREM_REM_ttest = st.ttest_rel(NREM_spike_avg,REM_spike_avg)
NREM_REM_ttest = NREM_REM_ttest[1]
p_vals = [wake_NREM_ttest,wake_REM_ttest,NREM_REM_ttest] 
   
po.returnWakeStuff(wake_spike_avg,NREM_spike_avg,REM_spike_avg, globalFont, globalName)    
po.plotDots_3(p_vals, wake_spike_avg,'Wake',  NREM_spike_avg, 'NREM','Firing rate (z-score)','changes in f.r. zscore', REM_spike_avg, 'REM',globalFont)









