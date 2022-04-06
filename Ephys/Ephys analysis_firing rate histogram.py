# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:19:45 2021

@author: Fu-Ptacek Lab
"""
##############################################
import numpy as np
import pandas as pd
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
z_score = False ## whether to z score the data
globalName = '3 mice combined zscore' ## attached to the end of files

      


data1 = pd.read_excel(fileName)
data2 = data1.values.tolist()
scores = dw.extract_column(data2, 1)
spikes = dw.extract_column(data2, 0)
total_spikes = ep.combineAllMice_totalFR(scores, spikes,z_score,score_len,recordingSleepBoutEnd)
total_spike_avg = []
for i in range(len(total_spikes)):
    total_spike_avg.append(np.mean(total_spikes[i]))
    
po.plot_firingRateHist(total_spike_avg)
    

