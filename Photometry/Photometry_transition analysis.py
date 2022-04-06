#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 10:17:01 2019

@author: jmw
"""

############################################
import csv
import matplotlib
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
from LocalResources import dataWrangling as dw
from LocalResources import plottingOptions as po
##############################################
data_dir = 'Data/'
all_name = 'samp_photo' ## This is attached to all transition files from a previous analysis step
NREM2REM_name = data_dir+all_name +'_NREM2REM.txt' ## filename for NREM-to-REM transitions
NREM2wake_name = data_dir+all_name+'_NREM2wake.txt' ## filename for NREM-to-wake transitions
REM2wake_name = data_dir+all_name +'_REM2wake.txt' ## filename for REM-to-wake transitions
wake2NREM_name =data_dir+ all_name+'_wake2NREM.txt' ## filename for wake-to-NREM transitions

zScore = True ##whether to z-score the date or plot raw values. If true, z-score the data
secBeforeAndAfter = 30 ## seconds to plot before and after the transition
laser_freq =1017.252625 ##laser collection frequency
globalName = '_' + all_name ## attached to all plots generated
globalFont = 13 ## font for figure plotting


with open(NREM2REM_name, newline='') as inputfile:
    NREM2REM = list(csv.reader(inputfile))
with open(NREM2wake_name, newline='') as inputfile:
    NREM2wake = list(csv.reader(inputfile))
with open(REM2wake_name, newline='') as inputfile:
    REM2wake = list(csv.reader(inputfile))   
with open(wake2NREM_name, newline='') as inputfile:
    wake2NREM = list(csv.reader(inputfile))    
NREM2REM = dw.clean_up_imports(NREM2REM, zScore)
NREM2wake = dw.clean_up_imports(NREM2wake,zScore)
REM2wake = dw.clean_up_imports(REM2wake,zScore)
wake2NREM = dw.clean_up_imports(wake2NREM,zScore)



NREM2REM_avg, NREM2REM_std = dw.avg_animals(NREM2REM)
NREM2wake_avg, NREM2wake_std = dw.avg_animals(NREM2wake)
REM2wake_avg, REM2wake_std = dw.avg_animals(REM2wake)
wake2NREM_avg, wake2NREM_std = dw.avg_animals(wake2NREM)
sec = dw.create_sec(NREM2REM_avg, laser_freq)
po.plot_transitions(NREM2wake_avg,NREM2wake_std, sec, '_NREM to wake', 'NREM', 'Wake',secBeforeAndAfter, globalFont, globalName)
po.plot_transitions(REM2wake_avg,REM2wake_std, sec, 'REM to wake', 'REM', 'Wake',secBeforeAndAfter, globalFont, globalName)
po.plot_transitions(wake2NREM_avg,wake2NREM_std, sec, 'Wake to NREM', 'Wake', 'NREM',secBeforeAndAfter, globalFont, globalName)
po.plot_transitions(NREM2REM_avg,NREM2REM_std, sec, '_NREM to REM', 'NREM', 'REM',secBeforeAndAfter, globalFont, globalName)