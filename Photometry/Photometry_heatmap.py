#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 10:17:01 2019

@author: jmw
"""

############################################
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.stats as st
import seaborn as sns
import seaborn
import matplotlib
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
from LocalResources import dataWrangling as dw
from LocalResources import plottingOptions as po
##############################################
data_dir = 'Data/'
all_name = 'samp_photo' ## This is attached to all transition files from a previous analysis step
NREM2REM_name = data_dir+ all_name +'_NREM2REM.txt' ## filename for NREM-to-REM transitions
NREM2wake_name = data_dir+all_name+'_NREM2wake.txt' ## filename for NREM-to-wake transitions
REM2wake_name = data_dir+all_name +'_REM2wake.txt' ## filename for REM-to-wake transitions
wake2NREM_name =data_dir+ all_name+'_wake2NREM.txt' ## filename for wake-to-NREM transitions
zScore = True ##whether to z-score the date or plot raw values. If true, z-score the data
secBeforeAndAfter = 30 ## seconds to plot before and after the transition
laser_freq =1017.252625 ##laser collection frequency
globalName = '_' + all_name ## attached to all plots generated
globalFont = 13 ## font for figure plotting
initialize = True ## if True, loads the data initially
secPerDownsample = .5 ## the sampling frequency is high, so this bins the data by seconds to downsample for plotting
secondsTick = 15 ## tickmarks for the x-axis
trialTick = 50 ## tickmarks for the y-axis
##############################################




with open(NREM2REM_name, newline='') as inputfile:
    NREM2REM = list(csv.reader(inputfile))
with open(NREM2wake_name, newline='') as inputfile:
    NREM2wake = list(csv.reader(inputfile))
with open(REM2wake_name, newline='') as inputfile:
    REM2wake = list(csv.reader(inputfile))
with open(wake2NREM_name, newline='') as inputfile:
    wake2NREM = list(csv.reader(inputfile))    

NREM2REM = dw.clean_up_imports(NREM2REM,zScore)
NREM2wake = dw.clean_up_imports(NREM2wake,zScore)
REM2wake = dw.clean_up_imports(REM2wake,zScore)
wake2NREM = dw.clean_up_imports(wake2NREM,zScore)
NREM2REM_downsample = dw.downsampleAll(NREM2REM,secBeforeAndAfter,secPerDownsample)
NREM2wake_downsample = dw.downsampleAll(NREM2wake,secBeforeAndAfter,secPerDownsample)
REM2wake_downsample = dw.downsampleAll(REM2wake,secBeforeAndAfter,secPerDownsample)
wake2NREM_downsample = dw.downsampleAll(wake2NREM,secBeforeAndAfter,secPerDownsample)
po.create_heatmap(NREM2REM_downsample, 'NREM to REM',globalName, secPerDownsample,secBeforeAndAfter,secondsTick,trialTick)
po.create_heatmap(NREM2wake_downsample, 'NREM to Wake',globalName, secPerDownsample,secBeforeAndAfter,secondsTick,trialTick)
po.create_heatmap(REM2wake_downsample, 'REM to Wake',globalName, secPerDownsample,secBeforeAndAfter,secondsTick,trialTick)
po.create_heatmap(wake2NREM_downsample, 'Wake to NREM',globalName, secPerDownsample,secBeforeAndAfter,secondsTick,trialTick)