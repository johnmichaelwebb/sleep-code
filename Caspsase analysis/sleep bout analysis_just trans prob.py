#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 19:10:45 2019

@author: jmw
"""




############################################
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.ticker import FuncFormatter
import matplotlib
import pylab as p
import scipy.stats as st
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from scipy.signal import periodogram
from matplotlib.lines import Line2D
from six import iteritems
from sklearn.metrics import mean_squared_error as mean_squared_error
from math import sqrt
import random
import seaborn as sns
import pandas as pd


from LocalResources import dataWrangling as dw
from LocalResources import plottingOptions as po
from LocalResources import sleepBoutProcessing as sbp

##############################################
##############################################
##############################################
##############################################
data_dir = 'Data/'
name_score = 'pTRN EEG scores.csv' #user scored epochs
filename = data_dir + name_score
max_bout = 8640 ## the maximum number of bouts in the file 
days = 2 ## number of days per mouse
bout_length = 10 ##number of seconds per bout
minutes_per_display = 60
globalFont = 15
hours = 24
STARTLIGHT = 0    
ENDLIGHT = 12

global_name ='_caspase EEG'
WT_NAME = 'WT'
MUTANT_NAME = 'Cre'



## Initialize the data
with open(filename, newline='\n' ) as inputfile:
   data = list(csv.reader(inputfile)) 
   data[0][0] = data[0][0][1:len(data[0][0])]
names = dw.extract_row(data,1)
Cre = dw.create_empty_matrix(names, 'Cre', data)
WT = dw.create_empty_matrix(names, 'WT',data)
wild = WT[:]
wildy = WT[:]
mutant = Cre[:]
mutanty = Cre[:]
mut_wake_to_wake, mut_wake_to_NREM, mut_wake_to_REM, mut_NREM_to_wake, mut_NREM_to_NREM, mut_NREM_to_REM, mut_REM_to_wake, mut_REM_to_NREM, mut_REM_to_REM = sbp.prob_state_all(Cre)
wt_wake_to_wake, wt_wake_to_NREM, wt_wake_to_REM, wt_NREM_to_wake, wt_NREM_to_NREM, wt_NREM_to_REM, wt_REM_to_wake, wt_REM_to_NREM, wt_REM_to_REM = sbp.prob_state_all(WT)



## Calculate the transition state probability
po.plotDots_stats(wt_wake_to_wake, WT_NAME, mut_wake_to_wake, MUTANT_NAME, 'Transition prob (%)', 'Wake to wake',globalFont, global_name)
wake_to_wake_ttest = st.ttest_ind(wt_wake_to_wake,mut_wake_to_wake, equal_var = False)
wake_to_wake_ttest = wake_to_wake_ttest[1]

po.plotDots_stats(wt_wake_to_NREM, WT_NAME, mut_wake_to_NREM, MUTANT_NAME, 'Transition prob (%)', 'Wake to NREM',globalFont, global_name)
wake_to_NREM_ttest = st.ttest_ind(wt_wake_to_NREM,mut_wake_to_NREM, equal_var = False)
wake_to_NREM_ttest = wake_to_NREM_ttest[1]

po.plotDots_stats(wt_wake_to_REM, WT_NAME, mut_wake_to_REM, MUTANT_NAME, 'Transition prob (%)', 'Wake to REM', globalFont,global_name)
wake_to_REM_ttest = st.ttest_ind(wt_wake_to_REM,mut_wake_to_REM, equal_var = False)
wake_to_REM_ttest = wake_to_REM_ttest[1]

po.plotDots_stats(wt_NREM_to_REM, WT_NAME, mut_NREM_to_REM, MUTANT_NAME, 'Transition prob (%)', 'NREM to REM',globalFont, global_name)
NREM_to_REM_ttest = st.ttest_ind(wt_NREM_to_REM,mut_NREM_to_REM, equal_var = False)
NREM_to_REM_ttest = NREM_to_REM_ttest[1]

po.plotDots_stats(wt_NREM_to_NREM, WT_NAME, mut_NREM_to_NREM, MUTANT_NAME, 'Transition prob (%)', 'NREM to NREM',globalFont, global_name)
NREM_to_NREM_ttest = st.ttest_ind(wt_NREM_to_NREM,mut_NREM_to_NREM, equal_var = False)
NREM_to_NREM_ttest = NREM_to_NREM_ttest[1]

po.plotDots_stats(wt_NREM_to_wake, WT_NAME, mut_NREM_to_wake, MUTANT_NAME, 'Transition prob (%)', 'NREM to Wake',globalFont, global_name)
NREM_to_wake_ttest = st.ttest_ind(wt_NREM_to_wake,mut_NREM_to_wake, equal_var = False)
NREM_to_wake_ttest = NREM_to_wake_ttest[1]

po.plotDots_stats(wt_REM_to_wake, WT_NAME, mut_REM_to_wake, MUTANT_NAME, 'Transition prob (%)', 'REM to Wake',globalFont, global_name)
REM_to_wake_ttest = st.ttest_ind(wt_REM_to_wake,mut_REM_to_wake, equal_var = False)
REM_to_wake_ttest = REM_to_wake_ttest[1]

po.plotDots_stats(wt_REM_to_REM, WT_NAME, mut_REM_to_REM, MUTANT_NAME, 'Transition prob (%)', 'REM to REM', globalFont,global_name)
REM_to_REM_ttest = st.ttest_ind(wt_REM_to_REM,mut_REM_to_REM, equal_var = False)
REM_to_REM_ttest = REM_to_REM_ttest[1]

po.plotDots_stats(wt_REM_to_NREM, WT_NAME, mut_REM_to_NREM, MUTANT_NAME, 'Transition prob (%)', 'REM to NREM', globalFont,global_name)
REM_to_NREM_ttest = st.ttest_ind(wt_REM_to_NREM,mut_REM_to_NREM, equal_var = False)
REM_to_NREM_ttest = REM_to_NREM_ttest[1]


