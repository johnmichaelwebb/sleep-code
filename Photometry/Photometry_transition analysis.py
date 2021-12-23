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
import matplotlib
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
##############################################

all_name = 'combinedFINAL' ## This is attached to all transition files from a previous analysis step
NREM2REM_name = all_name +'_NREM2REM.txt' ## filename for NREM-to-REM transitions
NREM2wake_name = all_name+'_NREM2wake.txt' ## filename for NREM-to-wake transitions
REM2wake_name = all_name +'_REM2wake.txt' ## filename for REM-to-wake transitions
REM2NREM_name = all_name+'_REM2NREM.txt' ## filename for REM-to-NREM transitions
wake2NREM_name = all_name+'_wake2NREM.txt' ## filename for wake-to-NREM transitions

zScore = True ##whether to z-score the date or plot raw values. If true, z-score the data
secBeforeAndAfter = 30 ## seconds to plot before and after the transition
laser_freq =1017.252625 ##laser collection frequency
globalName = '_' + all_name ## attached to all plots generated
globalFont = 13 ## font for figure plotting

def single_row_cleanup(a):
    ## returns floats for every entry in a single 1 x n matrix
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(float(a[i]))
    return new_matrix

def clean_up_imports(a):
    ## esentially returns floats for every entry in an n x n matrix 
    ## z-scores the results
    for i in range(len(a)):
        a[i] = single_row_cleanup(a[i])
    if zScore == True:
        for i in range(len(a)):
            a[i] = st.zscore(a[i])
    return a

def avg_animals(a):
    ##creates an average and standard deviation for matrix
    avg_matrix = []
    std_matrix = []
    for i in range(len(a[0])):
        curr_total = []
        for j in range(len(a)):
            curr_total.append(a[j][i])
        avg_matrix.append(np.mean(curr_total))
        std_matrix.append(st.sem(curr_total))
    return avg_matrix, std_matrix  

def create_sec(a):
    ## creates seconds based off the sampling frequency and the transition length
    new_matrix = []
    total_sec = int(laser_freq)
    for i in range(len(a)):
        new_matrix.append(i/total_sec)
    return new_matrix

def zero_transitions(a, amount = 10000):
    ## ensures each trial starts at 0
    subtract = np.mean(a[0:amount])
    for i in range(len(a)):
        a[i] = a[i] - subtract
    return a        

def plot_transitions(data,sem, sec, name, before, after):
    ## plots transtions. You feed in raw data (data), error (sem), a seconds matrix, 
    ## the title (name), the before state label (before) and after state label (after)
    beforeColor = ''
    afterColor = ''
    if before == 'REM':
        beforeColor = 'lightsalmon'
    elif before == 'NREM':
        beforeColor = 'lightskyblue'
    else:
        beforeColor = 'lemonchiffon'
    if after == 'REM':
        afterColor = 'lightsalmon'
    elif after == 'NREM':
        afterColor = 'lightskyblue'
    else:
        afterColor = 'lemonchiffon'    
    data = zero_transitions(data)
    ax1 = plt.axes(frameon=False)
    plt.errorbar(sec, data,sem, c = 'black', lw = 2, ecolor= 'dimgrey')
    # plt.xlabel('Time (sec)', fontsize = globalFont*2)
    plt.title(before + '                                  '+after, fontsize = globalFont)
    ax1.axvspan(0,secBeforeAndAfter, color = beforeColor)
    end = int(secBeforeAndAfter*2)
    ax1.axvspan(secBeforeAndAfter, end,  color = afterColor)
    ax1.tick_params(axis='x', which='major', labelsize=8*2)
    ax1.axes.get_yaxis().set_ticks([])
    ax1.axes.get_yaxis().set_visible(False)
    plt.xticks(ticks = [0,15,30,45, 60],labels = ['-30','-15','0','15','30'])
    ax1.xaxis.label.set_size(globalFont*2)
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    # plt.tight_layout()
    plt.savefig(name+globalName +'.pdf')
    plt.show()
    return

with open(NREM2REM_name, newline='') as inputfile:
    NREM2REM = list(csv.reader(inputfile))
with open(NREM2wake_name, newline='') as inputfile:
    NREM2wake = list(csv.reader(inputfile))
with open(REM2wake_name, newline='') as inputfile:
    REM2wake = list(csv.reader(inputfile))
with open(REM2NREM_name, newline='') as inputfile:
    REM2NREM = list(csv.reader(inputfile))    
with open(wake2NREM_name, newline='') as inputfile:
    wake2NREM = list(csv.reader(inputfile))    
NREM2REM = clean_up_imports(NREM2REM)
NREM2wake = clean_up_imports(NREM2wake)
REM2wake = clean_up_imports(REM2wake)
REM2NREM = clean_up_imports(REM2NREM)
wake2NREM = clean_up_imports(wake2NREM)
NREM2REM_avg, NREM2REM_std = avg_animals(NREM2REM)
NREM2wake_avg, NREM2wake_std = avg_animals(NREM2wake)
REM2wake_avg, REM2wake_std = avg_animals(REM2wake)
REM2NREM_avg, REM2NREM_std = avg_animals(REM2NREM)
wake2NREM_avg, wake2NREM_std = avg_animals(wake2NREM)
sec = create_sec(NREM2REM_avg)
plot_transitions(NREM2wake_avg,NREM2wake_std, sec, '_NREM to wake', 'NREM', 'Wake')
plot_transitions(REM2wake_avg,REM2wake_std, sec, 'REM to wake', 'REM', 'Wake')
plot_transitions(REM2NREM_avg,REM2NREM_std, sec, 'REM to NREM', 'REM', 'NREM')
plot_transitions(wake2NREM_avg,wake2NREM_std, sec, 'Wake to NREM', 'Wake', 'NREM')
plot_transitions(NREM2REM_avg,NREM2REM_std, sec, '_NREM to REM', 'NREM', 'REM')