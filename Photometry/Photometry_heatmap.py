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
initialize = True ## if True, loads the data initially
secPerDownsample = .5 ## the sampling frequency is high, so this bins the data by seconds to downsample for plotting
secondsTick = 15 ## tickmarks for the x-axis
trialTick = 50 ## tickmarks for the y-axis
##############################################

def single_row_cleanup(a):
    ## returns floats for every entry in a single 1 x n matrix
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(float(a[i]))
    return new_matrix

def extract_column(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(a[i][column])
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
        std_matrix.append(np.std(curr_total))
    return avg_matrix, std_matrix  

def create_sec(a):
    ## creates seconds based off the sampling frequency and the transition length
    
    new_matrix = []
    total_sec = int(laser_freq)
    for i in range(len(a)):
        new_matrix.append(i/total_sec)
    return new_matrix

def downsampleSingle(a):
    ## downsamples photometry transition for a single transition
    new_matrix = []
    numTrials = secBeforeAndAfter*2/secPerDownsample
    numTrialsInt = int(numTrials)
    samplesPerBout = len(a)/numTrials
    for i in range(numTrialsInt):
        start = int(i*samplesPerBout)
        stop = int((i+1)*samplesPerBout)
        currDownsample = a[start:stop]
        currMean = np.mean(currDownsample)
        new_matrix.append(currMean)
    return new_matrix

def downsampleAll(a):
    ## downsamples the photometry transitions for the transitions
    new_matrix = []
    for i in range(len(a)):
        curr_downsample = downsampleSingle(a[i])
        new_matrix.append(curr_downsample)
    return new_matrix

def create_x_axis(a):
    ## creates the x-axis for the heatmap
    new_matrix = []
    num_values = int(secBeforeAndAfter*2/secondsTick)
    include = []
    for i in range(num_values):
        include.append(i*secondsTick)
    shiftedValues = []
    for i in range(len(include)):
        shiftedValues.append(include[i]-secBeforeAndAfter)
    curr_sec =[]
    for i in range(len(a)):
        curr_sec.append(i*secPerDownsample)
    counter = 0
    for i in range(len(curr_sec)):
        if curr_sec[i] in include:
            new_matrix.append(shiftedValues[counter])
            counter+=1
        else:
            new_matrix.append('')
    last = len(new_matrix) -1
    new_matrix[last] = secBeforeAndAfter
    return new_matrix

def create_y_axis(a):
    ## creates the y-axis for the heatmap
    new_matrix = []
    num_labels = int(len(a)/trialTick)
    curr_trial = []
    include = []
    for i in range(num_labels):
        include.append((i+1)*trialTick)
    for i in range(len(a)):
        curr_trial.append(i+1)
    counter = 0
    for i in range(len(curr_trial)):
        if curr_trial[i] in include:
            new_matrix.append(include[counter])
            counter+=1
        else:
            new_matrix.append('')
    return new_matrix

def create_heatmap(a, name): 
    ## creates the heatmap for a single transition type. 'a' is the data; name is the savename
    sns.color_palette("Blues")
    x_labels = create_x_axis(a[1])
    y_labels = create_y_axis(a)
    ax = seaborn.heatmap(a,xticklabels = x_labels, yticklabels = y_labels, cmap = 'Blues',vmin = -2, vmax = 4, cbar_kws = {'label':'z-scored ∆F/F'})
    ax.tick_params(axis='both', which='both', length=0)
    plt.title(name)
    plt.xlabel('Time (sec)')
    plt.ylabel('Trial #')
    plt.tight_layout()
    plt.savefig(name + '_'+globalName+'_.pdf')
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
NREM2REM_downsample = downsampleAll(NREM2REM)
NREM2wake_downsample = downsampleAll(NREM2wake)
REM2wake_downsample = downsampleAll(REM2wake)
REM2NREM_downsample = downsampleAll(REM2NREM)
wake2NREM_downsample = downsampleAll(wake2NREM)
create_heatmap(NREM2REM_downsample, 'NREM to REM')
create_heatmap(NREM2wake_downsample, 'NREM to Wake')
create_heatmap(REM2wake_downsample, 'REM to Wake')
create_heatmap(REM2NREM_downsample, 'REM to NREM')
create_heatmap(wake2NREM_downsample, 'Wake to NREM')