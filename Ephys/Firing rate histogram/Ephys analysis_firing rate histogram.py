# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:19:45 2021

@author: Fu-Ptacek Lab
"""
##############################################
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import scipy.stats as st
import matplotlib
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
##############################################
score_len = 3 ## sleep score bout length in seconds
gross_filename = '3 mice spikesort combined.xlsx' ## filename with directions to spike and score files
endRecording = 6 ##end of recording in hrs
recordingSleepBoutEnd = endRecording*60*60/score_len ## finds the sleep/wake bout the firing rate file stops
globalFont = 20 ## fontsize for plotting
z_score = False ## whether to z score the data
globalName = '3 mice combined zscore' ## attached to the end of files

def import_scores(name):
    ##this function takes the User scores from the file and returns a matrix of 
    ## [REM, REM, NREM, NREM, Wake, Wake, ...]
    with open(name, newline='') as inputfile:
        results = list(csv.reader(inputfile))
    scoress = np.asarray(results) 
    scores = scoress[1:len(scoress),4]
    for i in range(len(scores)):
        if scores[i] == 'Non REM':
            scores[i] = 'NREM'   
    num_bout = len(scores)
    scores = scores[0:num_bout]
    return scores

def extractColumnByLabel(a, columnName):
    ## extracts a column by the initial label in the column
    names = extract_row(a, 0)
    index = names.index(columnName)
    column = extract_column(a, index)
    return column

def extract_row(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a[0])):
        new_matrix.append(a[column][i])
    return new_matrix 
def extract_column(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(a[i][column])
    return new_matrix 

def singleScore(name, score, spikePerBout):
    ## find the bouts of that sleep state and returns the spikes per bout
    ## for every bout
    score_spikes = []
    for i in range(len(spikePerBout)):
        if name == score[i]:
            if spikePerBout[i] != 0:
                score_spikes.append(spikePerBout[i]/score_len)     
    return score_spikes

def combine_3(a, b, c):
    #combines 3 matrixes togther 
    d = []
    for i in range(len(a)):
        d.append(a[i])
    for i in range(len(b)):
        d.append(b[i])
    for i in range(len(c)):
        d.append(c[i])
    return d        

def avgSpikePerBout(score, spikePerBout):
    ## finds the average for a single spike
    REM_avg = singleScore('REM', score, spikePerBout)
    NREM_avg = singleScore('NREM', score, spikePerBout)
    wake_avg = singleScore('Wake', score, spikePerBout)
    all_avg = combine_3(wake_avg, NREM_avg, REM_avg)
    return all_avg

def analysisAllSpikes(scoreName, spikeName):
    ## just finds the average firing rate for all units
    wake_spikes_tot = []
    spikeTIMEPERBOUT = []
    spikes = pd.read_pickle(spikeName)
    spikez = spikes.values.tolist()
    score = import_scores(scoreName)
    score = score.tolist()
    score = score[0:int(recordingSleepBoutEnd+1)]
    for i in range(len(spikes)):
        print('spike #' + str(i) + ' of ' +str(len(spikez))+ ' in ' + scoreName)
        if z_score == True:
            spikez[i] = st.zscore(spikez[i])
        wake_spikes= avgSpikePerBout(score, spikez[i])
        wake_spikes_tot.append(wake_spikes)
        spikeTIMEPERBOUT.append(spikes[i])
    return wake_spikes_tot   

def combineAllMice(score, spike):
    ## combines total firing rate from all mice
    wake_all = []
    for i in range(len(score)):
        wake = analysisAllSpikes(score[i], spike[i])
        for j in range((len(wake))):
            wake_all.append(wake[j])            
    return wake_all
data1 = pd.read_excel(gross_filename)
data2 = data1.values.tolist()
scores = extract_column(data2, 1)
spikes = extract_column(data2, 0)
total_spikes = combineAllMice(scores, spikes)
total_spike_avg = []
for i in range(len(total_spikes)):
    total_spike_avg.append(np.mean(total_spikes[i]))
ax1 = plt.axes(frameon=False)
plt.hist(total_spike_avg, color = ['black'], density = False, bins = 15)
xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
plt.ylabel('# Units')
plt.xlabel('Firing rate (Hz)')
plt.tight_layout()
plt.savefig("firing rate hist.pdf")
plt.show()
