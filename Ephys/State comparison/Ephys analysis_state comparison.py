# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:19:45 2021

@author: Fu-Ptacek Lab
"""
##############################################

import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as st
from statannot import add_stat_annotation
##############################################
score_len = 3 ## sleep score bout length in seconds
gross_filename = '3 mice spikesort combined.xlsx'
endRecording = 6 ## end of recording in hrs; unsed because usually not all the scored sleep is used
recordingSleepBoutEnd = endRecording*60*60/score_len ## finds the sleep/wake bout the firing rate file stops
globalFont = 20 ## fontsize for plotting
z_score = True ## whether or not to z-score the data
globalName = '3 mice combined zscore' # #attached to all the files generated
##############################################
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

def avgSpikePerBout(score, spikePerBout):
    ## finds the average firing rate for every sleep state
    REM_avg = singleScore('REM', score, spikePerBout)
    NREM_avg = singleScore('NREM', score, spikePerBout)
    wake_avg = singleScore('Wake', score, spikePerBout)
    return wake_avg, NREM_avg, REM_avg

def analysisAllSpikes(scoreName, spikeName):
    ## puts the spikes into a bout-by-bout matrix of the same size as the scored sleep
    wake_spikes_tot = []
    NREM_spikes_tot = []
    REM_spikes_tot = []
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
        wake_spikes, NREM_spikes, REM_spikes = avgSpikePerBout(score, spikez[i])
        wake_spikes_tot.append(wake_spikes)
        NREM_spikes_tot.append(NREM_spikes)
        REM_spikes_tot.append(REM_spikes)
        spikeTIMEPERBOUT.append(spikes[i])
    return wake_spikes_tot,NREM_spikes_tot,REM_spikes_tot, spikeTIMEPERBOUT    

def combineAllMice(score, spike):
    ## combines the spikes from all mice
    rem_all = []
    nrem_all = []
    wake_all = []
    all_spikes = []
    for i in range(len(score)):
        wake, nrem, rem, all_s = analysisAllSpikes(score[i], spike[i])
        for j in range((len(wake))):
            wake_all.append(wake[j])
            nrem_all.append(nrem[j])
            rem_all.append(rem[j])
            all_spikes.append(all_s[j])
    return wake_all, nrem_all, rem_all, all_spikes

def returnWakeStuff(wake, NREM, REM):
    ## creates the wake-NREM and NREM-REM plot
    wakeMinusNREM = []
    REMMinusNREM = []
    for i in range(len(wake)):
        wakeMinusNREM.append(wake[i]-NREM[i])
        REMMinusNREM.append(REM[i]-NREM[i])  
    d = {'wIn':wakeMinusNREM, 'rIn':REMMinusNREM}
    df = pd.DataFrame(data=d)
    sns.set(font_scale = 1.5)
    sns.set_style('white')
    plt.tight_layout()
    
    g = sns.JointGrid(data=df, 
                      x="wIn",
                      y="rIn", )
    plt.tight_layout()
    g.plot_joint(sns.scatterplot, s = 70,color='none', edgecolor='dodgerblue', linewidth =1.5)
    # plt.xlim(0,.1)

    plt.axvline(x=0, c= 'black')
    plt.axhline(y=0, c = 'black')
    plt.xlabel('Wake - NREM (Z-score)', fontsize = globalFont)
    plt.ylabel( 'REM - NREM (Z-score)', fontsize = globalFont)
    plt.tight_layout()
    
    g.plot_marginals(sns.kdeplot, linewidth =4, color = 'dodgerblue')    
    plt.tight_layout()
    plt.yticks(ticks = [-0.05, 0, 0.05],labels = ['-0.05','0','0.05'])
    # plt.xlim(0,.1)
    plt.savefig('wake vs NREM vs REM_'  +globalName +'.pdf')
    plt.show()
    return

def createDataFrame(frame, data, name, index):
    ## creates a pd dataframe given the input variables 
    for i in range(len(data)):
        frame[name][i+index] = data[i]
    return frame

def plotDots_3(pvalues,wake,wtName, NREM,mutName, ylabel,name, REM, mut2name):
    labels = []
    for i in range(len(wake)):
        labels.append(wtName)
    for i in range(len(NREM)):
        labels.append(mutName)
    for i in range(len(REM)):
        labels.append(mut2name)  
    height = len(wake) + len(NREM) +len(REM)
    framee = pd.DataFrame(np.random.randint(low=0.0, high=8, size=(height, 2)), columns=['cFos', 'condition'],dtype = np.float64)
    framee = createDataFrame(framee, wake, 'cFos', 0)
    framee = createDataFrame(framee, NREM, 'cFos', len(wake))
    framee = createDataFrame(framee, REM, 'cFos', len(wake)+len(NREM))
    framee = createDataFrame(framee, labels, 'condition',0)
    sns.set_style("ticks")
    sns.set(font_scale = 1.7)
    sns.set_style("ticks")
    order = [wtName, mutName, mut2name]
    test_short_name = 'ttest'
    g2 = sns.boxplot(x="condition", y="cFos", data=framee, palette = ["lemonchiffon",'lightblue','lightsalmon'], linewidth = .5 )
    # test_results = add_stat_annotation(g2, data=framee, x='condition', y='cFos', order=order,
    #                                box_pairs=[(wtName, mutName),(wtName, mut2name), (mutName, mut2name)],
    #                               perform_stat_test=False, pvalues=pvalues, test_short_name=test_short_name,
    #                                text_format='star', verbose=2)
    for i in range(len(wake)):
        indexes = [i, len(wake)*1 +i, len(wake)*2+i]
        curr_line = framee.iloc[indexes]
        sns.lineplot(x = 'condition', y = 'cFos',data = curr_line, sort = False, color = 'black', linewidth = 1)
    sns.despine()
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont)
    # plt.title(name, fontsize = globalFont)
    plt.savefig(name +".pdf",bbox_inches="tight")
    plt.show() 
    return framee

data1 = pd.read_excel(gross_filename)
data2 = data1.values.tolist()
scores = extract_column(data2, 1)
spikes = extract_column(data2, 0)
wake_spikes, NREM_spikes, REM_spikes, all_spikes = combineAllMice(scores, spikes)
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
returnWakeStuff(wake_spike_avg,NREM_spike_avg,REM_spike_avg)    
plotDots_3(p_vals, wake_spike_avg,'Wake',  NREM_spike_avg, 'NREM','Firing rate (z-score)','changes in f.r. zscore', REM_spike_avg, 'REM')