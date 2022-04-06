#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:25:36 2022

@author: jwebb2020
"""
import numpy as np
import pandas as pd
import scipy.stats as st
import csv
from LocalResources import dataWrangling as dw
data_dir = 'Data/'

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

def singleScore(name, score, spikePerBout,score_len):
    ## find the bouts of that sleep state and returns the spikes per bout
    ## for every bout
    score_spikes = []
    for i in range(len(spikePerBout)):
        if name == score[i]:
            if spikePerBout[i] != 0:
                score_spikes.append(spikePerBout[i]/score_len)     
    return score_spikes

def avgSpikePerBout_totalFR(score, spikePerBout,score_len):
    ## finds the average for a single spike
    REM_avg = singleScore('REM', score, spikePerBout,score_len)
    NREM_avg = singleScore('NREM', score, spikePerBout,score_len)
    wake_avg = singleScore('Wake', score, spikePerBout,score_len)
    all_avg = dw.combine_3(wake_avg, NREM_avg, REM_avg)
    return all_avg

def analysisAllSpikes_totalFR(scoreName, spikeName,z_score,score_len,recordingSleepBoutEnd):
    ## just finds the average firing rate for all units
    wake_spikes_tot = []
    spikeTIMEPERBOUT = []
    spikes = pd.read_pickle(data_dir+spikeName)
    spikez = spikes.values.tolist()
    score = import_scores(data_dir+scoreName)
    score = score.tolist()
    score = score[0:int(recordingSleepBoutEnd+1)]
    for i in range(len(spikes)):
        print('spike #' + str(i) + ' of ' +str(len(spikez))+ ' in ' + scoreName)
        if z_score == True:
            spikez[i] = st.zscore(spikez[i])
        wake_spikes= avgSpikePerBout_totalFR(score, spikez[i],score_len)
        wake_spikes_tot.append(wake_spikes)
        spikeTIMEPERBOUT.append(spikes[i])
    return wake_spikes_tot   

def combineAllMice_totalFR(score, spike,z_score,score_len,recordingSleepBoutEnd):
    ## combines total firing rate from all mice
    wake_all = []
    for i in range(len(score)):
        wake = analysisAllSpikes_totalFR(score[i], spike[i],z_score,score_len,recordingSleepBoutEnd)
        for j in range((len(wake))):
            wake_all.append(wake[j])            
    return wake_all

def avgSpikePerBout(score, spikePerBout,score_len):
    ## finds the average firing rate for every sleep state
    REM_avg = singleScore('REM', score, spikePerBout,score_len)
    NREM_avg = singleScore('NREM', score, spikePerBout,score_len)
    wake_avg = singleScore('Wake', score, spikePerBout,score_len)
    return wake_avg, NREM_avg, REM_avg

def analysisAllSpikes(scoreName, spikeName,z_score,score_len,recordingSleepBoutEnd):
    ## puts the spikes into a bout-by-bout matrix of the same size as the scored sleep
    wake_spikes_tot = []
    NREM_spikes_tot = []
    REM_spikes_tot = []
    spikeTIMEPERBOUT = []
    spikes = pd.read_pickle(data_dir+spikeName)
    spikez = spikes.values.tolist()
    score = import_scores(data_dir+scoreName)
    score = score.tolist()
    score = score[0:int(recordingSleepBoutEnd+1)]
    for i in range(len(spikes)):
        print('spike #' + str(i) + ' of ' +str(len(spikez))+ ' in ' + scoreName)
        if z_score == True:
            spikez[i] = st.zscore(spikez[i])
        wake_spikes, NREM_spikes, REM_spikes = avgSpikePerBout(score, spikez[i],score_len)
        wake_spikes_tot.append(wake_spikes)
        NREM_spikes_tot.append(NREM_spikes)
        REM_spikes_tot.append(REM_spikes)
        spikeTIMEPERBOUT.append(spikes[i])
    return wake_spikes_tot,NREM_spikes_tot,REM_spikes_tot, spikeTIMEPERBOUT  


def combineAllMice(score, spike,z_score,score_len,recordingSleepBoutEnd):
    ## combines the spikes from all mice
    rem_all = []
    nrem_all = []
    wake_all = []
    all_spikes = []
    for i in range(len(score)):
        wake, nrem, rem, all_s = analysisAllSpikes(score[i], spike[i],z_score,score_len,recordingSleepBoutEnd)
        for j in range((len(wake))):
            wake_all.append(wake[j])
            nrem_all.append(nrem[j])
            rem_all.append(rem[j])
            all_spikes.append(all_s[j])
    return wake_all, nrem_all, rem_all, all_spikes