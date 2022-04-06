#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 08:58:11 2022

@author: jwebb2020
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib
import scipy.stats as st
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from statannot import add_stat_annotation

from LocalResources import dataWrangling as dw
from LocalResources import plottingOptions as po
from LocalResources import sleepBoutProcessing as sbp

trials_per_animal = 50


def calc_perc(a, name):
    counter = 0
    for i in range(len(a)):
        if a[i] == name:
            counter +=1
    perc = counter/len(a)*100
    return perc

def calc_percent_tot(a):
    wake_perc = []
    NREM_perc= []
    REM_perc = []
    new_matrix = []
    # if len(a) ==1:
    #     for i in range(len(NREM_avg)):
    #         new_matrix.append(0)
    #     return new_matrix, new_matrix, new_matrix
    for i in range(len(a[0])):
        column = dw.extract_column(a, i)
        curr_wake = calc_perc(column, 'Wake')
        curr_NREM = calc_perc(column, 'NREM')
        curr_REM = calc_perc(column, 'REM')
        wake_perc.append(curr_wake)
        NREM_perc.append(curr_NREM)
        REM_perc.append(curr_REM)
    return wake_perc, NREM_perc, REM_perc

def extract_single_stage_before(a, name, bout_per_min, beforeAndAfter):
    matrix = []
    index = int(bout_per_min*beforeAndAfter)
    for i in range(len(a)):
        if a[i][index] == name:
            matrix.append(a[i])
    new_matrix = []
    if len(matrix) == 0:
        new_matrix.append(0)
        matrix = new_matrix
    return matrix

def perc_per_animal(a, interval = trials_per_animal):
    wake_tot = []
    NREM_tot = []
    REM_tot = []
    num_animals = int(len(a)/interval)
    for i in range(num_animals):
        start = i*interval
        stop = interval +i*interval
        WAKEPER, NREMPER, REMPER = calc_percent_tot(a[start:stop])
        wake_tot.append(WAKEPER)
        NREM_tot.append(NREMPER)
        REM_tot.append(REMPER)
    return wake_tot, NREM_tot, REM_tot

def perc_per_animal_variable(a, TRIALS):
    wake_tot = []
    NREM_tot = []
    REM_tot = []
    num_animals = int(len(TRIALS))
    start = 0
    stop = 0
    for i in range(num_animals):
        start = stop
        stop = stop + TRIALS[i]
        WAKEPER, NREMPER, REMPER = calc_percent_tot(a[start:stop])
        wake_tot.append(WAKEPER)
        NREM_tot.append(NREMPER)
        REM_tot.append(REMPER)
    return wake_tot, NREM_tot, REM_tot

def trials_by_animal(a, TRIALS):
    b = []
    start = 0
    stop = 0
    num_animals = int(len(TRIALS))

    for i in range(num_animals):
        start = stop
        stop = stop + TRIALS[i]
        curr_b = a[start:stop]
        b.append(curr_b)
    return b
    
def cleanup_neg_trials(a):
    count = 0
    for i in range(len(a)):
        all_zeros = np.sum(a[i-count])
       # print(np.sum(a[i])-1)
        if all_zeros == 0:
            curr_delete = i-count
            a.pop(curr_delete)
            count+=1
    return a

def perc_per_animal_single_stage(a, name, interval = trials_per_animal):
    wake_tot = []
    NREM_tot = []
    REM_tot = []
    num_animals = int(len(a)/interval)
    total_trials = 0
    for i in range(num_animals):
        start = i*interval
        stop = interval +i*interval
        curr_data = a[start:stop]
        curr_before = extract_single_stage_before(curr_data, name)
        total_trials = total_trials + len(curr_before)
        WAKEPER, NREMPER, REMPER = calc_percent_tot(curr_before)
        wake_tot.append(WAKEPER)
        NREM_tot.append(NREMPER)
        REM_tot.append(REMPER)
    wake_tot = cleanup_neg_trials(wake_tot)
    NREM_tot = cleanup_neg_trials(NREM_tot)
    REM_tot = cleanup_neg_trials(REM_tot)
    return wake_tot, NREM_tot, REM_tot, total_trials

def perc_per_animal_single_stage_variable(a, TRIAL, name):
    wake_tot = []
    NREM_tot = []
    REM_tot = []
    num_animals = int(len(TRIAL))
    total_trials = 0
    start = 0
    stop = 0
    for i in range(num_animals):
        start = stop
        stop = stop + TRIAL[i]
        curr_data = a[start:stop]
        curr_before = extract_single_stage_before(curr_data, name)
        total_trials = total_trials + len(curr_before)
        WAKEPER, NREMPER, REMPER = calc_percent_tot(curr_before)
        wake_tot.append(WAKEPER)
        NREM_tot.append(NREMPER)
        REM_tot.append(REMPER)
    wake_tot = cleanup_neg_trials(wake_tot)
    NREM_tot = cleanup_neg_trials(NREM_tot)
    REM_tot = cleanup_neg_trials(REM_tot)
    return wake_tot, NREM_tot, REM_tot, total_trials

def avg_animals(a):
    ## basically for a given nested matrix of multiple trials
    ##returns 2 single matrixes: the average and one of standard deviation
    ## there's probably a python function that does this already but made my own 
    avg_matrix = []
    std_matrix = []
    if len(a) == 0:
        print('this metric has zero occurances')
        avg_matrix = 0
        for i in range(len(avg_matrix)):
            avg_matrix[i] = 0
        std_matrix = avg_matrix
        return avg_matrix, std_matrix
    for i in range(len(a[0])):
        curr_total = []
        for j in range(len(a)):
            curr_total.append(a[j][i])
        avg_matrix.append(np.mean(curr_total))
        std_matrix.append(st.sem(curr_total))
    return avg_matrix, std_matrix      

def determineTrialsPerAnimal(trials):
    diff_trials = []
    counter = []
    for i in range(len(trials)):
        if trials[i][0] == "'":
            trials[i] = trials[i][1:len(trials[i])]
    for i in range(len(trials)):
        if trials[i] not in diff_trials:
            diff_trials.append(trials[i])
    trials = np.asarray(trials)
    for i in range(len(diff_trials)):
        new_num = (trials == diff_trials[i]).sum()
        counter.append(new_num)
    return counter

def undueBootstrapFormatting(a):
    b = []
    for i in range(len(a)):
        for j in range(len(a[i])):
            b.append(a[i][j])
    return b

def single_mouse_bootstrap(data):
    num_trials = len(data)
    new_mouse = []
    for i in range(num_trials):
        index = int(np.random.rand()*num_trials)
        new_mouse.append(data[index])
    return new_mouse

def single_mouse_bootstrap_better(data):
    # print(len(data))
    num_trials = len(data)
    new_mouse = []
    for i in range(num_trials):
        index = int(np.random.rand()*num_trials)
        new_mouse.append(data[index])
    return new_mouse

def all_mouse_single_bootstrap(a):
    all_new_mice = []
    for i in range(len(a)):
        curr_new_mouse = single_mouse_bootstrap(a[i])
        all_new_mice.append(curr_new_mouse)
    better_format = undueBootstrapFormatting(all_new_mice)
    return better_format

def all_mouse_single_bootstrap_better(a):
    num_trials = len(a)
    b = []
    for i in range(num_trials):
        index = int(np.random.rand()*num_trials)
        b.append(a[index])
    return b
    
def combine_state_prob_from_all_animals(wake, NREM, REM):
    all_wake = []
    all_NREM = []
    all_REM = []
    for i in range(len(wake[0])):
        curr_wake = dw.extract_column(wake, i)
        curr_NREM = dw.extract_column(NREM, i)
        curr_REM = dw.extract_column(REM, i)
        all_wake.append(curr_wake)
        all_NREM.append(curr_NREM)
        all_REM.append(curr_REM)
    return all_wake, all_NREM, all_REM

def state_probabilities_single_bootstrap(a, trialOcc):
    trials4bootstrap = trials_by_animal(a,trialOcc)
    curr_bootstrap = all_mouse_single_bootstrap(trials4bootstrap)
    wake, NREM, REM = perc_per_animal_variable(curr_bootstrap, trialOcc)
    wake_tot, NREM_tot, REM_tot =  combine_state_prob_from_all_animals(wake, NREM, REM)
    return wake_tot, NREM_tot, REM_tot

def state_probabilities_single_bootstrap_better(a, trialOcc):
    trials4bootstrap = trials_by_animal(a,trialOcc)
    trials4bootstrap = undueBootstrapFormatting(trials4bootstrap)
    curr_bootstrap = all_mouse_single_bootstrap_better(trials4bootstrap)
    wake, NREM, REM = perc_per_animal_variable(curr_bootstrap, trialOcc)
    wake_tot, NREM_tot, REM_tot =  calc_percent_tot(curr_bootstrap)
    return wake_tot, NREM_tot, REM_tot

def combine2matrix(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i])
    for i in range(len(b)):
        c.append(b[i])
    return c

def combine2matrix_test(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i])
    c.append(b)
    return c

def combineIterations(a, b):
    c = []
    for i in range(len(b)):
        if len(a) >0:
            new = combine2matrix(a[i],b[i])
            c.append(new)
        else: 
            return b
    return c

def combineIterations_test(a, b):
    c = []
    # print((len(b[0])))
    for i in range(len(b)):
        if len(a) >0:
            curr_bmean = np.mean(b[i])
            new = a[i].append(curr_bmean)
            c.append(new)
        else: 
            c.append(np.mean(b[i]))
    return c
                   
def combineCI(a, b):
    c = []
    for i in range(len(b)):
        c_curr = np.array([a[i], b[i]])
        c.append(c_curr)
    c = np.array(c)
    return c

def state_probabilities_all_bootstrap_better(a,trialOcc, iterations):
    wake_all = []
    NREM_all = []
    REM_all = []
    for i in range(iterations):
        curr_wake, curr_NREM, curr_REM = state_probabilities_single_bootstrap_better(a, trialOcc)
        wake_all.append(curr_wake)
        NREM_all.append(curr_NREM)
        REM_all.append(curr_REM)
    return wake_all, NREM_all, REM_all    

def dataRealign(a):
    c = []
    for i in range(len(a[0])):
        b = dw.extract_column(a,i)
        c.append(b)
    return c


        


def wakeMaintenance_single(wake,NREM, REM, bout_length, stim_length,beforeAndAfter):
    wakeb4 = []
    NREMb4 = []
    REMb4 = []
    
    wake_stim = []
    NREM_stim = []
    REM_stim = []
    
    
    ###find start of stim
    boutPerMinute = int(60/bout_length)
    # print(stim_start)
    stim_start = int(beforeAndAfter*60/bout_length+stim_length/bout_length +2*boutPerMinute)
    # print(qstim_start)

    stim_bout_num = boutPerMinute
    
    before_start = 0
    for i in range(stim_bout_num):
        wakeb4.append(wake[before_start +i])
        NREMb4.append(NREM[before_start +i])
        REMb4.append(REM[before_start +i])
    for i in range(stim_bout_num):
        wake_stim.append(wake[stim_start +i])
        NREM_stim.append(NREM[stim_start +i])
        REM_stim.append(REM[stim_start +i]) 
    wakeb4 = np.mean(wakeb4)
    NREMb4 = np.mean(NREMb4)
    REMb4 = np.mean(REMb4) 
    wake_stim = np.mean(wake_stim)
    NREM_stim = np.mean(NREM_stim)
    REM_stim = np.mean(REM_stim)
    return wakeb4, wake_stim, NREMb4, NREM_stim, REMb4, REM_stim


def wakeMaintenance_all(wake, NREM, REM, globalFont, global_name):
    wakeb4 = []
    NREMb4 = []
    REMb4 = []
    
    wake_after = []
    NREM_after = []
    REM_after = [] 
    for i in range(len(wake)):
        WB, WA, NB,NA, RB, RA = wakeMaintenance_single(wake[i], NREM[i], REM[i])
        wakeb4.append(WB)
        NREMb4.append(NB)
        REMb4.append(RB)      
        wake_after.append(WA)
        NREM_after.append(NA)
        REM_after.append(RA)    
    po.plotDots_stats(wakeb4, '3 min before stim', wake_after, '3 min after stim', '% Wake', 'first min compared to last min_wake',globalFont, global_name)
    po.plotDots_stats(NREMb4, 'first min', NREM_after, 'last min', '% NREM', 'first min compared to last min_NREM',globalFont, global_name)
    po.plotDots_stats(REMb4, 'first min', REM_after, 'last min', '% REM', 'first min compared to last min_REM',globalFont, global_name)
    return wakeb4, wake_after



def mean_probabilites(wake, NREM, REM,bout_length, stim_length,beforeAndAfter):
    ###calc b4 prob
    wakeb4 = []
    NREMb4 = []
    REMb4 = []
    
    wake_stim = []
    NREM_stim = []
    REM_stim = []
    
    
    ###find start of stim
    stim_start = int(beforeAndAfter*60/bout_length)
    stim_bout_num = int(stim_length/bout_length)
    
    before_start = stim_start - stim_bout_num
    for i in range(stim_bout_num):
        wakeb4.append(wake[before_start +i])
        NREMb4.append(NREM[before_start +i])
        REMb4.append(REM[before_start +i])
    for i in range(stim_bout_num):
        wake_stim.append(wake[stim_start +i])
        NREM_stim.append(NREM[stim_start +i])
        REM_stim.append(REM[stim_start +i]) 
    wakeb4 = np.mean(wakeb4)
    NREMb4 = np.mean(NREMb4)
    REMb4 = np.mean(REMb4)
    
    wake_stim = np.mean(wake_stim)
    NREM_stim = np.mean(NREM_stim)
    REM_stim = np.mean(REM_stim)
    wake_diff = wake_stim-wakeb4
    NREM_diff = NREM_stim-NREMb4
    REM_diff = REM_stim-REMb4

    return wake_diff, NREM_diff, REM_diff

def mean_prob_bootstrap(wake, NREM, REM,bout_length, stim_length,beforeAndAfter):
    wake_all = []
    NREM_all = []
    REM_all = []
    for i in range(len(wake)):
        wake_diff, NREM_diff, REM_diff = mean_probabilites(wake[i], NREM[i], REM[i],bout_length, stim_length,beforeAndAfter)
        wake_all.append(wake_diff)
        NREM_all.append(NREM_diff)
        REM_all.append(REM_diff)
    return wake_all, NREM_all, REM_all
        
def increaseORdecrease(a):
    inc = []
    dec = []
    for i in range(len(a)):
        if a[i] > 0:
            inc.append(a[i])
        elif a[i] < 0:
            dec.append(a[i])
    return inc, dec

def mean_probabilites_wakeMaintenance_firstAndLastMin(wake, NREM, REM, bout_length, stim_length, beforeAndAfter):
    ###calc b4 prob
    wakeb4 = []
    NREMb4 = []
    REMb4 = []
    
    wake_stim = []
    NREM_stim = []
    REM_stim = []
    
    
    ###find start of stim
    boutPerMinute = int(60/bout_length)
    # print(stim_start)
    stim_start = int(beforeAndAfter*60/bout_length+stim_length/bout_length +2*boutPerMinute)
    # print(qstim_start)

    stim_bout_num = boutPerMinute
    
    before_start = 0
    for i in range(stim_bout_num):
        wakeb4.append(wake[before_start +i])
        NREMb4.append(NREM[before_start +i])
        REMb4.append(REM[before_start +i])
    for i in range(stim_bout_num):
        wake_stim.append(wake[stim_start +i])
        NREM_stim.append(NREM[stim_start +i])
        REM_stim.append(REM[stim_start +i]) 
    wakeb4 = np.mean(wakeb4)
    NREMb4 = np.mean(NREMb4)
    REMb4 = np.mean(REMb4)
    
    wake_stim = np.mean(wake_stim)
    NREM_stim = np.mean(NREM_stim)
    REM_stim = np.mean(REM_stim)
    wake_diff = wake_stim-wakeb4
    NREM_diff = NREM_stim-NREMb4
    REM_diff = REM_stim-REMb4

    return wake_diff, NREM_diff, REM_diff    

def mean_prob_bootstrap_main1(wake, NREM, REM,bout_length, stim_length,beforeAndAfter):
    wake_all = []
    NREM_all = []
    REM_all = []
    for i in range(len(wake)):
        wake_diff, NREM_diff, REM_diff = mean_probabilites_wakeMaintenance_firstAndLastMin(wake[i], NREM[i], REM[i],bout_length, stim_length,beforeAndAfter)
        wake_all.append(wake_diff)
        NREM_all.append(NREM_diff)
        REM_all.append(REM_diff)
    return wake_all, NREM_all, REM_all

def mean_probabilites_wakeMaintenance_firstAndLastMin_2(wake, NREM, REM,bout_length, stim_length, beforeAndAfter):
    ###calc b4 prob
    wakeb4 = []
    NREMb4 = []
    REMb4 = []
    
    wake_stim = []
    NREM_stim = []
    REM_stim = []
    
    
    ###find start of stim
    boutPerMinute = int(60/bout_length)
    # print(stim_start)
    stim_start = int(beforeAndAfter*60/bout_length+stim_length/bout_length +1*boutPerMinute)
    # print(qstim_start)

    stim_bout_num = boutPerMinute
    
    before_start = boutPerMinute
    for i in range(stim_bout_num):
        wakeb4.append(wake[before_start +i])
        NREMb4.append(NREM[before_start +i])
        REMb4.append(REM[before_start +i])
    for i in range(stim_bout_num):
        wake_stim.append(wake[stim_start +i])
        NREM_stim.append(NREM[stim_start +i])
        REM_stim.append(REM[stim_start +i]) 
    wakeb4 = np.mean(wakeb4)
    NREMb4 = np.mean(NREMb4)
    REMb4 = np.mean(REMb4)
    
    wake_stim = np.mean(wake_stim)
    NREM_stim = np.mean(NREM_stim)
    REM_stim = np.mean(REM_stim)
    wake_diff = wake_stim-wakeb4
    NREM_diff = NREM_stim-NREMb4
    REM_diff = REM_stim-REMb4

    return wake_diff, NREM_diff, REM_diff    

def mean_prob_bootstrap_main2(wake, NREM, REM, bout_length, stim_length, beforeAndAfter):
    wake_all = []
    NREM_all = []
    REM_all = []
    for i in range(len(wake)):
        wake_diff, NREM_diff, REM_diff = mean_probabilites_wakeMaintenance_firstAndLastMin_2(wake[i], NREM[i], REM[i], bout_length, stim_length, beforeAndAfter)
        wake_all.append(wake_diff)
        NREM_all.append(NREM_diff)
        REM_all.append(REM_diff)
    return wake_all, NREM_all, REM_all




def mean_probabilites_wakeMaintenance_firstAndLastMin_3(wake, NREM, REM,bout_length, stim_length, beforeAndAfter):
    ###calc b4 prob
    wakeb4 = []
    NREMb4 = []
    REMb4 = []
    
    wake_stim = []
    NREM_stim = []
    REM_stim = []
    
    
    ###find start of stim
    boutPerMinute = int(60/bout_length)
    # print(stim_start)
    stim_start = int(beforeAndAfter*60/bout_length+stim_length/bout_length +0*boutPerMinute)
    # print(qstim_start)

    stim_bout_num = boutPerMinute
    
    before_start = boutPerMinute*2
    for i in range(stim_bout_num):
        wakeb4.append(wake[before_start +i])
        NREMb4.append(NREM[before_start +i])
        REMb4.append(REM[before_start +i])
    for i in range(stim_bout_num):
        wake_stim.append(wake[stim_start +i])
        NREM_stim.append(NREM[stim_start +i])
        REM_stim.append(REM[stim_start +i]) 
    wakeb4 = np.mean(wakeb4)
    NREMb4 = np.mean(NREMb4)
    REMb4 = np.mean(REMb4)
    
    wake_stim = np.mean(wake_stim)
    NREM_stim = np.mean(NREM_stim)
    REM_stim = np.mean(REM_stim)
    wake_diff = wake_stim-wakeb4
    NREM_diff = NREM_stim-NREMb4
    REM_diff = REM_stim-REMb4

    return wake_diff, NREM_diff, REM_diff    

def mean_prob_bootstrap_main3(wake, NREM, REM, bout_length, stim_length, beforeAndAfter):
    wake_all = []
    NREM_all = []
    REM_all = []
    for i in range(len(wake)):
        wake_diff, NREM_diff, REM_diff = mean_probabilites_wakeMaintenance_firstAndLastMin_3(wake[i], NREM[i], REM[i], bout_length, stim_length, beforeAndAfter)
        wake_all.append(wake_diff)
        NREM_all.append(NREM_diff)
        REM_all.append(REM_diff)
    return wake_all, NREM_all, REM_all

def generatePlotableCI(a, err):
    
    for i in range(len(a)):
        err[0][i] = a[i] - err[0][i]
        err[1][i] =err[1][i] - a[i]
    return err


def extractConfidenceIntervals_single(a, index, lower = 0.025, upper = 0.975):
    curr_a = dw.extract_column(a, index)
    a_sorted = curr_a[:]
    a_sorted.sort()
    lower_val = a_sorted[int(len(a_sorted)*lower)]
    upper_val = a_sorted[int(len(a_sorted)*upper)]
    return lower_val, upper_val

def extractConfidenceIntervals_all(a, lower = 0.025, upper = 0.975):
    lower_all =[]
    upper_all = []
    for i in range(len(a[0])):
        low_curr, up_curr = extractConfidenceIntervals_single(a,i)
        lower_all.append(float(low_curr))
        upper_all.append(float(up_curr))
    lower_all = np.array(lower_all)
    upper_all = np.array(upper_all)
    CI = [lower_all, upper_all]
    return CI