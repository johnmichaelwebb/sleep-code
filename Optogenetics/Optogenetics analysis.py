
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
import matplotlib
import scipy.stats as st
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from statannot import add_stat_annotation


from LocalResources import dataWrangling as dw
from LocalResources import plottingOptions as po
# from LocalResources import sleepBoutProcessing as sbp
from LocalResources import optoBootstrap as ob



matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')


##############################################
##############################################
##############################################
##############################################
data_dir = 'Data/'
name_score = 'pTRN_300trials_6mice.csv' #user scored epochs
filename = data_dir + name_score
days = 2 ## number of days per mouse
bout_length = 5 ##number of seconds per bout
minutes_per_display = 60
globalFont = 15
global_name = name_score[0:len(name_score)-4]
NREM_before = '_' +global_name + ' NREM before'
REM_before = '_' +global_name +'REM before'
wake_before = '_' +global_name +'Wake before'
min_before1st = 3 #min before the first trial
beforeAndAfter = 3 #min before and after to visualize
num_trials = 300
inhibitory = False
ITERATIONS = 10000


bout_per_min = int(60/bout_length)
stim_freq = 20 ## how ofter stim repeats in min
stim_length = 90 ##stim length in sec
trials_per_animal = 50




with open(filename, newline='\n' ) as inputfile:
   data = list(csv.reader(inputfile)) 
   data[0][0] = data[0][0][1:len(data[0][0])] 
    

for i in range(len(data)-1):
    data[i+1][0] = data[i+1][0].split(",")
new_data = [0 for i in range(len(data))]
new_data[0] = data[0]
for i in range(len(data)-1):
    new_data[i+1] = data[i+1][0]
new_data = dw.scrub_formating(new_data, new_data)
new_data[0][0]


names = dw.extract_column(new_data, 0)
names[0] = names[0][0:len(names[0])]
trialOccurances = ob.determineTrialsPerAnimal(names)
justData = dw.just_scores(new_data)



wake_total, NREM_total, REM_total = ob.perc_per_animal_variable(justData, trialOccurances)


wake_avg, wake_sem = dw.avg_animals(wake_total)
NREM_avg, NREM_sem = dw.avg_animals(NREM_total)
REM_avg, REM_sem = dw.avg_animals(REM_total)

NREMbefore = ob.extract_single_stage_before(justData, 'NREM', bout_per_min, beforeAndAfter)
REMbefore = ob.extract_single_stage_before(justData, 'REM', bout_per_min, beforeAndAfter)
wakebefore = ob.extract_single_stage_before(justData, 'Wake', bout_per_min, beforeAndAfter)



wakePer, NREMper, REMper = ob.calc_percent_tot(justData)
NREMbefore_wakePer, NREMbefore_NREMper, NREMbefore_REMper = ob.calc_percent_tot(NREMbefore)
REMbefore_wakePer, REMbefore_NREMper, REMbefore_REMper = ob.calc_percent_tot(REMbefore)
wakebefore_wakePer, wakebefore_NREMper, wakebefore_REMper = ob.calc_percent_tot(wakebefore)
trials4bootstrap = ob.trials_by_animal(justData,trialOccurances)
trials4bootstrap_realigned = ob.undueBootstrapFormatting(trials4bootstrap)
bootstrap_w, bootstrap_n, bootstrap_r = ob.state_probabilities_all_bootstrap_better(trials4bootstrap_realigned, trialOccurances, iterations = ITERATIONS)
wake_CI = ob.extractConfidenceIntervals_all(bootstrap_w)
NREM_CI = ob.extractConfidenceIntervals_all(bootstrap_n)
REM_CI = ob.extractConfidenceIntervals_all(bootstrap_r)
wake_CI = ob.generatePlotableCI(wake_avg, wake_CI)
NREM_CI = ob.generatePlotableCI(NREM_avg, NREM_CI)
REM_CI = ob.generatePlotableCI(REM_avg, REM_CI)

sec = po.create_sec(bout_per_min,beforeAndAfter,stim_length,bout_length)
po.plotPercCI(wake_avg, wake_CI, NREM_avg, NREM_CI, REM_avg, REM_CI, sec, global_name +'CI',globalFont, global_name, inhibitory, stim_length)



wake_diff, NREM_diff, REM_diff = ob.mean_probabilites(wake_avg, NREM_avg, REM_avg,bout_length, stim_length,beforeAndAfter)
wake_diff_main_1, NREM_diff_main_2, REM_diff_main_3 = ob.mean_probabilites_wakeMaintenance_firstAndLastMin(wake_avg, NREM_avg, REM_avg,bout_length, stim_length, beforeAndAfter)

wake_diff_bs, NREM_diff_bs, REM_diff_bs = ob.mean_prob_bootstrap(bootstrap_w, bootstrap_n,bootstrap_r,bout_length, stim_length,beforeAndAfter)
wake_diff_bs_main1, NREM_diff_bs_main1, REM_diff_bs_main1 = ob.mean_prob_bootstrap_main1(bootstrap_w, bootstrap_n,bootstrap_r,bout_length, stim_length,beforeAndAfter)
wake_diff_bs_main2, NREM_diff_bs_main2, REM_diff_bs_main2= ob.mean_prob_bootstrap_main2(bootstrap_w, bootstrap_n,bootstrap_r, bout_length, stim_length, beforeAndAfter)
wake_diff_bs_main3, NREM_diff_bs_main3, REM_diff_bs_main3= ob.mean_prob_bootstrap_main3(bootstrap_w, bootstrap_n,bootstrap_r, bout_length, stim_length, beforeAndAfter)



wake_inc, wake_dec = ob.increaseORdecrease(wake_diff_bs)
NREM_inc, NREM_dec = ob.increaseORdecrease(NREM_diff_bs)
REM_inc, REM_dec = ob.increaseORdecrease(REM_diff_bs)

wake_inc_main1, wake_dec_main1 = ob.increaseORdecrease(wake_diff_bs_main1)
NREM_inc_main1, NREM_dec_main1 = ob.increaseORdecrease(NREM_diff_bs_main1)
REM_inc_main1, REM_dec_main1 = ob.increaseORdecrease(REM_diff_bs_main1)

wake_inc_main2, wake_dec_main2 = ob.increaseORdecrease(wake_diff_bs_main2)
NREM_inc_main2, NREM_dec_main2 = ob.increaseORdecrease(NREM_diff_bs_main2)
REM_inc_main2, REM_dec_main2 = ob.increaseORdecrease(REM_diff_bs_main2)


wake_inc_main3, wake_dec_main3 = ob.increaseORdecrease(wake_diff_bs_main3)
NREM_inc_main3, NREM_dec_main3 = ob.increaseORdecrease(NREM_diff_bs_main3)
REM_inc_main3, REM_dec_main3 = ob.increaseORdecrease(REM_diff_bs_main3)





