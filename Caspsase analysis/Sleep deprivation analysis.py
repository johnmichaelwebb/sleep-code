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
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')


from LocalResources import dataWrangling as dw
from LocalResources import sleepBoutProcessing as sbp
from LocalResources import plottingOptions as po

##############################################
##############################################
##############################################
##############################################

data_dir = 'Data/'
name_score = 'pTRN EEG scores.csv' ##user scored epochs
filename = data_dir + name_score
sleep_dep_name = 'pTRNcaspSD.csv' ##sleep deprivation epochs
filename_SD = data_dir + sleep_dep_name
days = 2 ## number of days per mouse
bout_length = 10 ##number of seconds per bout
minutes_per_display = 60 ## number of minutes to display in the hr-by-hr graph
globalFont = 20 ## sets font size for figures
global_name = '_pTRN caspase SD' ## attaches this name to the end of every file
MUTANT_NAME = 'Cre' ## x label for experimental mice
WILDTYPE_NAME = 'WT' ## x label for WT



with open(filename, newline='\n' ) as inputfile:
   data = list(csv.reader(inputfile)) 
   data[0][0] = data[0][0][1:len(data[0][0])]
with open(filename_SD, newline='\n' ) as inputfile:
   data_SD = list(csv.reader(inputfile)) 
   data_SD[0][0] = data_SD[0][0][1:len(data_SD[0][0])]
names = dw.extract_row(data,1)
Cre = dw.create_empty_matrix(names, MUTANT_NAME, data)
WT = dw.create_empty_matrix(names, WILDTYPE_NAME,data)
wild = WT[:]
wildy = WT[:]
mutant = Cre[:]
mutanty = Cre[:]
max_bout = len(WT[0])-2
wt_wake_perc, wt_NREM_perc, wt_REM_perc = sbp.percent_by_hour(wild, bout_length, days = 2,  minutes = minutes_per_display)
mut_wake_perc, mut_NREM_perc, mut_REM_perc = sbp.percent_by_hour(mutant,bout_length,days = 2,  minutes = minutes_per_display)
wt_wake_min, wt_NREM_min, wt_REM_min = sbp.perc2min(wt_wake_perc, wt_NREM_perc, wt_REM_perc)
mut_wake_min, mut_NREM_min, mut_REM_min = sbp.perc2min(mut_wake_perc, mut_NREM_perc, mut_REM_perc)



names_SD = dw.extract_row(data_SD,1)
Cre_SD = dw.create_empty_matrix(names_SD, MUTANT_NAME, data_SD)
WT_SD = dw.create_empty_matrix(names_SD, WILDTYPE_NAME, data_SD)
max_bout = len(WT_SD[0])-2
WT_SD = sbp.replaceSDwithWake(WT_SD, 'Wake', 5, bout_length)
Cre_SD = sbp.replaceSDwithWake(Cre_SD, 'Wake',5, bout_length)
wild_SD = WT_SD[:]              
             
wt_SD_wake_perc, wt_SD_NREM_perc, wt_SD_REM_perc = sbp.percent_by_hour(WT_SD,bout_length, days = 1,  minutes = minutes_per_display)
mut_SD_wake_perc, mut_SD_NREM_perc, mut_SD_REM_perc = sbp.percent_by_hour(Cre_SD,bout_length,days = 1,  minutes = minutes_per_display)
wt_SD_wake_min, wt_SD_NREM_min, wt_SD_REM_min = sbp.perc2min(wt_SD_wake_perc, wt_SD_NREM_perc, wt_SD_REM_perc)
mut_SD_wake_min, mut_SD_NREM_min, mut_SD_REM_min = sbp.perc2min(mut_SD_wake_perc, mut_SD_NREM_perc, mut_SD_REM_perc)
wt_NREM_debt = sbp.sleepDebt(wt_NREM_min, wt_SD_NREM_min)
mut_NREM_debt = sbp.sleepDebt(mut_NREM_min,mut_SD_NREM_min)
wt_REM_debt = sbp.sleepDebt(wt_REM_min, wt_SD_REM_min)
mut_REM_debt = sbp.sleepDebt(mut_REM_min, mut_SD_REM_min)
wt_NREM_debt_avg, wt_NREM_debt_std = dw.avg_animals_debt(wt_NREM_debt)
wt_REM_debt_avg, wt_REM_debt_std = dw.avg_animals_debt(wt_REM_debt)
mut_NREM_debt_avg, mut_NREM_debt_std = dw.avg_animals_debt(mut_NREM_debt)
mut_REM_debt_avg, mut_REM_debt_std = dw.avg_animals_debt(mut_REM_debt)
po.plot_perc('% NREM recovery',wt_NREM_debt_avg, wt_NREM_debt_std,mut_NREM_debt_avg,mut_NREM_debt_std, WILDTYPE_NAME, MUTANT_NAME, 'NREM sleep debt',globalFont, global_name)
po.plot_perc('% REM recovery',wt_REM_debt_avg, wt_REM_debt_std,mut_REM_debt_avg,mut_REM_debt_std,WILDTYPE_NAME, MUTANT_NAME,  'REM sleep debt',globalFont, global_name)
final_wt_NREM_dep = dw.extract_column(wt_NREM_debt,23)
final_mut_NREM_dep = dw.extract_column(mut_NREM_debt,23)
final_wt_REM_dep = dw.extract_column(wt_REM_debt,23)
final_mut_REM_dep = dw.extract_column(mut_REM_debt,23)
po.plotDots_stats(final_wt_NREM_dep, WILDTYPE_NAME, final_mut_NREM_dep, MUTANT_NAME, 'Final NREM recovery %', 'final NREM sleep recovery',globalFont, global_name)
po.plotDots_stats(final_wt_REM_dep, WILDTYPE_NAME, final_mut_REM_dep, MUTANT_NAME, 'Final REM recovery %', 'final REM sleep recovery',globalFont, global_name)