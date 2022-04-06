#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 19:10:45 2019

@author: jmw
"""
############################################
import csv
import scipy.stats as st
import matplotlib

from LocalResources import dataWrangling as dw
from LocalResources import plottingOptions as po
from LocalResources import sleepBoutProcessing as sbp

matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
##############################################
##############################################
##############################################
##############################################
data_dir = 'Data/'
name_score = 'pTRN EEG scores.csv' ## user scored epochs
fileName = data_dir + name_score
max_bout = 8640 ## the maximum number of bouts in the file 
days = 2 ## number of days per mouse
bout_length = 10 ## number of seconds per bout
minutes_per_display = 60 ## number of minutes to display in the hr-by-hr graph
globalFont = 15 ## sets font size for figures
hours = 24 ## the number of hours in a day
STARTLIGHT = 0 ## beginning of light period for display
ENDLIGHT = 12 ## end of light period for display
global_name ='_caspase EEG' ## attaches this name to the end of every file
WT_NAME = 'WT' ## x label for WT
MUTANT_NAME = 'Cre' ## x label for experimental mice




with open(fileName, newline='\n' ) as inputfile:
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

max_bout = len(WT[0])-2
wt_REM_len, wt_NREM_len, wt_wake_len = sbp.find_all_len(WT, max_bout)
wt_REM_bouts, wt_REM_len_avg = sbp.avg_bout_len(wt_REM_len, bout_length)
wt_NREM_bouts, wt_NREM_len_avg = sbp.avg_bout_len(wt_NREM_len, bout_length)
wt_wake_bouts, wt_wake_len_avg = sbp.avg_bout_len(wt_wake_len, bout_length)
mut_REM_len, mut_NREM_len, mut_wake_len = sbp.find_all_len(Cre, max_bout)
mut_REM_bouts, mut_REM_len_avg = sbp.avg_bout_len(mut_REM_len, bout_length)
mut_NREM_bouts, mut_NREM_len_avg = sbp.avg_bout_len(mut_NREM_len, bout_length)
mut_wake_bouts, mut_wake_len_avg = sbp.avg_bout_len(mut_wake_len, bout_length)
mut_wake_len_avg_min = sbp.sec2min(mut_wake_len_avg)
wt_wake_len_avg_min = sbp.sec2min(wt_wake_len_avg)
mut_NREM_len_avg_min = sbp.sec2min(mut_NREM_len_avg)
wt_NREM_len_avg_min = sbp.sec2min(wt_NREM_len_avg)
wt_wake_totalTime, wt_NREM_totalTime, wt_REM_totalTime = sbp.total_sleep_time(wt_wake_len, wt_NREM_len, wt_REM_len)
mut_wake_totalTime, mut_NREM_totalTime, mut_REM_totalTime = sbp.total_sleep_time(mut_wake_len, mut_NREM_len, mut_REM_len)
po.plotDots(wt_wake_totalTime,WT_NAME, mut_wake_totalTime, MUTANT_NAME, 'Wake time (hr)', 'Total wake time', globalFont, global_name, wake = True)
po.plotDots(wt_NREM_totalTime,WT_NAME, mut_NREM_totalTime, MUTANT_NAME, 'NREM time (hr)', 'Total NREM time',globalFont, global_name)
po.plotDots(wt_REM_totalTime,WT_NAME, mut_REM_totalTime, MUTANT_NAME, 'REM time (min)', 'Total REM time',globalFont, global_name)
wt_wake_perc, wt_NREM_perc, wt_REM_perc = sbp.percent_by_hour(wild, bout_length,minutes = minutes_per_display)
mut_wake_perc, mut_NREM_perc, mut_REM_perc = sbp.percent_by_hour(mutant, bout_length,minutes = minutes_per_display)
wt_wake_perc_avg, wt_wake_perc_std = dw.avg_animals(wt_wake_perc)
wt_NREM_perc_avg, wt_NREM_perc_std = dw.avg_animals(wt_NREM_perc)
wt_REM_perc_avg, wt_REM_perc_std = dw.avg_animals(wt_REM_perc)
mut_wake_perc_avg, mut_wake_perc_std = dw.avg_animals(mut_wake_perc)
mut_NREM_perc_avg, mut_NREM_perc_std = dw.avg_animals(mut_NREM_perc)
mut_REM_perc_avg, mut_REM_perc_std = dw.avg_animals(mut_REM_perc)
po.plotPerc(wt_wake_perc_avg, wt_wake_perc_std,mut_wake_perc_avg,mut_wake_perc_std, 'Wake percentage by hour',WT_NAME, MUTANT_NAME, 'Wake %',globalFont, global_name)
po.plotPerc(wt_NREM_perc_avg, wt_NREM_perc_std,mut_NREM_perc_avg,mut_NREM_perc_std, 'NREM percentage by hour',WT_NAME, MUTANT_NAME, 'NREM %',globalFont, global_name)
po.plotPerc(wt_REM_perc_avg, wt_REM_perc_std,mut_REM_perc_avg,mut_REM_perc_std, 'REM percentage by hour',WT_NAME, MUTANT_NAME, 'REM %',globalFont, global_name)
wake_by_hr_ttest = po.ttest_by_hr(wt_wake_perc,mut_wake_perc)
NREM_by_hr_ttest = po.ttest_by_hr(wt_NREM_perc,mut_NREM_perc)
REM_by_hr_ttest = po.ttest_by_hr(wt_REM_perc,mut_REM_perc)
wtLight, wtDark = sbp.LDsplit(wildy)
mutLight, mutDark = sbp.LDsplit(mutanty)
max_bout = len(wtLight[0])-2
wt_dark_REM_len, wt_dark_NREM_len, wt_dark_wake_len = sbp.find_all_len(wtDark, max_bout)
mut_dark_REM_len, mut_dark_NREM_len, mut_dark_wake_len = sbp.find_all_len(mutDark, max_bout)
wt_dark_wake_totalTime, wt_dark_NREM_totalTime, wt_dark_REM_totalTime = sbp.total_sleep_time(wt_dark_wake_len, wt_dark_NREM_len, wt_dark_REM_len)
mut_dark_wake_totalTime, mut_dark_NREM_totalTime, mut_dark_REM_totalTime = sbp.total_sleep_time(mut_dark_wake_len, mut_dark_NREM_len, mut_dark_REM_len)
po.plotDots(wt_dark_wake_totalTime,WT_NAME, mut_dark_wake_totalTime, MUTANT_NAME, 'Time (hr)', 'Total dark wake time',globalFont, global_name)
po.plotDots(wt_dark_NREM_totalTime,WT_NAME, mut_dark_NREM_totalTime, MUTANT_NAME, 'Time (hr)', 'Total dark NREM time',globalFont, global_name)
po.plotDots(wt_dark_REM_totalTime,WT_NAME, mut_dark_REM_totalTime, MUTANT_NAME, 'Time (m)', 'Total dark REM time',globalFont, global_name)
wt_light_REM_len, wt_light_NREM_len, wt_light_wake_len = sbp.find_all_len(wtLight, max_bout)
mut_light_REM_len, mut_light_NREM_len, mut_light_wake_len = sbp.find_all_len(mutLight,max_bout)
wt_light_wake_totalTime, wt_light_NREM_totalTime, wt_light_REM_totalTime = sbp.total_sleep_time(wt_light_wake_len, wt_light_NREM_len, wt_light_REM_len)
mut_light_wake_totalTime, mut_light_NREM_totalTime, mut_light_REM_totalTime = sbp.total_sleep_time(mut_light_wake_len, mut_light_NREM_len, mut_light_REM_len)
po.plotDots(wt_light_wake_totalTime,WT_NAME, mut_light_wake_totalTime, MUTANT_NAME, 'Time (hr)', 'Total light wake time',globalFont, global_name)
po.plotDots(wt_light_NREM_totalTime,WT_NAME, mut_light_NREM_totalTime, MUTANT_NAME, 'Time (hr)', 'Total light NREM time',globalFont, global_name)
po.plotDots(wt_light_REM_totalTime,WT_NAME, mut_light_REM_totalTime, MUTANT_NAME, 'Time (min)', 'Total light REM time',globalFont, global_name)
po.plotHist(wt_REM_len, mut_REM_len, 'REM bout len (m)', 'Frequency (%)', 'REM hist', WT_NAME, MUTANT_NAME, bout_length,globalFont, global_name)
po.plotHist(wt_NREM_len, mut_NREM_len, 'NREM bout len (m)', 'Frequency (%)', 'NREM hist', WT_NAME, MUTANT_NAME, bout_length,globalFont, global_name)
po.plotHist(wt_wake_len, mut_wake_len, 'Wake bout len (m)', 'Frequency (%)', 'Wake hist', WT_NAME , MUTANT_NAME, bout_length,globalFont, global_name)
total_wake_ttest = st.ttest_ind(wt_wake_totalTime,mut_wake_totalTime, equal_var = False)
total_wake_ttest = total_wake_ttest[1]
total_NREM_ttest = st.ttest_ind(wt_NREM_totalTime,mut_NREM_totalTime, equal_var = False)
total_NREM_ttest = total_NREM_ttest[1]
total_REM_ttest = st.ttest_ind(wt_REM_totalTime,mut_REM_totalTime, equal_var = False)
total_REM_ttest = total_REM_ttest[1]
total_dark_wake_ttest = st.ttest_ind(wt_dark_wake_totalTime,mut_dark_wake_totalTime, equal_var = False)
total_dark_wake_ttest = total_dark_wake_ttest[1]
total_dark_NREM_ttest = st.ttest_ind(wt_dark_NREM_totalTime,mut_dark_NREM_totalTime, equal_var = False)
total_dark_NREM_ttest = total_dark_NREM_ttest[1]
total_dark_REM_ttest = st.ttest_ind(wt_dark_REM_totalTime,mut_dark_REM_totalTime, equal_var = False)
total_dark_REM_ttest = total_dark_REM_ttest[1]
total_light_wake_ttest = st.ttest_ind(wt_light_wake_totalTime,mut_light_wake_totalTime, equal_var = False)
total_light_wake_ttest = total_light_wake_ttest[1]
total_light_NREM_ttest = st.ttest_ind(wt_light_NREM_totalTime,mut_light_NREM_totalTime, equal_var = False)
total_light_NREM_ttest = total_light_NREM_ttest[1]
total_light_REM_ttest = st.ttest_ind(wt_light_REM_totalTime,mut_light_REM_totalTime, equal_var = False)
total_light_REM_ttest = total_light_REM_ttest[1]
po.plot_barplot(wt_wake_bouts,WT_NAME, mut_wake_bouts, MUTANT_NAME, 'Bout number', 'Average Wake Bout #',globalFont, global_name)
po.plot_barplot(wt_NREM_bouts,WT_NAME, mut_NREM_bouts, MUTANT_NAME, 'Bout number', 'Average NREM Bout #',globalFont, global_name)
po.plot_barplot(wt_REM_bouts,WT_NAME, mut_REM_bouts, MUTANT_NAME, 'Bout number', 'Average REM Bout #',globalFont, global_name)
po.plot_barplot(wt_wake_len_avg_min,WT_NAME, mut_wake_len_avg_min, MUTANT_NAME, 'Bout len (m)', 'Average wake bout length',globalFont, global_name)
po.plot_barplot(wt_NREM_len_avg_min,WT_NAME, mut_NREM_len_avg_min, MUTANT_NAME, 'Bout len (m)', 'Average NREM bout length',globalFont, global_name)
po.plot_barplot_REM(wt_REM_len_avg,WT_NAME, mut_REM_len_avg, MUTANT_NAME, 'Bout len (sec)', 'Average REM bout length',globalFont, global_name)
po.plot_barplot_4groups(wt_wake_bouts,WT_NAME, mut_wake_bouts, MUTANT_NAME, wt_NREM_bouts, mut_NREM_bouts, 'Bout #', 'Average Wake & NREM Bout #',globalFont, global_name)
po.plot_barplot_6groups(wt_wake_bouts,WT_NAME, mut_wake_bouts, MUTANT_NAME, wt_NREM_bouts, mut_NREM_bouts,wt_REM_bouts, mut_REM_bouts, 'Bout #', 'Average Wake & NREM &REM Bout #',globalFont, global_name)
po.plot_barplot_4groups(wt_wake_len_avg_min,WT_NAME, mut_wake_len_avg_min, MUTANT_NAME, wt_NREM_len_avg_min,mut_NREM_len_avg_min,'Bout len (min)', 'Average wake bout length + NREM',globalFont, global_name)

