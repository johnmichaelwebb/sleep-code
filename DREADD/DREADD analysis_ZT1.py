 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 19:10:45 2019

@author: jmw
"""




############################################
import csv
import matplotlib
import scipy.stats as st
from LocalResources import dataWrangling as dw
from LocalResources import localStats as ls
from LocalResources import sleepBoutProcessing as sbp
from LocalResources import plottingOptions as po

matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')

##############################################
##############################################
##############################################
##############################################
data_dir = 'Data/'
name_score = 'pTRN DREADD_ZT1.csv' #user scored epochs
filename = data_dir + name_score
days = 2 ## number of days per mouse
bout_length = 10 ##number of seconds per bout
minutes_per_display = 60
globalFont = 15
global_name = '_ZT1_DREADD Gq whole'
half = False  ## Whether to plot half of the data



##Initial data importation
with open(filename, newline='\n' ) as inputfile:
   data = list(csv.reader(inputfile)) 
   data[0][0] = data[0][0][1:len(data[0][0])]
names = dw.extract_row(data,1)
Cre = dw.create_empty_matrix(names, 'CNO_Gq', data)
WT = dw.create_empty_matrix(names, 'Sal_Gq',data) 
wild = WT[:]
wildy = WT[:]
mutant = Cre[:]
mcherryCNO = dw.create_empty_matrix(names, 'CNO_mcherry', data)
mcherryCNOy = mcherryCNO[:]
mcherrySAL = dw.create_empty_matrix(names, 'Sal_mcherry', data)
mcherrySALy = mcherrySAL[:]
max_bout = len(WT[0])-2
tot_hrs = int(len(WT[0])/60/60*bout_length)              

if half == True:
    Cre = po.createHalf(Cre)
    WT = po.createHalf(wildy)
    max_bout = len(WT[0])-2


##Some initial data cleaning and basic parameter calculations
wt_REM_len, wt_NREM_len, wt_wake_len = sbp.find_all_len(WT)
mcherrySAL_REM_len, mcherrySAL_NREM_len, mcherrySAL_wake_len = sbp.find_all_len(mcherrySAL)
mut_REM_len, mut_NREM_len, mut_wake_len = sbp.find_all_len(Cre)
mcherryCNO_REM_len, mcherryCNO_NREM_len, mcherryCNO_wake_len = sbp.find_all_len(mcherryCNO)
wt_wake_totalTime, wt_NREM_totalTime, wt_REM_totalTime = sbp.total_sleep_time(wt_wake_len, wt_NREM_len, wt_REM_len, mulDays = False)
mcherrySAL_wake_totalTime, mcherrySAL_NREM_totalTime, mcherrySAL_REM_totalTime = sbp.total_sleep_time(mcherrySAL_wake_len, mcherrySAL_NREM_len, mcherrySAL_REM_len, mulDays = False)
mut_wake_totalTime, mut_NREM_totalTime, mut_REM_totalTime = sbp.total_sleep_time(mut_wake_len, mut_NREM_len, mut_REM_len, mulDays = False)
mcherryCNO_wake_totalTime, mcherryCNO_NREM_totalTime, mcherryCNO_REM_totalTime = sbp.total_sleep_time(mcherryCNO_wake_len, mcherryCNO_NREM_len, mcherryCNO_REM_len, mulDays = False)


##CNO v/s saline conditons for the 
po.plotDots(wt_wake_totalTime,'Sal', mut_wake_totalTime, 'CNO', 'time (hr)', 'Total wake time', globalFont, global_name)
po.plotDots(wt_NREM_totalTime,'Sal', mut_NREM_totalTime, 'CNO', 'time (hr)', 'Total NREM time', globalFont, global_name)
po.plotDots(wt_REM_totalTime,'Sal', mut_REM_totalTime, 'CNO', 'time (m)', 'Total REM time', globalFont, global_name)


po.plotDots(mcherrySAL_wake_totalTime,'Sal', mcherryCNO_wake_totalTime, 'CNO', 'time (hr)', 'Total wake time mcherry', globalFont, global_name)
po.plotDots(mcherrySAL_NREM_totalTime,'Sal', mcherryCNO_NREM_totalTime, 'CNO', 'time (hr)', 'Total NREM time mcherry', globalFont, global_name)
po.plotDots(mcherrySAL_REM_totalTime,'Sal', mcherryCNO_REM_totalTime, 'CNO', 'time (m)', 'Total REM time mcherry', globalFont, global_name)





##calculate the percent by hour for each of the 4 experimental groups
wt_wake_perc, wt_NREM_perc, wt_REM_perc = sbp.percent_by_hour(wild, minutes = minutes_per_display, mul_days = False)
mcherrySAL_wake_perc, mcherrySAL_NREM_perc, mcherrySAL_REM_perc = sbp.percent_by_hour(mcherrySALy, minutes = minutes_per_display, mul_days = False)
mut_wake_perc, mut_NREM_perc, mut_REM_perc = sbp.percent_by_hour(mutant,minutes = minutes_per_display, mul_days = False)
mcherryCNO_wake_perc, mcherryCNO_NREM_perc, mcherryCNO_REM_perc = sbp.percent_by_hour(mcherryCNOy,minutes = minutes_per_display, mul_days = False)



### calculate the averages for the 4 experimental groups for each sleep stage 
wt_wake_perc_avg, wt_wake_perc_std = sbp.avg_animals(wt_wake_perc)
wt_NREM_perc_avg, wt_NREM_perc_std = sbp.avg_animals(wt_NREM_perc)
wt_REM_perc_avg, wt_REM_perc_std = sbp.avg_animals(wt_REM_perc)
mcherrySAL_wake_perc_avg, mcherrySAL_wake_perc_std = sbp.avg_animals(mcherrySAL_wake_perc)
mcherrySAL_NREM_perc_avg, mcherrySAL_NREM_perc_std = sbp.avg_animals(mcherrySAL_NREM_perc)
mcherrySAL_REM_perc_avg, mcherrySAL_REM_perc_std = sbp.avg_animals(mcherrySAL_REM_perc)
mut_wake_perc_avg, mut_wake_perc_std = sbp.avg_animals(mut_wake_perc)
mut_NREM_perc_avg, mut_NREM_perc_std = sbp.avg_animals(mut_NREM_perc)
mut_REM_perc_avg, mut_REM_perc_std = sbp.avg_animals(mut_REM_perc)
mcherryCNO_wake_perc_avg, mcherryCNO_wake_perc_std = sbp.avg_animals(mcherryCNO_wake_perc)
mcherryCNO_NREM_perc_avg, mcherryCNO_NREM_perc_std = sbp.avg_animals(mcherryCNO_NREM_perc)
mcherryCNO_REM_perc_avg, mcherryCNO_REM_perc_std = sbp.avg_animals(mcherryCNO_REM_perc)

## plot the Sal v/s CNO injected groups for the mCherry v/s DREADD condition for each
##sleep stage
po.plot_perc(wt_wake_perc_avg, wt_wake_perc_std,mut_wake_perc_avg,mut_wake_perc_std, tot_hrs, 'Wake percentage', globalFont, global_name)
po.plot_perc(wt_NREM_perc_avg, wt_NREM_perc_std,mut_NREM_perc_avg,mut_NREM_perc_std,tot_hrs, 'NREM percentage', globalFont, global_name)
po.plot_perc(wt_REM_perc_avg, wt_REM_perc_std,mut_REM_perc_avg,mut_REM_perc_std,tot_hrs, 'REM percentage', globalFont, global_name)
po.plot_perc(mcherrySAL_wake_perc_avg, mcherrySAL_wake_perc_std,mcherryCNO_wake_perc_avg,mcherryCNO_wake_perc_std, tot_hrs, 'Wake percentage mcherry', globalFont, global_name)
po.plot_perc(mcherrySAL_NREM_perc_avg, mcherrySAL_NREM_perc_std,mcherryCNO_NREM_perc_avg,mut_NREM_perc_std,tot_hrs, 'NREM percentage mcherry', globalFont, global_name)
po.plot_perc(mcherrySAL_REM_perc_avg, mcherrySAL_REM_perc_std,mcherryCNO_REM_perc_avg,mcherryCNO_REM_perc_std,tot_hrs, 'REM percentage mcherry', globalFont, global_name)

## plot the CNO v/s saline condition for all 4 experimental conditions
po.plot_perc_4groups(wt_wake_perc_avg, wt_wake_perc_std,mut_wake_perc_avg,mut_wake_perc_std,mcherrySAL_wake_perc_avg, mcherrySAL_wake_perc_std,mcherryCNO_wake_perc_avg,mcherryCNO_wake_perc_std, tot_hrs, 'Wake percentage 4 groups', 'Wake %', globalFont, global_name)
po.plot_perc_4groups(wt_NREM_perc_avg, wt_NREM_perc_std,mut_NREM_perc_avg,mut_NREM_perc_std,mcherrySAL_NREM_perc_avg, mcherrySAL_NREM_perc_std,mcherryCNO_NREM_perc_avg,mcherryCNO_NREM_perc_std, tot_hrs, 'NREM percentage 4 groups','NREM %', globalFont, global_name)
po.plot_perc_4groups(wt_REM_perc_avg, wt_REM_perc_std,mut_REM_perc_avg,mut_REM_perc_std,mcherrySAL_REM_perc_avg, mcherrySAL_REM_perc_std,mcherryCNO_REM_perc_avg,mcherryCNO_REM_perc_std, tot_hrs, 'REM percentage 4 groups', 'REM %', globalFont, global_name)

##Plotting the ∆NREM sleep for individual mice
po.plotWithLines(wt_NREM_totalTime, mut_NREM_totalTime, 'NREM time total lines', globalFont, global_name)

## Plotting the total time spent in wake for each of the 4 experimental groups
wake_gqSAL_gqCNO_ttest = st.ttest_rel(wt_wake_totalTime,mut_wake_totalTime)
wake_gqSAL_gqCNO_ttest = wake_gqSAL_gqCNO_ttest[1]
wake_mcherryCNPgqCNO_ttest = st.ttest_ind(mcherryCNO_wake_totalTime,mut_wake_totalTime, equal_var = False)
wake_mcherryCNPgqCNO_ttest = wake_mcherryCNPgqCNO_ttest[1]
wake_mcherrySALgqSAL_ttest = st.ttest_ind(mcherrySAL_wake_totalTime,wt_wake_totalTime, equal_var = False)
wake_mcherrySALgqSAL_ttest = wake_mcherrySALgqSAL_ttest[1]
wake_mcherrySALmcherryCNO_ttest = st.ttest_rel(mcherrySAL_wake_totalTime,mcherryCNO_wake_totalTime)
wake_mcherrySALmcherryCNO_ttest = wake_mcherrySALmcherryCNO_ttest[1]
wake_pval = [wake_gqSAL_gqCNO_ttest,wake_mcherryCNPgqCNO_ttest,wake_mcherrySALmcherryCNO_ttest]
wake_pval = [0.0134, 0.0064, 0.9786]
po.plot_boxplot_4groups(wake_pval,wt_wake_totalTime, 'gqs', mut_wake_totalTime, 'gqc', mcherrySAL_wake_totalTime, mcherryCNO_wake_totalTime, 'Time (hr)', 'Gq + mcherry wake | sal + CNO', globalFont, global_name)

## Plotting the total time spent in NREM for each of the 4 experimental groups
NREM_gqSAL_gqCNO_ttest = st.ttest_rel(wt_NREM_totalTime,mut_NREM_totalTime)
NREM_gqSAL_gqCNO_ttest = NREM_gqSAL_gqCNO_ttest[1]
NREM_mcherryCNPgqCNO_ttest = st.ttest_ind(mcherryCNO_NREM_totalTime,mut_NREM_totalTime, equal_var = False)
NREM_mcherryCNPgqCNO_ttest = NREM_mcherryCNPgqCNO_ttest[1]
NREM_mcherrySALgqSAL_ttest = st.ttest_ind(mcherrySAL_NREM_totalTime,wt_NREM_totalTime, equal_var = False)
NREM_mcherrySALgqSAL_ttest = NREM_mcherrySALgqSAL_ttest[1]
NREM_mcherrySALmcherryCNO_ttest = st.ttest_rel(mcherrySAL_NREM_totalTime,mcherryCNO_NREM_totalTime)
NREM_mcherrySALmcherryCNO_ttest = NREM_mcherrySALmcherryCNO_ttest[1]
NREM_pval = [NREM_gqSAL_gqCNO_ttest,NREM_mcherryCNPgqCNO_ttest,NREM_mcherrySALmcherryCNO_ttest]
NREM_pval = [0.0142, 0.0066, 0.9863]
po.plot_boxplot_4groups(NREM_pval,wt_NREM_totalTime, 'gqs', mut_NREM_totalTime, 'gqc', mcherrySAL_NREM_totalTime, mcherryCNO_NREM_totalTime, 'Time (hr)', 'Gq + mcherry NREM | sal + CNO', globalFont, global_name)

## Plotting the total time spent in REM for each of the 4 experimental groups
REM_gqSAL_gqCNO_ttest = st.ttest_rel(wt_REM_totalTime,mut_REM_totalTime)
REM_gqSAL_gqCNO_ttest = REM_gqSAL_gqCNO_ttest[1]
REM_mcherryCNPgqCNO_ttest = st.ttest_ind(mcherryCNO_REM_totalTime,mut_REM_totalTime, equal_var = False)
REM_mcherryCNPgqCNO_ttest = REM_mcherryCNPgqCNO_ttest[1]
REM_mcherrySALgqSAL_ttest = st.ttest_ind(mcherrySAL_REM_totalTime,wt_REM_totalTime, equal_var = False)
REM_mcherrySALgqSAL_ttest = REM_mcherrySALgqSAL_ttest[1]
REM_mcherrySALmcherryCNO_ttest = st.ttest_rel(mcherrySAL_REM_totalTime,mcherryCNO_REM_totalTime)
REM_mcherrySALmcherryCNO_ttest = REM_mcherrySALmcherryCNO_ttest[1]
REM_pval = [REM_gqSAL_gqCNO_ttest,REM_mcherryCNPgqCNO_ttest,REM_mcherrySALmcherryCNO_ttest]
REM_pval = [0.0516, 0.0503, 0.7402]
po.plot_boxplot_4groups(REM_pval,wt_REM_totalTime, 'gqs', mut_REM_totalTime, 'gqc', mcherrySAL_REM_totalTime, mcherryCNO_REM_totalTime, 'Time (min)', 'Gq + mcherry REM | sal + CNO', globalFont, global_name)

##calculating the t test per hour for each sleep stage
GqCNO_GqSAL_wake_ttest = ls.hrByhr_ttest(wt_wake_perc, mut_wake_perc)
GqCNO_GqSAL_NREM_ttest = ls.hrByhr_ttest(wt_NREM_perc, mut_NREM_perc)
GqCNO_GqSAL_REM_ttest = ls.hrByhr_ttest(wt_REM_perc, mut_REM_perc)




