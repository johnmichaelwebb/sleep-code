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
from statannot import add_stat_annotation
import matplotlib
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
##############################################
##############################################
##############################################
##############################################
name_score = 'RTG EEG scores.csv' ##user scored epochs
sleep_dep_name = 'RTGcaspSD.csv' ##sleep deprivation epochs
days = 2 ## number of days per mouse
bout_length = 10 ##number of seconds per bout
minutes_per_display = 60 ## number of minutes to display in the hr-by-hr graph
globalFont = 20 ## sets font size for figures
global_name = '_RTG caspase SD' ## attaches this name to the end of every file
MUTANT_NAME = 'Cre' ## x label for experimental mice
WILDTYPE_NAME = 'WT' ## x label for WT


def extract_column(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(a[i][column])
    return new_matrix 

def matrix_float(a):
    #converts a matrix of strings to floats
    ##often necessary when importing the data csv files into python
    new_matrix = []
    for i in range(len(a)):
        try:
            new_matrix.append((float(a[i])))
        except ValueError:
            print(i)
            print("matrixFloat")
    return new_matrix

def extract_row(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a[0])):
        new_matrix.append(a[column][i])
    return new_matrix 

def create_empty_matrix(a, label,data):
    ## creates an empty matrix of length a 
    ## puts data into it and attaches label to it
    new_matrix = []
    for i in range(len(a)):
        if a[i] == label:
            column = extract_column(data, i)
            new_matrix.append(column)
    return new_matrix

def bout_loc(a, name):
    ## for a given 'name' finds the STARTING location of that bout
    ## this will also give the number of bouts
    ## a is the scored data, name is the sleep stage
    loc = []
    for i in range(len(a)):
        if a[i] == name and a[i-1] != name:
            loc.append(i)
    return loc

def bout_len(pos, full,  name):
    ## pos is the bout loc, name is the sleep stage 
    ## full is the fully scored data
    ## calculates the length of the bouts
    ## and returns them in seconds
    bout_len = []
    
    for i in range(len(pos)):
        GO = True
        counter = 0
        loc = pos[i]
        while GO == True and loc < max_bout:
            counter +=1
            if full[loc] != name:
                GO = False
            loc +=1
        bout_len.append(counter-1)
    return bout_len

def find_all_len(a):
    ## essentially finds the REM, NREM and Wake bout lengths from the raw data
    REM_len = []
    NREM_len = []
    wake_len = []
    for i in range(len(a)):
        curr = a[i]
        curr_REM_loc = bout_loc(curr, 'REM')
        curr_REM_len = bout_len(curr_REM_loc, curr, 'REM')
        REM_len.append(curr_REM_len)
    for i in range(len(a)):
        curr = a[i]
        
        curr_NREM_loc = bout_loc(curr, 'NREM')
        curr_NREM_len = bout_len(curr_NREM_loc, curr, 'NREM')
        NREM_len.append(curr_NREM_len)        
    for i in range(len(a)):
        curr = a[i]
        curr_wake_loc = bout_loc(curr, 'Wake')
        curr_wake_len = bout_len(curr_wake_loc, curr, 'Wake')
        wake_len.append(curr_wake_len)                
    return REM_len, NREM_len, wake_len

def avg_bout_len(a):
    ## a is the bout len matrix
    ## it finds the number of bouts as well as the average bout length
    ## then it averages over multiple days
    avg_len = []
    num_bouts = []
    for i in range(len(a)):
        num_bouts.append(len(a[i]))
        curr_len = a[i]
        curr_avg = np.mean(curr_len)
        avg_len.append(curr_avg)
    avg_leng_comb = []
    num_bouts_comb = []
    ## averages over the number of days (set in the master variables 
    ## at the beginning)
    for i in range(int(len(avg_len)/days)):
        pointer = i * days
        curr_comb_len = 0
        curr_comb_num_bouts = 0
        for j in range(days):
            curr_comb_len = curr_comb_len + avg_len[pointer +j]
        for j in range(days):
            curr_comb_num_bouts = curr_comb_num_bouts + num_bouts[pointer +j]
        curr_comb_len = curr_comb_len / days
        curr_comb_num_bouts = curr_comb_num_bouts/days
        avg_leng_comb.append(curr_comb_len)
        num_bouts_comb.append(curr_comb_num_bouts)
    ##convert the bout length to seconds
    avg_len_sec = []
    for i in range(len(avg_leng_comb)):
        number = avg_leng_comb[i] * bout_length
        avg_len_sec.append(number)                    
    return num_bouts_comb, avg_len_sec

def sec2min(a):
    ## converts seconds to minutes
    for i in range(len(a)):
        a[i]= a[i]/60
    return a

def min2hour(a):
    ## converts minutes to hours
    for i in range(len(a)):
        a[i] = a[i]/60
    return a

def total_sleep_time(wake,NREM, REM):
    ## sum the total sleep amount every day
    ## averages over numbers of days
    wake_tot = []
    NREM_tot =[]
    REM_tot = []
    for i in range(len(wake)):
        wake_tot.append(np.sum(wake[i]))
        NREM_tot.append(np.sum(NREM[i]))
        REM_tot.append(np.sum(REM[i]))
    wake_averaged =[]
    NREM_averaged =[]
    REM_averaged=[]
    ## average across the number of days (set globally; ususally 2)
    for i in range(int(len(wake_tot)/days)):
        pointer = i * days
        curr_comb_REM = 0
        curr_comb_NREM = 0
        curr_comb_wake = 0
        for j in range(days):
            curr_comb_REM = curr_comb_REM + REM_tot[pointer +j]
            curr_comb_NREM = curr_comb_NREM + NREM_tot[pointer +j]
            curr_comb_wake = curr_comb_wake + wake_tot[pointer +j]
        curr_comb_REM = curr_comb_REM / days
        curr_comb_NREM = curr_comb_NREM/days
        curr_comb_wake = curr_comb_wake/days
        wake_averaged.append(curr_comb_wake)
        NREM_averaged.append(curr_comb_NREM)
        REM_averaged.append(curr_comb_REM)
    ## convert the bout length to seconds
    for i in range(len(wake_averaged)):
        wake_averaged[i] = wake_averaged[i]*10
        NREM_averaged[i] = NREM_averaged[i]*10
        REM_averaged[i] = REM_averaged[i]*10
    ## convert NREM and wake to hours
    wake_averaged = sec2min(wake_averaged)
    wake_averaged = min2hour(wake_averaged)
    NREM_averaged = sec2min(NREM_averaged)
    NREM_averaged = min2hour(NREM_averaged)
    ## convert REM time to min
    REM_averaged = sec2min(REM_averaged)
    return wake_averaged, NREM_averaged, REM_averaged

def hour2percent(a, name, minutes):
    ## takes a given time segment and returns the percent of that segment by the hour 
    epochPerMin = 60/bout_length
    divisor = epochPerMin*minutes
    numerator = 0
    for i in range(len(a)):
        if a[i] == name:
            numerator+=1
    if divisor !=0:
        perc = numerator/divisor*100
    else:
        perc = 0
    return perc

def percent_by_hour(a, minutes = 60, multiple_days = True):
    ##this takes the raw scoring and converts it to a percent by hour
    ## first break up each day into one hour chunks
    ##the time it's broken up by is determined by minutes the variable
    totSec = (len(a[0])-2)*bout_length
    totMin = totSec/60
    divisions = int(totMin/minutes)
    epochPerMin = 60/bout_length
    epochsPerDivision = minutes*epochPerMin
    wake_perc_tot = []
    NREM_perc_tot = []
    REM_perc_tot = []
    for i in range(len(a)):
        wake_perc = []
        NREM_perc = []
        REM_perc = []
        for j in range(divisions):
            start = int(j*epochsPerDivision)
            stop = int((j+1)*epochsPerDivision)
            toAppend = a[i][start:stop]
            curr_REM = hour2percent(toAppend, 'REM', minutes = minutes)
            curr_NREM = hour2percent(toAppend, 'NREM', minutes = minutes)
            curr_wake = hour2percent(toAppend, 'Wake', minutes = minutes)
            wake_perc.append(curr_wake)
            NREM_perc.append(curr_NREM)
            REM_perc.append(curr_REM)
        wake_perc_tot.append(wake_perc)
        NREM_perc_tot.append(NREM_perc)
        REM_perc_tot.append(REM_perc)
    wake_averaged =[]
    NREM_averaged =[]
    REM_averaged=[]    
    
    if multiple_days == True:
        for i in range(int(len(wake_perc_tot)/days)):
            pointer = i * days
            REM_hold = []
            NREM_hold = []
            wake_hold = []
            for j in range(len(wake_perc_tot[0])):
                curr_comb_REM = 0
                curr_comb_NREM = 0
                curr_comb_wake = 0
                for k in range(days):
                        curr_comb_REM = curr_comb_REM + REM_perc_tot[pointer +k][j]
                        curr_comb_NREM = curr_comb_NREM + NREM_perc_tot[pointer +k][j]
                        curr_comb_wake = curr_comb_wake + wake_perc_tot[pointer +k][j]
                curr_comb_REM = curr_comb_REM / days
                curr_comb_NREM = curr_comb_NREM/days
                curr_comb_wake = curr_comb_wake/days
                REM_hold.append(curr_comb_REM)
                NREM_hold.append(curr_comb_NREM)
                wake_hold.append(curr_comb_wake)
            wake_averaged.append(wake_hold)
            NREM_averaged.append(NREM_hold)
            REM_averaged.append(REM_hold)        
        return wake_averaged, NREM_averaged, REM_averaged 
    return wake_perc_tot, NREM_perc_tot, REM_perc_tot

def createDataFrame(frame, data, name, index):
    ## creates a pd dataframe given the input variables 
    for i in range(len(data)):
        frame[name][i+index] = data[i]
    return frame

def plotDots(wt,wtName, mut,mutName, ylabel,name):
    ## plots a boxplot with a swarmplot for a give dataset
    ## wt is wt data, wtName is the x label for wt
    ## mut is mut data, mutName is x label of mut
    ## ylabel labels y axis, name is attached to the title and savename
    labels = []
    for i in range(len(wt)):
        labels.append(wtName)
    for i in range(len(mut)):
        labels.append(mutName)
    height = len(wt) + len(mut)
    framee = pd.DataFrame(np.random.randint(low=0.0, high= 100, size=(height, 2)), columns=['cFos', 'condition'],dtype = np.float64)
    framee = createDataFrame(framee, wt, 'cFos', 0)
    framee = createDataFrame(framee, mut, 'cFos', len(wt))
    framee = createDataFrame(framee, labels, 'condition',0)
    sns.set_style("ticks")
    sns.set(font_scale = 1.4)
    sns.set_style("ticks")
    plt.tight_layout()
    g2 = sns.boxplot(x="condition", y="cFos", data=framee, palette = ["lightgray",'dodgerblue'], linewidth =1)
    sns.swarmplot(x = "condition", y = "cFos", data = framee,palette = ["dimgray",'dimgray'], alpha = 1, s = 8)
    sns.despine()
    # order = [wtName, mutName]
    # test_results = add_stat_annotation(g2, data=framee, x='condition', y='cFos', order=order,
    #                                box_pairs=[(wtName, mutName)],
    #                                test='t-test_ind', text_format='star',
    #                                loc='outside', verbose=2)
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont)
    plt.tight_layout()

    # plt.title(name, fontsize = globalFont)
    plt.savefig(name +global_name+".pdf")
    plt.show()
    return
    
def avg_animals(a):
    avg_matrix = []
    std_matrix = []
    for i in range(len(a[0])):
        curr_total = []
        for j in range(len(a)):
            curr_total.append(a[j][i])
        avg_matrix.append(np.mean(curr_total))
        std_matrix.append(np.std(curr_total))
    return avg_matrix, std_matrix   

def avg_animals_debt(a):
    ## calculates the average and std of sleep debt given totals for every group
    avg_matrix = [0]
    std_matrix = [0]
    for i in range(len(a[0])):
        curr_total = []
        for j in range(len(a)):
            curr_total.append(a[j][i])
        avg_matrix.append(np.mean(curr_total))
        std_matrix.append(np.std(curr_total))
    return avg_matrix, std_matrix 

def create_hrs(std):
    ## creates 24 hour matrix
    interval = 24/len(std)
    matrix = []
    for i in range(len(std)):
        matrix.append(i*interval)
    return matrix

def plot_perc(ylabel, wtAVG, wtSTD, mutAVG, mutSTD,WT_NAME,MUT_NAME, name):
    ## plot percent by hour
    x_axis = create_hrs(wtSTD)
    ax1 = plt.axes(frameon=False)
    plt.errorbar(x_axis, wtAVG, yerr = wtSTD, fmt = '-o', c = 'darkgray')
    plt.errorbar(x_axis, mutAVG, yerr = mutSTD, fmt = '-o', c= 'dodgerblue')
    plt.xlabel('Time (hr)',fontsize = globalFont*.8)
    plt.ylabel(ylabel,fontsize = globalFont*.8 )
    ax1.tick_params(axis = 'both', which = 'major', labelsize = globalFont*.6)
    plt.xticks(ticks = [0, 6, 12, 18,24],labels = ['0','6','12','18','24'])
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((0, 0), (ymin, ymax), color='black', linewidth=2))  
    plt.xlim(0,24)
    title = name
    plt.tight_layout()
    # plt.title(title)
    plt.savefig(title + global_name+'.pdf')
    plt.show()
    return

def LDsplit(a):
    ## breaks up activity by light and dark    
    light = []
    dark = []
    for i in range(len(a)):
        midpt = int((len(a[0]))/2+1)
        end = len(a[0])
        currLight = a[i][2:midpt]
        currDark = a[i][midpt:end]
        light.append(currLight)
        dark.append(currDark)
    return light, dark

def replaceSDwithWake(a, name, limit):
    ## a is the scored data
    ## name is the thing to replace it with
    ## limit = the hours of sleep deprivation
    ##f irst calculate the number of slots to covert
    totBouts = int(limit*60*60/bout_length)
    for i in range(len(a)):
        for j in range(totBouts):
            ## the +2 is becuase the data has 2662 plus the label liek Cre or WT
            a[i][j+2] = name
    return a

def perc2min(wake,NREM,REM, timeBin = 60):
    ## converts from percent of sleep to minutes
    wakeTot = []
    NREMtot = []
    REMtot = []
    for i in range(len(wake)):
        wake_new = []
        NREM_new = []
        REM_new = []
        for j in range(len(wake[0])):
            wake_new.append(timeBin*wake[i][j]/100)
            NREM_new.append(timeBin*NREM[i][j]/100)
            REM_new.append(timeBin*REM[i][j]/100)
        wakeTot.append(wake_new)
        NREMtot.append(NREM_new)
        REMtot.append(REM_new)
    return wakeTot,NREMtot,REMtot

def normalizeSleepDebt(a, Debtpos = 5):
    #  normalizes to the position of greatest debt
    ## in this case 5 hours so its a five
     greatestDebt = a[Debtpos-1]
     for i in range(len(a)):
         a[i] = -a[i]/greatestDebt*100
     return a

def sleepDebt(norm, SD):
    ## calculates sleep debt
    tot_debt = []
    for i in range(len(norm)):
        curr_diff = []
        for j in range(len(norm[0])):
            immediate_diff = SD[i][j] - norm[i][j]
            curr_diff.append(immediate_diff)
        tracking = 0
        debt = []
        for i in range(len(curr_diff)):
            tracking = curr_diff[i] + tracking
            debt.append(tracking)
        debt = normalizeSleepDebt(debt)
        tot_debt.append(debt)
    return tot_debt

with open(name_score, newline='\n' ) as inputfile:
   data = list(csv.reader(inputfile)) 
   data[0][0] = data[0][0][1:len(data[0][0])]
with open(sleep_dep_name, newline='\n' ) as inputfile:
   data_SD = list(csv.reader(inputfile)) 
   data_SD[0][0] = data_SD[0][0][1:len(data_SD[0][0])]
names = extract_row(data,1)
Cre = create_empty_matrix(names, MUTANT_NAME, data)
WT = create_empty_matrix(names, WILDTYPE_NAME,data)
wild = WT[:]
wildy = WT[:]
mutant = Cre[:]
mutanty = Cre[:]
max_bout = len(WT[0])-2
wt_wake_perc, wt_NREM_perc, wt_REM_perc = percent_by_hour(wild, minutes = minutes_per_display)
mut_wake_perc, mut_NREM_perc, mut_REM_perc = percent_by_hour(mutant,minutes = minutes_per_display)
wt_wake_min, wt_NREM_min, wt_REM_min = perc2min(wt_wake_perc, wt_NREM_perc, wt_REM_perc)
mut_wake_min, mut_NREM_min, mut_REM_min = perc2min(mut_wake_perc, mut_NREM_perc, mut_REM_perc)
names_SD = extract_row(data_SD,1)
Cre_SD = create_empty_matrix(names_SD, MUTANT_NAME, data_SD)
WT_SD = create_empty_matrix(names_SD, WILDTYPE_NAME, data_SD)
max_bout = len(WT_SD[0])-2
WT_SD = replaceSDwithWake(WT_SD, 'Wake', 5)
Cre_SD = replaceSDwithWake(Cre_SD, 'Wake',5)
wild_SD = WT_SD[:]              
             
wt_SD_wake_perc, wt_SD_NREM_perc, wt_SD_REM_perc = percent_by_hour(WT_SD, minutes = minutes_per_display, multiple_days = False)
mut_SD_wake_perc, mut_SD_NREM_perc, mut_SD_REM_perc = percent_by_hour(Cre_SD,minutes = minutes_per_display, multiple_days = False)
wt_SD_wake_min, wt_SD_NREM_min, wt_SD_REM_min = perc2min(wt_SD_wake_perc, wt_SD_NREM_perc, wt_SD_REM_perc)
mut_SD_wake_min, mut_SD_NREM_min, mut_SD_REM_min = perc2min(mut_SD_wake_perc, mut_SD_NREM_perc, mut_SD_REM_perc)
wt_NREM_debt = sleepDebt(wt_NREM_min, wt_SD_NREM_min)
mut_NREM_debt = sleepDebt(mut_NREM_min,mut_SD_NREM_min)
wt_REM_debt = sleepDebt(wt_REM_min, wt_SD_REM_min)
mut_REM_debt = sleepDebt(mut_REM_min, mut_SD_REM_min)
wt_NREM_debt_avg, wt_NREM_debt_std = avg_animals_debt(wt_NREM_debt)
wt_REM_debt_avg, wt_REM_debt_std = avg_animals_debt(wt_REM_debt)
mut_NREM_debt_avg, mut_NREM_debt_std = avg_animals_debt(mut_NREM_debt)
mut_REM_debt_avg, mut_REM_debt_std = avg_animals_debt(mut_REM_debt)
plot_perc('% NREM recovery',wt_NREM_debt_avg, wt_NREM_debt_std,mut_NREM_debt_avg,mut_NREM_debt_std, WILDTYPE_NAME, MUTANT_NAME, 'NREM sleep debt')
plot_perc('% REM recovery',wt_REM_debt_avg, wt_REM_debt_std,mut_REM_debt_avg,mut_REM_debt_std,WILDTYPE_NAME, MUTANT_NAME,  'REM sleep debt')
final_wt_NREM_dep = extract_column(wt_NREM_debt,23)
final_mut_NREM_dep = extract_column(mut_NREM_debt,23)
final_wt_REM_dep = extract_column(wt_REM_debt,23)
final_mut_REM_dep = extract_column(mut_REM_debt,23)
plotDots(final_wt_NREM_dep, WILDTYPE_NAME, final_mut_NREM_dep, MUTANT_NAME, 'Final NREM recovery %', 'final NREM sleep recovery')
plotDots(final_wt_REM_dep, WILDTYPE_NAME, final_mut_REM_dep, MUTANT_NAME, 'Final REM recovery %', 'final REM sleep recovery')