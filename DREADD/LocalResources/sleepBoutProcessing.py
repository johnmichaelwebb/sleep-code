#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 20:43:51 2022

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

bout_length = 10 ##number of seconds per bout
days = 2 ## number of days per mouse



def bout_loc(a, name):
    ## for a given 'name' finds the STARTING location of that bout
    ## this will also give the number of bouts
    ## a is the scored data, name is the sleep stage
    loc = []
    for i in range(len(a)):
        if a[i] == name and a[i-1] != name:
            loc.append(i)
    return loc

def bout_len(pos, full,  name,maxBout):
    ## pos is the bout loc, name is the sleep stage 
    ## full is the fully scored data
    ## calculates the length of the bouts
    ## and returns them in seconds
    bout_len = []
    
    for i in range(len(pos)):
        GO = True
        counter = 0
        loc = pos[i]
        while GO == True and loc < maxBout:
            counter +=1
            if full[loc] != name:
                GO = False
            loc +=1
        bout_len.append(counter-1)
    return bout_len

def find_all_len(a):
    REM_len = []
    NREM_len = []
    wake_len = []
    maxBout = len(a)
    for i in range(len(a)):
        curr = a[i]
        curr_REM_loc = bout_loc(curr, 'REM')
        curr_REM_len = bout_len(curr_REM_loc, curr, 'REM', maxBout)
        REM_len.append(curr_REM_len)
    for i in range(len(a)):
        curr = a[i]
        
        curr_NREM_loc = bout_loc(curr, 'NREM')
        curr_NREM_len = bout_len(curr_NREM_loc, curr, 'NREM',maxBout)
        NREM_len.append(curr_NREM_len)        
    for i in range(len(a)):
        curr = a[i]
        curr_wake_loc = bout_loc(curr, 'Wake')
        curr_wake_len = bout_len(curr_wake_loc, curr, 'Wake',maxBout)
        wake_len.append(curr_wake_len)                
    return REM_len, NREM_len, wake_len

# def avg_bout_len(a):
#     ## a is the bout len matrix
#     ## it finds the number of bouts as well as the average bout length
#     ##Then it averages over multiple days
#     avg_len = []
#     num_bouts = []
#     for i in range(len(a)):
#         num_bouts.append(len(a[i]))
#         curr_len = a[i]
#         curr_avg = np.mean(curr_len)
#         avg_len.append(curr_avg)
#     avg_leng_comb = []
#     num_bouts_comb = []
#     ##averages over the number of days (set in the master variables 
#     ##at the beginning)
#     for i in range(int(len(avg_len)/days)):
#         pointer = i * days
#         curr_comb_len = 0
#         curr_comb_num_bouts = 0
#         for j in range(days):
#             curr_comb_len = curr_comb_len + avg_len[pointer +j]
#         for j in range(days):
#             curr_comb_num_bouts = curr_comb_num_bouts + num_bouts[pointer +j]
#         curr_comb_len = curr_comb_len / days
#         curr_comb_num_bouts = curr_comb_num_bouts/days
#         avg_leng_comb.append(curr_comb_len)
#         num_bouts_comb.append(curr_comb_num_bouts)
#     ##convert the bout length to seconds
#     avg_len_sec = []
#     for i in range(len(avg_leng_comb)):
#         number = avg_leng_comb[i] * bout_length
#         avg_len_sec.append(number)                    
#     return num_bouts_comb, avg_len_sec

def sec2min(a):
    for i in range(len(a)):
        a[i]= a[i]/60
    return a

def min2hour(a):
    for i in range(len(a)):
        a[i] = a[i]/60
    return a

def total_sleep_time(wake,NREM, REM, mulDays = True):
    ##sum the total sleep amount every day
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
    if mulDays == True:
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
            wake_averaged[i] = wake_averaged[i]*bout_length
            NREM_averaged[i] = NREM_averaged[i]*bout_length
            REM_averaged[i] = REM_averaged[i]*bout_length
                        ## convert NREM and wake to hours
            wake_averaged = sec2min(wake_averaged)
            wake_averaged = min2hour(wake_averaged)
            NREM_averaged = sec2min(NREM_averaged)
            NREM_averaged = min2hour(NREM_averaged)
    ##convert REM time to min
            REM_averaged = sec2min(REM_averaged)   
        return wake_averaged, NREM_averaged, REM_averaged
    
    wake_averaged = wake_tot
    NREM_averaged = NREM_tot
    REM_averaged = REM_tot
    
    
    ## convert the bout length to seconds
    for i in range(len(wake_averaged)):
        wake_averaged[i] = wake_averaged[i]*bout_length
        NREM_averaged[i] = NREM_averaged[i]*bout_length
        REM_averaged[i] = REM_averaged[i]*bout_length
    ## convert NREM and wake to hours
    wake_averaged = sec2min(wake_averaged)
    wake_averaged = min2hour(wake_averaged)
    NREM_averaged = sec2min(NREM_averaged)
    NREM_averaged = min2hour(NREM_averaged)
    ##convert REM time to min
    REM_averaged = sec2min(REM_averaged)

    return wake_averaged, NREM_averaged, REM_averaged

def hour2percent(a, name, minutes):
    ##takes a given time segment and returns the percent of that segment by the hour 
    epochPerMin = 60/bout_length
    #epochsPerDivision = minutes*epochPerMin
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

def percent_by_hour(a, minutes = 60, mul_days = True):
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
    
    if mul_days == True:
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
    
    for i in range(len(data)):
       # print(data[i])
        frame[name][i+index] = data[i]
    return frame

def plotDots(wt,wtName, mut,mutName, ylabel,name):
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
    sns.boxplot(x="condition", y="cFos", data=framee, palette = ["lightgray",'lightblue'] )
    sns.swarmplot(x = "condition", y = "cFos", data = framee,palette = ["black",'darkblue'], alpha = 1, s = 10)
    sns.despine()
    plt.xlabel('')
    #plt.ylim(10,14)
    plt.ylabel(ylabel, fontsize = globalFont)
    plt.title(name, fontsize = globalFont)
    plt.savefig(name +global_name+".pdf")
    plt.show()
    return
    


def create_hrs(std, hrs):
    interval = hrs/len(std)
    matrix = []
    for i in range(len(std)):
        matrix.append(i*interval+1)
    return matrix

def plot_perc(wtAVG, wtSTD, mutAVG, mutSTD, tot_hrs,name):
    x_axis = create_hrs(wtSTD, tot_hrs)
    ax1 = plt.axes(frameon=False)
    plt.errorbar(x_axis, wtAVG, yerr = wtSTD, fmt = '-o', linewidth = 5, markersize = 12)
    plt.errorbar(x_axis, mutAVG, yerr = mutSTD, fmt = '-o',linewidth = 5, markersize = 12)
    plt.xlabel('time post CNO (hr)',fontsize = globalFont*.8)
    plt.ylabel('%',fontsize = globalFont*.8 )
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))  
    ax1.tick_params(labelsize=globalFont*.8) 
    plt.xticks(ticks = [0, 1, 2, 3,4,5],labels = ['1','2','3','4','5','6'])
    title = name
    # plt.title(title)
    plt.savefig(title + global_name+'.pdf')
    plt.show()
    return

def plot_perc_4groups(wtAVG, wtSTD, mutAVG, mutSTD, wt2AVG, wt2STD, mut2AVG, mut2STD, tot_hrs,name, ylabel):
    x_axis = create_hrs(wtSTD, tot_hrs)
    ax1 = plt.axes(frameon=False)
    plt.errorbar(x_axis, wtAVG, yerr = wtSTD, fmt = '-o', linewidth = 5, markersize = 12, c = 'paleturquoise')
    plt.errorbar(x_axis, mutAVG, yerr = mutSTD, fmt = '-o',linewidth = 5, markersize = 12, c = 'steelblue')
    plt.errorbar(x_axis, wt2AVG, yerr = wt2STD, fmt = '-o',linewidth = 5, markersize = 12, c = 'lightgreen')
    plt.errorbar(x_axis, mut2AVG, yerr = mut2STD, fmt = '-o',linewidth = 5, markersize = 12, c = 'mediumseagreen')
    plt.xlabel('Time post CNO (hr)',fontsize = globalFont*.8)
    plt.ylabel(ylabel,fontsize = globalFont*.8 )
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))  
    ax1.tick_params(labelsize=globalFont*.8)  
    title = name
    # plt.title(title)
    plt.savefig(title + global_name+'.pdf')
    plt.show()
    return

def plotWithLines(day1, day2, name):
    x_axis = [1,2]
    x_names = ['','sal', '','','','','CNO']
    ## create y axes
    y_axis = []
    ax1 = plt.axes(frameon=False)
    for i in range(len(day1)):
        y_curr = []
        y_curr.append(day1[i])
        y_curr.append(day2[i])
        y_axis.append(y_curr)
    for i in range(len(y_axis)):
        plt.plot(x_axis, y_axis[i], 'ro-')
    #plt.xlabel('',fontsize = globalFont*.8)
    plt.ylabel('time (hr)',fontsize = globalFont*.8 )
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))  
    ax1.tick_params(labelsize=globalFont*.8)    
    ax1.set_xticklabels(x_names)

    plt.title(name)
    plt.savefig(name + global_name+'.pdf')
    plt.show()
    return
    
def createHalf(a):
    stop = int(len(a[0])/2)
    for i in range(len(a)):
        a[i] = a[i][0:stop]
    return a

def plot_boxplot_4groups(pval, wt,wtName, mut,mutName, wt_2,mut_2, ylabel,name):
    labels = []
    for i in range(len(wt)):
        labels.append('Gq Sal')
    for i in range(len(mut)):
        labels.append('Gq CNO')
    for i in range(len(wt_2)):
        labels.append('mCherry Sal')
    for i in range(len(mut_2)):
        labels.append('mCherry CNO')
    height = len(wt) + len(mut) + len(wt_2) + len(mut_2)
    framee = pd.DataFrame(np.random.randint(low=0.0, high= 100, size=(height, 3)), columns=['cFos','cFos_2', 'condition'],dtype = np.float64)
    curr_length = 0
    framee = createDataFrame(framee, wt, 'cFos', curr_length)
    curr_length = curr_length + len(wt)
    framee = createDataFrame(framee, mut, 'cFos', curr_length)
    curr_length = curr_length + len(mut)
    framee = createDataFrame(framee, wt_2, 'cFos', curr_length)
    curr_length = curr_length + len(wt_2)
    framee = createDataFrame(framee, mut_2, 'cFos', curr_length)
    framee = createDataFrame(framee, labels, 'condition',0)
    sns.set_style("ticks")
    sns.set(font_scale = 1)
    sns.set_style("ticks")
    order = ['Gq Sal','Gq CNO','mCherry Sal','mCherry CNO']
    ax1 = sns.boxplot(x="condition", y="cFos", data=framee, palette = ["paleturquoise",'steelblue','lightgreen','mediumseagreen'] )
    sns.swarmplot(x = "condition", y = "cFos", data = framee, color = 'darkgray', alpha = 1, s = 5)
    sns.despine()
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont)
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