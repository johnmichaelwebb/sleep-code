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
import scipy.stats as st
from matplotlib.lines import Line2D
import seaborn as sns
from statannot import add_stat_annotation
import pandas as pd
import matplotlib
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
##############################################
##############################################
##############################################
##############################################
name_score = 'RTG EEG scores.csv' ## user scored epochs
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

def create_hrs(hrs = hours):
    ##creates the number of hours i a day
    matrix = []
    for i in range(hrs):
        matrix.append(i+1)
    return matrix

def extract_column(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(a[i][column])
    return new_matrix 

def matrix_float(a):
    ## converts a matrix of strings to floats
    ## often necessary when importing the data csv files into python
    new_matrix = []
    for i in range(len(a)):
        try:
            new_matrix.append((float(a[i])))
        except ValueError:
            print(i)
            print("matrixFloat")
    return new_matrix

def extract_row(a, column):
    # take a single column from a more complex matrix
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

def percent_by_hour(a, minutes = 60):
    ## this takes the raw scoring and converts it to a percent by hour
    ## first break up each day into one hour chunks
    ## the time it's broken up by is determined by minutes the variable
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

def createDataFrame(frame, data, name, index):
    ## creates a pd dataframe given the input variables 
    for i in range(len(data)):
        frame[name][i+index] = data[i]
    return frame

def plotDots(wt,wtName, mut,mutName, ylabel,name, wake = False):
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

    g2 = sns.boxplot(x="condition", y="cFos", data=framee, palette = ["lightgray",'dodgerblue'], linewidth =1)
    sns.swarmplot(x = "condition", y = "cFos", data = framee,palette = ["dimgray",'dimgray'], alpha = 1, s = 8)
    sns.despine()
    # order = [wtName, mutName]
    # test_results = add_stat_annotation(g2, data=framee, x='condition', y='cFos', order=order,
    #                                box_pairs=[(wtName, mutName)],
    #                                test='t-test_ind', text_format='star',
    #                                loc='outside', verbose=2)
    plt.tight_layout()
    if wake == True:
        plt.yticks(ticks = [12, 14, 16],labels = ['12','14','16'])
        
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont*1.2)
    # plt.title(name, fontsize = globalFont)
    plt.savefig(name +global_name+".pdf")
    plt.show()
    return

def plot_barplot(wt,wtName, mut,mutName, ylabel,name):
    ## creates a barplot for 2 groups
    labels = []
    for i in range(len(wt)):
        labels.append(wtName)
    for i in range(len(mut)):
        labels.append(mutName)
    height = len(wt) + len(mut)
    framee = pd.DataFrame(np.random.randint(low=0.0, high= 100, size=(height, 3)), columns=['cFos','cFos_2', 'condition'],dtype = np.float64)
    framee = createDataFrame(framee, wt, 'cFos', 0)
    framee = createDataFrame(framee, mut, 'cFos', len(wt))
    framee = createDataFrame(framee, labels, 'condition',0)
    sns.set_style("ticks")
    sns.set(font_scale = 1.4)
    sns.set_style("ticks")
    g1=  sns.barplot(x="condition", y="cFos", data=framee,capsize=.1, errcolor = 'red',palette = ["lightgray",'dodgerblue'], linewidth =1 )
    g1.set(xlabel = None)
    g1.set(xticklabels=[])  # remove the tick labels
    g2=  sns.swarmplot(x = "condition", y = "cFos", data = framee, palette = ["dimgray",'dimgray'], alpha = 1, s = 5)
    g2.set(xlabel= None)
    g2.set(xticklabels=[])  # remove the tick labels
    g2.tick_params(bottom=False)  # remove the ticks    
    sns.despine()
    # order = [wtName, mutName]
    # test_results = add_stat_annotation(g2, data=framee, x='condition', y='cFos', order=order,
    #                                box_pairs=[(wtName, mutName)],
    #                                test='t-test_ind', text_format='star',
    #                                loc='outside', verbose=2)
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont)
    # plt.title(name, fontsize = globalFont)
    plt.savefig(name +global_name+".pdf")
    plt.show()
    return 

def plot_barplot_REM(wt,wtName, mut,mutName, ylabel,name):
    ## creates a barplot for 2 groups
    labels = []
    for i in range(len(wt)):
        labels.append(wtName)
    for i in range(len(mut)):
        labels.append(mutName)
    height = len(wt) + len(mut)
    framee = pd.DataFrame(np.random.randint(low=0.0, high= 100, size=(height, 3)), columns=['cFos','cFos_2', 'condition'],dtype = np.float64)
    framee = createDataFrame(framee, wt, 'cFos', 0)
    framee = createDataFrame(framee, mut, 'cFos', len(wt))
    framee = createDataFrame(framee, labels, 'condition',0)
    sns.set_style("ticks")
    sns.set(font_scale = 2.4)
    sns.set_style("ticks")
    g1=  sns.barplot(x="condition", y="cFos", data=framee,capsize=.2, errcolor = 'red',palette = ["lightgray",'dodgerblue'], linewidth =2 )
    g1.set(xlabel = None)
    g1.set(xticklabels=[])  # remove the tick labels
    g2=  sns.swarmplot(x = "condition", y = "cFos", data = framee, palette = ["dimgray",'dimgray'], alpha = 1, s = 10)
    g2.set(xlabel= None)
    g2.set(xticklabels=[])  # remove the tick labels
    g2.tick_params(bottom=False)  # remove the ticks    
    sns.despine()
    # order = [wtName, mutName]
    # test_results = add_stat_annotation(g2, data=framee, x='condition', y='cFos', order=order,
    #                                box_pairs=[(wtName, mutName)],
    #                                test='t-test_ind', text_format='star',
    #                                loc='outside', verbose=2)
    plt.yticks(ticks = [0, 50, 100],labels = ['0','50','100'])

    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont*2)
    plt.tight_layout()
    # plt.title(name, fontsize = globalFont)
    plt.savefig(name +global_name+".pdf")
    plt.show()
    return 
def plot_barplot_4groups(wt,wtName, mut,mutName, wt_2,mut_2, ylabel,name):
    ## creates a barplot for 4 groups
    labels = []
    wt2Name = wtName + str('_2')
    mut2Name = mutName + str('_2')
    for i in range(len(wt)):
        labels.append(wtName)
    for i in range(len(mut)):
        labels.append(mutName)
    for i in range(len(wt_2)):
        labels.append(wt2Name)
    for i in range(len(mut_2)):
        labels.append(mut2Name)
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
    sns.set(font_scale = 1.4)
    sns.set_style("ticks")
    g1=  sns.barplot(x="condition", y="cFos", data=framee,capsize=.1, errcolor = 'red',palette = ["lightgray",'dodgerblue'], linewidth= 1 )
    g1.set(xlabel = None)
    g1.set(xticklabels=[])  # remove the tick labels
    g2=  sns.swarmplot(x = "condition", y = "cFos", data = framee, palette = ["dimgray",'dimgray'], alpha = 1, s = 5)
    g2.set(xlabel= None)
    g2.set(xticklabels=[])  # remove the tick labels
    g2.tick_params(bottom=False)  # remove the ticks 
    # order = [wtName, mutName, wt2Name, mut2Name]
    # test_results = add_stat_annotation(g2, data=framee, x='condition', y='cFos', order=order,
    #                                box_pairs=[(wtName, mutName),(wt2Name, mut2Name)],
    #                                test='t-test_ind', text_format='star',
    #                                loc='outside', verbose=2)
    sns.despine()
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont)
    # plt.title(name, fontsize = globalFont)
    plt.savefig(name +global_name+".pdf")
    plt.show()
    return

def plot_barplot_6groups(wt,wtName, mut,mutName, wt_2,mut_2,wt_3, mut_3, ylabel,name):
    ## creates a barplot for 6 groups
    labels = []
    wt2Name = wtName + str('_2')
    mut2Name = mutName + str('_2')
    wt3Name = wtName + str('_3')
    mut3Name = mutName + str('_3')
    for i in range(len(wt)):
        labels.append(wtName)
    for i in range(len(mut)):
        labels.append(mutName)
    for i in range(len(wt_2)):
        labels.append(wt2Name)
    for i in range(len(mut_2)):
        labels.append(mut2Name)
    for i in range(len(wt_3)):
        labels.append(wt3Name)
    for i in range(len(mut_3)):
        labels.append(mut3Name)
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
    curr_length = curr_length + len(mut_2)
    framee = createDataFrame(framee, wt_3, 'cFos', curr_length)
    curr_length = curr_length + len(wt_3)
    framee = createDataFrame(framee, mut_3, 'cFos', curr_length)    
    framee = createDataFrame(framee, labels, 'condition',0)
    sns.set_style("ticks")
    sns.set(font_scale = 1.4)
    sns.set_style("ticks")
    g1=  sns.barplot(x="condition", y="cFos", data=framee,capsize=.1, errcolor = 'red',palette = ["lightgray",'dodgerblue'], linewidth =1 )
    g1.set(xlabel = None)
    g1.set(xticklabels=[])  # remove the tick labels
    g2=  sns.swarmplot(x = "condition", y = "cFos", data = framee, palette = ["dimgray",'dimgray'], alpha = 1, s = 5)
    g2.set(xlabel= None)
    g2.set(xticklabels=[])  # remove the tick labels
    g2.tick_params(bottom=False)  # remove the ticks    
    # order = [wtName, mutName, wt2Name, mut2Name, wt3Name, mut3Name]
    # test_results = add_stat_annotation(g2, data=framee, x='condition', y='cFos', order=order,
    #                                box_pairs=[(wtName, mutName),(wt2Name, mut2Name), (wt3Name, mut3Name)],
    #                                test='t-test_ind', text_format='star',
    #                                loc='outside', verbose=2)
    sns.despine()
    plt.xlabel('')
    plt.ylim(0,150)
    plt.ylabel(ylabel, fontsize = globalFont)
    # plt.title(name, fontsize = globalFont)
    plt.savefig(name +global_name+".pdf")
    plt.show()
    return
    
def avg_animals(a):
    ## find the avg and std for a matrix of length n animals 
    avg_matrix = []
    std_matrix = []
    for i in range(len(a[0])):
        curr_total = []
        for j in range(len(a)):
            curr_total.append(a[j][i])
        avg_matrix.append(np.mean(curr_total))
        std_matrix.append(np.std(curr_total))
    return avg_matrix, std_matrix   

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

def bout2sec(a):
    ## calculate bout length from number of bouts
    for i in range(len(a)):
        a[i] = a[i]*bout_length/60
    return a

def combineMatrix(a):
    ## combine many matrixes into a single matrix
    matrix = []
    for i in range(len(a)):
        for j in range(len(a[i])):
            matrix.append(a[i][j])
    matrix = bout2sec(matrix)
    return matrix

def plotHist(wt, mut, xname, yname, title, wtname, mutname, BIN = 30):
    ## plot histogram of state bout lengths
    wt = combineMatrix(wt)
    mut = combineMatrix(mut)
    ax1 = plt.axes(frameon=False)
    plt.hist(wt, bins = BIN, normed = True, histtype = 'step', lw = 6, label = wtname, color = 'black', range = (0,30))
    plt.hist(mut, bins = BIN,normed = True, histtype = 'step', lw = 6, label = mutname, color = 'deeppink', range = (0,30))
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))  
    ax1.legend(loc = 'upper right', frameon = False, prop = {'size':20})
    plt.title(title)    
    plt.xlabel(xname,fontsize = globalFont*.8)
    plt.ylabel(yname,fontsize = globalFont*.8)    
    plt.savefig(title + global_name+'.pdf')
    plt.show()
    return

def plotPerc(wtAVG, wtSTD, mutAVG, mutSTD,name, wt_name, mut_name, yname): 
    ## plot percent by hour
    x_axis = create_hrs()    
    ax1 = plt.axes(frameon=False)
    plt.errorbar(x_axis, wtAVG, yerr = wtSTD, c = 'darkgray', fmt = '-o',linewidth = 3, markersize = 8, label = wt_name)
    plt.errorbar(x_axis, mutAVG, yerr = mutSTD, c = 'dodgerblue', fmt = '-o',linewidth = 3, markersize = 8, label = mut_name)
    plt.xlabel('ZT', fontsize = globalFont*1.3)
    plt.ylabel(yname,fontsize = globalFont*1.3)
    plt.xticks(ticks = [0, 6, 12, 18,24],labels = ['0','6','12','18','24'])
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))  
    ax1.tick_params(labelsize=globalFont*1.0)  
    title = name
    plt.tight_layout()
    # plt.title(title, fontsize = globalFont*.5)
    plt.savefig(name + '_' + global_name+'.pdf')
    plt.show()
    return

def prob_single_state(a, STATE):
    ## find the probability of going from one state to another for one mouse
    a = a[2:len(a)]
    to_wake = []
    to_NREM = []
    to_REM = []
    for i in range(len(a)-1):
        before = a[i]
        after = a[i+1]
        if before == STATE:
            if after == 'Wake':
                to_wake.append(1)
            elif after == 'NREM':
                to_NREM.append(1)
            elif after == 'REM':
                to_REM.append(1)
    
    tot_trans = len(to_wake) + len(to_NREM) + len(to_REM)
    wake_prob_perc = len(to_wake)/tot_trans *100
    NREM_prob_perc = len(to_NREM)/tot_trans *100
    REM_prob_perc = len(to_REM)/tot_trans *100
    return wake_prob_perc,NREM_prob_perc,REM_prob_perc

def prob_state_all(a):
    ## find the probability of going from one state to another for all mice
    NREM_to_NREM = []
    NREM_to_wake = []
    NREM_to_REM = []
    wake_to_wake = []
    wake_to_NREM = []
    wake_to_REM = []
    REM_to_wake = []
    REM_to_NREM = []
    REM_to_REM = []
    for i in range(len(a)):
        curr_NREM_to_wake, curr_NREM_to_NREM, curr_NREM_to_REM = prob_single_state(a[i], 'NREM')
        NREM_to_wake.append(curr_NREM_to_wake)
        NREM_to_NREM.append(curr_NREM_to_NREM)
        NREM_to_REM.append(curr_NREM_to_REM)
    for i in range(len(a)):
        curr_wake_to_wake, curr_wake_to_NREM, curr_wake_to_REM = prob_single_state(a[i], 'Wake')
        wake_to_wake.append(curr_wake_to_wake)
        wake_to_NREM.append(curr_wake_to_NREM)
        wake_to_REM.append(curr_wake_to_REM)
    for i in range(len(a)):
        curr_REM_to_wake, curr_REM_to_NREM, curr_REM_to_REM = prob_single_state(a[i], 'REM')
        REM_to_wake.append(curr_REM_to_wake)
        REM_to_NREM.append(curr_REM_to_NREM)
        REM_to_REM.append(curr_REM_to_REM)
    return wake_to_wake, wake_to_NREM, wake_to_REM, NREM_to_wake, NREM_to_NREM, NREM_to_REM, REM_to_wake, REM_to_NREM, REM_to_REM

def ttest_by_hr(wt, mut):
    ## generate the t test by hour
    t = []
    for i in range(len(wt[0])):
        wt_curr = extract_column(wt, i)
        mut_curr = extract_column(mut,i)
        curr_t = st.ttest_ind(wt_curr,mut_curr, equal_var = False)
        curr_t = curr_t[1]
        t.append(curr_t)
    return t



with open(name_score, newline='\n' ) as inputfile:
   data = list(csv.reader(inputfile)) 
   data[0][0] = data[0][0][1:len(data[0][0])]
   

names = extract_row(data,1)
Cre = create_empty_matrix(names, 'Cre', data)


WT = create_empty_matrix(names, 'WT',data)
wild = WT[:]
wildy = WT[:]
mutant = Cre[:]
mutanty = Cre[:]


mut_wake_to_wake, mut_wake_to_NREM, mut_wake_to_REM, mut_NREM_to_wake, mut_NREM_to_NREM, mut_NREM_to_REM, mut_REM_to_wake, mut_REM_to_NREM, mut_REM_to_REM = prob_state_all(Cre)

max_bout = len(WT[0])-2
wt_REM_len, wt_NREM_len, wt_wake_len = find_all_len(WT)
wt_REM_bouts, wt_REM_len_avg = avg_bout_len(wt_REM_len)
wt_NREM_bouts, wt_NREM_len_avg = avg_bout_len(wt_NREM_len)
wt_wake_bouts, wt_wake_len_avg = avg_bout_len(wt_wake_len)
mut_REM_len, mut_NREM_len, mut_wake_len = find_all_len(Cre)
mut_REM_bouts, mut_REM_len_avg = avg_bout_len(mut_REM_len)
mut_NREM_bouts, mut_NREM_len_avg = avg_bout_len(mut_NREM_len)
mut_wake_bouts, mut_wake_len_avg = avg_bout_len(mut_wake_len)
mut_wake_len_avg_min = sec2min(mut_wake_len_avg)
wt_wake_len_avg_min = sec2min(wt_wake_len_avg)
mut_NREM_len_avg_min = sec2min(mut_NREM_len_avg)
wt_NREM_len_avg_min = sec2min(wt_NREM_len_avg)
wt_wake_totalTime, wt_NREM_totalTime, wt_REM_totalTime = total_sleep_time(wt_wake_len, wt_NREM_len, wt_REM_len)
mut_wake_totalTime, mut_NREM_totalTime, mut_REM_totalTime = total_sleep_time(mut_wake_len, mut_NREM_len, mut_REM_len)
plotDots(wt_wake_totalTime,WT_NAME, mut_wake_totalTime, MUTANT_NAME, 'Wake time (hr)', 'Total wake time', wake = True)
plotDots(wt_NREM_totalTime,WT_NAME, mut_NREM_totalTime, MUTANT_NAME, 'NREM time (hr)', 'Total NREM time')
plotDots(wt_REM_totalTime,WT_NAME, mut_REM_totalTime, MUTANT_NAME, 'REM time (min)', 'Total REM time')
wt_wake_perc, wt_NREM_perc, wt_REM_perc = percent_by_hour(wild, minutes = minutes_per_display)
mut_wake_perc, mut_NREM_perc, mut_REM_perc = percent_by_hour(mutant,minutes = minutes_per_display)
wt_wake_perc_avg, wt_wake_perc_std = avg_animals(wt_wake_perc)
wt_NREM_perc_avg, wt_NREM_perc_std = avg_animals(wt_NREM_perc)
wt_REM_perc_avg, wt_REM_perc_std = avg_animals(wt_REM_perc)
mut_wake_perc_avg, mut_wake_perc_std = avg_animals(mut_wake_perc)
mut_NREM_perc_avg, mut_NREM_perc_std = avg_animals(mut_NREM_perc)
mut_REM_perc_avg, mut_REM_perc_std = avg_animals(mut_REM_perc)
plotPerc(wt_wake_perc_avg, wt_wake_perc_std,mut_wake_perc_avg,mut_wake_perc_std, 'Wake percentage by hour',WT_NAME, MUTANT_NAME, 'Wake %')
plotPerc(wt_NREM_perc_avg, wt_NREM_perc_std,mut_NREM_perc_avg,mut_NREM_perc_std, 'NREM percentage by hour',WT_NAME, MUTANT_NAME, 'NREM %')
plotPerc(wt_REM_perc_avg, wt_REM_perc_std,mut_REM_perc_avg,mut_REM_perc_std, 'REM percentage by hour',WT_NAME, MUTANT_NAME, 'REM %')
wake_by_hr_ttest = ttest_by_hr(wt_wake_perc,mut_wake_perc)
NREM_by_hr_ttest = ttest_by_hr(wt_NREM_perc,mut_NREM_perc)
REM_by_hr_ttest = ttest_by_hr(wt_REM_perc,mut_REM_perc)
wtLight, wtDark = LDsplit(wildy)
mutLight, mutDark = LDsplit(mutanty)
max_bout = len(wtLight[0])-2
wt_dark_REM_len, wt_dark_NREM_len, wt_dark_wake_len = find_all_len(wtDark)
mut_dark_REM_len, mut_dark_NREM_len, mut_dark_wake_len = find_all_len(mutDark)
wt_dark_wake_totalTime, wt_dark_NREM_totalTime, wt_dark_REM_totalTime = total_sleep_time(wt_dark_wake_len, wt_dark_NREM_len, wt_dark_REM_len)
mut_dark_wake_totalTime, mut_dark_NREM_totalTime, mut_dark_REM_totalTime = total_sleep_time(mut_dark_wake_len, mut_dark_NREM_len, mut_dark_REM_len)
plotDots(wt_dark_wake_totalTime,WT_NAME, mut_dark_wake_totalTime, MUTANT_NAME, 'Time (hr)', 'Total dark wake time')
plotDots(wt_dark_NREM_totalTime,WT_NAME, mut_dark_NREM_totalTime, MUTANT_NAME, 'Time (hr)', 'Total dark NREM time')
plotDots(wt_dark_REM_totalTime,WT_NAME, mut_dark_REM_totalTime, MUTANT_NAME, 'Time (m)', 'Total dark REM time')
wt_light_REM_len, wt_light_NREM_len, wt_light_wake_len = find_all_len(wtLight)
mut_light_REM_len, mut_light_NREM_len, mut_light_wake_len = find_all_len(mutLight)
wt_light_wake_totalTime, wt_light_NREM_totalTime, wt_light_REM_totalTime = total_sleep_time(wt_light_wake_len, wt_light_NREM_len, wt_light_REM_len)
mut_light_wake_totalTime, mut_light_NREM_totalTime, mut_light_REM_totalTime = total_sleep_time(mut_light_wake_len, mut_light_NREM_len, mut_light_REM_len)
plotDots(wt_light_wake_totalTime,WT_NAME, mut_light_wake_totalTime, MUTANT_NAME, 'Time (hr)', 'Total light wake time')
plotDots(wt_light_NREM_totalTime,WT_NAME, mut_light_NREM_totalTime, MUTANT_NAME, 'Time (hr)', 'Total light NREM time')
plotDots(wt_light_REM_totalTime,WT_NAME, mut_light_REM_totalTime, MUTANT_NAME, 'Time (min)', 'Total light REM time')
plotHist(wt_REM_len, mut_REM_len, 'REM bout len (m)', 'Frequency (%)', 'REM hist', WT_NAME, MUTANT_NAME)
plotHist(wt_NREM_len, mut_NREM_len, 'NREM bout len (m)', 'Frequency (%)', 'NREM hist', WT_NAME, MUTANT_NAME)
plotHist(wt_wake_len, mut_wake_len, 'Wake bout len (m)', 'Frequency (%)', 'Wake hist', WT_NAME , MUTANT_NAME)
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
plot_barplot(wt_wake_bouts,WT_NAME, mut_wake_bouts, MUTANT_NAME, 'Bout number', 'Average Wake Bout #')
plot_barplot(wt_NREM_bouts,WT_NAME, mut_NREM_bouts, MUTANT_NAME, 'Bout number', 'Average NREM Bout #')
plot_barplot(wt_REM_bouts,WT_NAME, mut_REM_bouts, MUTANT_NAME, 'Bout number', 'Average REM Bout #')
plot_barplot(wt_wake_len_avg_min,WT_NAME, mut_wake_len_avg_min, MUTANT_NAME, 'Bout len (m)', 'Average wake bout length')
plot_barplot(wt_NREM_len_avg_min,WT_NAME, mut_NREM_len_avg_min, MUTANT_NAME, 'Bout len (m)', 'Average NREM bout length')
plot_barplot_REM(wt_REM_len_avg,WT_NAME, mut_REM_len_avg, MUTANT_NAME, 'Bout len (sec)', 'Average REM bout length')
plot_barplot_4groups(wt_wake_bouts,WT_NAME, mut_wake_bouts, MUTANT_NAME, wt_NREM_bouts, mut_NREM_bouts, 'Bout #', 'Average Wake & NREM Bout #')
plot_barplot_6groups(wt_wake_bouts,WT_NAME, mut_wake_bouts, MUTANT_NAME, wt_NREM_bouts, mut_NREM_bouts,wt_REM_bouts, mut_REM_bouts, 'Bout #', 'Average Wake & NREM &REM Bout #')
plot_barplot_4groups(wt_wake_len_avg_min,WT_NAME, mut_wake_len_avg_min, MUTANT_NAME, wt_NREM_len_avg_min,mut_NREM_len_avg_min,'Bout len (min)', 'Average wake bout length + NREM')

