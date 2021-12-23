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
from matplotlib.ticker import FuncFormatter
import matplotlib
import pylab as p
import scipy.stats as st
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from scipy.signal import periodogram
from matplotlib.lines import Line2D
from six import iteritems
from sklearn.metrics import mean_squared_error as mean_squared_error
from math import sqrt
import random
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
name_score = 'RTG DREADD_ZT1.csv' #user scored epochs
#max_bout = 8640 ## the maximum number of bouts in the file 
days = 2 ## number of days per mouse
bout_length = 10 ##number of seconds per bout
minutes_per_display = 60
globalFont = 15
global_name = '_ZT1_DREADD Gq whole'
half = False


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
    ##Then it averages over multiple days
    
    
    avg_len = []
    num_bouts = []
    for i in range(len(a)):
        num_bouts.append(len(a[i]))
        curr_len = a[i]
        curr_avg = np.mean(curr_len)
        avg_len.append(curr_avg)
    avg_leng_comb = []
    num_bouts_comb = []
    
    ##averages over the number of days (set in the master variables 
    ##at the beginning)
    
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
    for i in range(len(a)):
        a[i]= a[i]/60
    return a
def min2hour(a):
    for i in range(len(a)):
        a[i] = a[i]/60
    return a
def total_sleep_time(wake,NREM, REM, mulDays = True):
    ##sum the total sleep amount every day
    ##
    
    
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
    
def avg_animals(a):
    avg_matrix = []
    std_matrix = []

    for i in range(len(a[0])):
        curr_total = []
        for j in range(len(a)):
            curr_total.append(a[j][i])
        #print(curr_total)
        avg_matrix.append(np.mean(curr_total))
        std_matrix.append(np.std(curr_total))
    return avg_matrix, std_matrix   
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


def LDsplit(a):
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
    # test_results = add_stat_annotation(ax1, data=framee, x='condition', y='cFos', order=order,
    #                                box_pairs=[('Gq Sal', 'Gq CNO'),('Gq CNO', 'mCherry CNO'),('mCherry Sal','mCherry CNO')],
    #                                perform_stat_test=False, pvalues=pval,
    #                                text_format='star', verbose=2)

    # sns.barplot(x="condition", y="cFos_2", data=framee,capsize=.1, color = "lightblue" )


    sns.swarmplot(x = "condition", y = "cFos", data = framee, color = 'darkgray', alpha = 1, s = 5)

    # sns.swarmplot(x = "condition", y = "cFos", data = framee, palette = ["dodgerblue",'navy','seagreen','darkgreen'], alpha = 1, s = 5)
    sns.despine()
    # plt.setp(ax1.artists, edgecolor = 'k', facecolor='w')
    # plt.setp(ax1.lines, color='k')

    plt.xlabel('')
    #plt.ylim(10,14)
    plt.ylabel(ylabel, fontsize = globalFont)
    # plt.title(name, fontsize = globalFont)
    plt.savefig(name +global_name+".pdf")
    plt.show()
    
    return


def hrByhr_ttest(a,b):
    ttest_all = []
    for i in range(len(a[0])):
        a_curr = extract_column(a, i)
        b_curr = extract_column(b,i)
        ttest = st.ttest_ind(a_curr,b_curr, equal_var = False)
        ttest = ttest[1]
        ttest_all.append(ttest)
    return ttest_all
                   



with open(name_score, newline='\n' ) as inputfile:
   data = list(csv.reader(inputfile)) 
   data[0][0] = data[0][0][1:len(data[0][0])]
   

names = extract_row(data,1)
Cre = create_empty_matrix(names, 'CNO_Gq', data)
WT = create_empty_matrix(names, 'Sal_Gq',data) 
wild = WT[:]
wildy = WT[:]
mutant = Cre[:]
mutanty = Cre[:]

mcherryCNO = create_empty_matrix(names, 'CNO_mcherry', data)
mcherryCNOy = mcherryCNO[:]

mcherrySAL = create_empty_matrix(names, 'Sal_mcherry', data)
mcherrySALy = mcherrySAL[:]





max_bout = len(WT[0])-2
tot_hrs = int(len(WT[0])/60/60*bout_length)              
if half == True:
    
    Cre = createHalf(Cre)
    WT = createHalf(wildy)
    max_bout = len(WT[0])-2


wt_REM_len, wt_NREM_len, wt_wake_len = find_all_len(WT)
mcherrySAL_REM_len, mcherrySAL_NREM_len, mcherrySAL_wake_len = find_all_len(mcherrySAL)


#wt_REM_len, wt_NREM_len, wt_wake_len = find_all_len(WT_half)




#wt_REM_bouts, wt_REM_len_avg = avg_bout_len(wt_REM_len)
#wt_NREM_bouts, wt_NREM_len_avg = avg_bout_len(wt_NREM_len)
#wt_wake_bouts, wt_wake_len_avg = avg_bout_len(wt_wake_len)


mut_REM_len, mut_NREM_len, mut_wake_len = find_all_len(Cre)
mcherryCNO_REM_len, mcherryCNO_NREM_len, mcherryCNO_wake_len = find_all_len(mcherryCNO)

#mut_REM_bouts, mut_REM_len_avg = avg_bout_len(mut_REM_len)
#mut_NREM_bouts, mut_NREM_len_avg = avg_bout_len(mut_NREM_len)
#mut_wake_bouts, mut_wake_len_avg = avg_bout_len(mut_wake_len)

    



#mut_wake_len_avg_min = sec2min(mut_wake_len_avg)
#wt_wake_len_avg_min = sec2min(wt_wake_len_avg)
#plotDots(wt_wake_len_avg_min,'WT', mut_wake_len_avg_min, 'Cre', 'time (m)', 'Average wake bout length')
#mut_NREM_len_avg_min = sec2min(mut_NREM_len_avg)
#wt_NREM_len_avg_min = sec2min(wt_NREM_len_avg)
#plotDots(wt_NREM_len_avg_min,'WT', mut_NREM_len_avg_min, 'Cre', 'time (m)', 'Average NREM bout length')
#plotDots(wt_REM_len_avg,'WT', mut_REM_len_avg, 'Cre', 'time (s)', 'Average REM bout length')
#plotDots(wt_wake_bouts,'WT', mut_wake_bouts, 'Cre', '# of bouts', 'Number wake bouts')
#plotDots(wt_NREM_bouts,'WT', mut_NREM_bouts, 'Cre', '# of bouts', 'Number NREM bouts')
#plotDots(wt_REM_bouts,'WT', mut_REM_bouts, 'Cre', '# of bouts', 'Number REM bouts')

wt_wake_totalTime, wt_NREM_totalTime, wt_REM_totalTime = total_sleep_time(wt_wake_len, wt_NREM_len, wt_REM_len, mulDays = False)
mcherrySAL_wake_totalTime, mcherrySAL_NREM_totalTime, mcherrySAL_REM_totalTime = total_sleep_time(mcherrySAL_wake_len, mcherrySAL_NREM_len, mcherrySAL_REM_len, mulDays = False)




mut_wake_totalTime, mut_NREM_totalTime, mut_REM_totalTime = total_sleep_time(mut_wake_len, mut_NREM_len, mut_REM_len, mulDays = False)

mcherryCNO_wake_totalTime, mcherryCNO_NREM_totalTime, mcherryCNO_REM_totalTime = total_sleep_time(mcherryCNO_wake_len, mcherryCNO_NREM_len, mcherryCNO_REM_len, mulDays = False)






plotDots(wt_wake_totalTime,'Sal', mut_wake_totalTime, 'CNO', 'time (hr)', 'Total wake time')
plotDots(wt_NREM_totalTime,'Sal', mut_NREM_totalTime, 'CNO', 'time (hr)', 'Total NREM time')
plotDots(wt_REM_totalTime,'Sal', mut_REM_totalTime, 'CNO', 'time (m)', 'Total REM time')

plotDots(mcherrySAL_wake_totalTime,'Sal', mcherryCNO_wake_totalTime, 'CNO', 'time (hr)', 'Total wake time mcherry')
plotDots(mcherrySAL_NREM_totalTime,'Sal', mcherryCNO_NREM_totalTime, 'CNO', 'time (hr)', 'Total NREM time mcherry')
plotDots(mcherrySAL_REM_totalTime,'Sal', mcherryCNO_REM_totalTime, 'CNO', 'time (m)', 'Total REM time mcherry')






wt_wake_perc, wt_NREM_perc, wt_REM_perc = percent_by_hour(wild, minutes = minutes_per_display, mul_days = False)
mcherrySAL_wake_perc, mcherrySAL_NREM_perc, mcherrySAL_REM_perc = percent_by_hour(mcherrySALy, minutes = minutes_per_display, mul_days = False)





mut_wake_perc, mut_NREM_perc, mut_REM_perc = percent_by_hour(mutant,minutes = minutes_per_display, mul_days = False)
mcherryCNO_wake_perc, mcherryCNO_NREM_perc, mcherryCNO_REM_perc = percent_by_hour(mcherryCNOy,minutes = minutes_per_display, mul_days = False)

wt_wake_perc_avg, wt_wake_perc_std = avg_animals(wt_wake_perc)
wt_NREM_perc_avg, wt_NREM_perc_std = avg_animals(wt_NREM_perc)
wt_REM_perc_avg, wt_REM_perc_std = avg_animals(wt_REM_perc)

mcherrySAL_wake_perc_avg, mcherrySAL_wake_perc_std = avg_animals(mcherrySAL_wake_perc)
mcherrySAL_NREM_perc_avg, mcherrySAL_NREM_perc_std = avg_animals(mcherrySAL_NREM_perc)
mcherrySAL_REM_perc_avg, mcherrySAL_REM_perc_std = avg_animals(mcherrySAL_REM_perc)





mut_wake_perc_avg, mut_wake_perc_std = avg_animals(mut_wake_perc)
mut_NREM_perc_avg, mut_NREM_perc_std = avg_animals(mut_NREM_perc)
mut_REM_perc_avg, mut_REM_perc_std = avg_animals(mut_REM_perc)

mcherryCNO_wake_perc_avg, mcherryCNO_wake_perc_std = avg_animals(mcherryCNO_wake_perc)
mcherryCNO_NREM_perc_avg, mcherryCNO_NREM_perc_std = avg_animals(mcherryCNO_NREM_perc)
mcherryCNO_REM_perc_avg, mcherryCNO_REM_perc_std = avg_animals(mcherryCNO_REM_perc)



plot_perc(wt_wake_perc_avg, wt_wake_perc_std,mut_wake_perc_avg,mut_wake_perc_std, tot_hrs, 'Wake percentage')
plot_perc(wt_NREM_perc_avg, wt_NREM_perc_std,mut_NREM_perc_avg,mut_NREM_perc_std,tot_hrs, 'NREM percentage')
plot_perc(wt_REM_perc_avg, wt_REM_perc_std,mut_REM_perc_avg,mut_REM_perc_std,tot_hrs, 'REM percentage')

plot_perc(mcherrySAL_wake_perc_avg, mcherrySAL_wake_perc_std,mcherryCNO_wake_perc_avg,mcherryCNO_wake_perc_std, tot_hrs, 'Wake percentage mcherry')
plot_perc(mcherrySAL_NREM_perc_avg, mcherrySAL_NREM_perc_std,mcherryCNO_NREM_perc_avg,mut_NREM_perc_std,tot_hrs, 'NREM percentage mcherry')
plot_perc(mcherrySAL_REM_perc_avg, mcherrySAL_REM_perc_std,mcherryCNO_REM_perc_avg,mcherryCNO_REM_perc_std,tot_hrs, 'REM percentage mcherry')

plot_perc_4groups(wt_wake_perc_avg, wt_wake_perc_std,mut_wake_perc_avg,mut_wake_perc_std,mcherrySAL_wake_perc_avg, mcherrySAL_wake_perc_std,mcherryCNO_wake_perc_avg,mcherryCNO_wake_perc_std, tot_hrs, 'Wake percentage 4 groups', 'Wake %')
plot_perc_4groups(wt_NREM_perc_avg, wt_NREM_perc_std,mut_NREM_perc_avg,mut_NREM_perc_std,mcherrySAL_NREM_perc_avg, mcherrySAL_NREM_perc_std,mcherryCNO_NREM_perc_avg,mcherryCNO_NREM_perc_std, tot_hrs, 'NREM percentage 4 groups','NREM %')
plot_perc_4groups(wt_REM_perc_avg, wt_REM_perc_std,mut_REM_perc_avg,mut_REM_perc_std,mcherrySAL_REM_perc_avg, mcherrySAL_REM_perc_std,mcherryCNO_REM_perc_avg,mcherryCNO_REM_perc_std, tot_hrs, 'REM percentage 4 groups', 'REM %')




plotWithLines(wt_NREM_totalTime, mut_NREM_totalTime, 'NREM time total lines')


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

# wake_pval = [0.0134]
plot_boxplot_4groups(wake_pval,wt_wake_totalTime, 'gqs', mut_wake_totalTime, 'gqc', mcherrySAL_wake_totalTime, mcherryCNO_wake_totalTime, 'Time (hr)', 'Gq + mcherry wake | sal + CNO')



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


plot_boxplot_4groups(NREM_pval,wt_NREM_totalTime, 'gqs', mut_NREM_totalTime, 'gqc', mcherrySAL_NREM_totalTime, mcherryCNO_NREM_totalTime, 'Time (hr)', 'Gq + mcherry NREM | sal + CNO')



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






plot_boxplot_4groups(REM_pval,wt_REM_totalTime, 'gqs', mut_REM_totalTime, 'gqc', mcherrySAL_REM_totalTime, mcherryCNO_REM_totalTime, 'Time (min)', 'Gq + mcherry REM | sal + CNO')



GqCNO_GqSAL_wake_ttest = hrByhr_ttest(wt_wake_perc, mut_wake_perc)
GqCNO_GqSAL_NREM_ttest = hrByhr_ttest(wt_NREM_perc, mut_NREM_perc)
GqCNO_GqSAL_REM_ttest = hrByhr_ttest(wt_REM_perc, mut_REM_perc)


#wtLight, wtDark = LDsplit(wildy)
#mutLight, mutDark = LDsplit(mutanty)
#max_bout = len(wtLight[0])-2
              
              
#wt_dark_REM_len, wt_dark_NREM_len, wt_dark_wake_len = find_all_len(wtDark)
#mut_dark_REM_len, mut_dark_NREM_len, mut_dark_wake_len = find_all_len(mutDark)

#wt_dark_wake_totalTime, wt_dark_NREM_totalTime, wt_dark_REM_totalTime = total_sleep_time(wt_dark_wake_len, wt_dark_NREM_len, wt_dark_REM_len)
#mut_dark_wake_totalTime, mut_dark_NREM_totalTime, mut_dark_REM_totalTime = total_sleep_time(mut_dark_wake_len, mut_dark_NREM_len, mut_dark_REM_len)
#plotDots(wt_dark_wake_totalTime,'WT', mut_dark_wake_totalTime, 'Cre', 'time (hr)', 'Total dark wake time')
#plotDots(wt_dark_NREM_totalTime,'WT', mut_dark_NREM_totalTime, 'Cre', 'time (hr)', 'Total dark NREM time')
#plotDots(wt_dark_REM_totalTime,'WT', mut_dark_REM_totalTime, 'Cre', 'time (m)', 'Total dark REM time')







#wt_light_REM_len, wt_light_NREM_len, wt_light_wake_len = find_all_len(wtLight)
#mut_light_REM_len, mut_light_NREM_len, mut_light_wake_len = find_all_len(mutLight)

#wt_light_wake_totalTime, wt_light_NREM_totalTime, wt_light_REM_totalTime = total_sleep_time(wt_light_wake_len, wt_light_NREM_len, wt_light_REM_len)
#mut_light_wake_totalTime, mut_light_NREM_totalTime, mut_light_REM_totalTime = total_sleep_time(mut_light_wake_len, mut_light_NREM_len, mut_light_REM_len)
#plotDots(wt_light_wake_totalTime,'WT', mut_light_wake_totalTime, 'Cre', 'time (hr)', 'Total light wake time')
#plotDots(wt_light_NREM_totalTime,'WT', mut_light_NREM_totalTime, 'Cre', 'time (hr)', 'Total light NREM time')
#plotDots(wt_light_REM_totalTime,'WT', mut_light_REM_totalTime, 'Cre', 'time (m)', 'Total light REM time')



