
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
import matplotlib
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')


##############################################
##############################################
##############################################
##############################################
name_score = 'RTG_300trials_6mice.csv' #user scored epochs
#max_bout = 8640 ## the maximum number of bouts in the file 
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
from statannot import add_stat_annotation




bout_per_min = int(60/bout_length)
stim_freq = 20 ## how ofter stim repeats in min
stim_length = 90 ##stim length in sec
trials_per_animal = 50



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
def total_sleep_time(wake,NREM, REM):
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
def percent_by_hour(a, minutes = 60):
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
    
    for i in range(len(data)):
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
    g2 = sns.boxplot(x="condition", y="cFos", data=framee, palette = ["lightgray",'dodgerblue'], linewidth =1)
    sns.swarmplot(x = "condition", y = "cFos", data = framee,palette = ["dimgray",'dimgray'], alpha = 1, s = 8)
    sns.despine()
    order = [wtName, mutName]

    test_results = add_stat_annotation(g2, data=framee, x='condition', y='cFos', order=order,
                                   box_pairs=[(wtName, mutName)],
                                   test='t-test_ind', text_format='star',
                                   loc='outside', verbose=2)
    plt.xlabel('')
    #plt.ylim(10,14)
    plt.ylabel(ylabel, fontsize = globalFont)
    # plt.title(name, fontsize = globalFont)
    plt.savefig(name +global_name+".pdf")
    plt.show()
    
    return

def plotDots_ttest(wt,wtName, mut,mutName, ylabel,name):
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

    g2 = sns.boxplot(x="condition", y="cFos", data=framee, palette = ["khaki",'mediumorchid'], linewidth =1)
    sns.swarmplot(x = "condition", y = "cFos", data = framee,palette = ["goldenrod",'blueviolet'], alpha = 1, s = 8)
  
    
    
    sns.despine()
    order = [wtName, mutName]

    # test_results = add_stat_annotation(g2, data=framee, x='condition', y='cFos', order=order,
    #                                box_pairs=[(wtName, mutName)],
    #                                test='t-test_ind', text_format='star',
    #                                loc='outside', verbose=2)
    plt.xlabel('')
    #plt.ylim(10,14)
    plt.ylabel(ylabel, fontsize = globalFont)
    
    
    
    
    total_wake_ttest = st.ttest_ind(wt,mut, equal_var = False)
    total_wake_ttest = total_wake_ttest[1]
    # plt.title(str(total_wake_ttest), fontsize = globalFont)
    plt.savefig(name +global_name+".pdf")
    plt.show()
    
    return
    
def create_hrs(std):
    interval = 24/len(std)
    matrix = []
    for i in range(len(std)):
        matrix.append(i*interval)
    return matrix

def plot_perc(wtAVG, wtSTD, mutAVG, mutSTD,name):
    x_axis = create_hrs(wtSTD)
    ax1 = plt.axes(frameon=False)
    plt.errorbar(x_axis, wtAVG, yerr = wtSTD, fmt = '-o')
    plt.errorbar(x_axis, mutAVG, yerr = mutSTD, fmt = '-o')
    plt.xlabel('time (hr)',fontsize = globalFont*.8)
    plt.ylabel('%',fontsize = globalFont*.8 )


    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))  
    plt.ylim(0,140)

    title = name
    plt.title(title)
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
def bout2sec(a):
    for i in range(len(a)):
        a[i] = a[i]*bout_length/60
    return a

def combineMatrix(a):
    matrix = []
    for i in range(len(a)):
        for j in range(len(a[i])):
            matrix.append(a[i][j])
    matrix = bout2sec(matrix)
    return matrix

def plotHist(wt, mut, xname, yname, title, wtname, mutname, BIN = 30):
    wt = combineMatrix(wt)
    mut = combineMatrix(mut)
    ax1 = plt.axes(frameon=False)
    #LABEL = []
    #LABEL.append(wtname)
    #LABEL.append(mutname)
    plt.hist(wt, bins = BIN, normed = True, histtype = 'step', lw = 6, label = wtname, color = 'black')
    plt.hist(mut, bins = BIN,normed = True, histtype = 'step', lw = 6, label = mutname, color = 'deeppink')
    
    
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


with open(name_score, newline='\n' ) as inputfile:
   data = list(csv.reader(inputfile)) 
   data[0][0] = data[0][0][1:len(data[0][0])]
def stimStartMatrix():
    matrix = []
    start = min_before1st*bout_per_min
    for i in range(num_trials):
        index = int(start + i*stim_freq*60/bout_length)
        matrix.append(index)
    return matrix
        


def extract_trials(a):
    matrix = [0 for i in range(num_trials +1)]
    bout_per_trial = int(beforeAndAfter*2*bout_per_min +2)
    stim_start = stimStartMatrix()
    matName = a[0:2]
    a = a[2:len(a)]
    matrix[0] = matName
    for i in range(num_trials):
        sub_matrix = []
        for j in range(bout_per_trial):
            pointer = beforeAndAfter * bout_per_min 
            index = stim_start[i] - pointer +j
            sub_matrix.append(a[index])
        matrix[i +1] = sub_matrix
            
    return matrix
def extract_all_trials(a):
    matrix = [0 for i in range((len(a)))]
    for i in range(len(a)):
        matrix[i] = extract_trials(a[i])
def scrub_formating(a):
    stop = len(a[0])-1
    for i in range(len(a)):
        for j in range(len(a[0])):
            
            a[i][j] = a[i][j][2:len(a[i][j])-1]
    for i in range(len(new_data)):
        new_data[i][stop] = new_data[i][stop][0:len(new_data[i][stop])-1]
    a[0][0] = a[0][0][1:len(a[0][0])] 
    a[0][stop] = a[0][stop][0:len(a[0][stop])-1]     
    return a
def just_scores(a):
    for i in range(len(a)):
            a[i] = a[i][2:len(a[i])]
    return a
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
    
    if len(a) ==1:
        for i in range(len(NREM_avg)):
            new_matrix.append(0)
        return new_matrix, new_matrix, new_matrix
    for i in range(len(a[0])):
        column = extract_column(a, i)
        curr_wake = calc_perc(column, 'Wake')
        curr_NREM = calc_perc(column, 'NREM')
        curr_REM = calc_perc(column, 'REM')
        wake_perc.append(curr_wake)
        NREM_perc.append(curr_NREM)
        REM_perc.append(curr_REM)
    return wake_perc, NREM_perc, REM_perc



def create_sec():
    tot_bout = int(2*bout_per_min*beforeAndAfter + stim_length/bout_length)

    sec = []
    
    for i in range(tot_bout):
        sec.append(bout_length*i)
    for i in range(len(sec)):
        sec[i] = sec[i] - bout_per_min*beforeAndAfter*bout_length  + bout_length*2
    # for i in range(len(sec)):
        
    return sec



def plotPerc(wake, NREM, REM, sec, name):
    ax1 = plt.axes(frameon=False)
    plt.plot(sec, wake, color ='blue', linewidth = 10, label = 'Wake')
    plt.plot(sec, NREM, color = 'purple', linewidth = 10, label = 'NREM')
    plt.plot(sec, REM, color = 'gray', linewidth = 10, label = 'REM')
    
    plt.xlabel('time (m)',fontsize = globalFont)
    plt.ylabel('%',fontsize = globalFont)    
    plt.ylim(0,100)
    
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.tick_params(labelsize=globalFont*.8)  
    plt.xticks(ticks = [-180,-120,-60, 0, 60, 120, 180, 240],labels = ['-3','-2','-1','0','1','2','3', '4'])
    
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))  
    #ax1.legend(loc = 'best', frameon = False, prop = {'size':20})
    if inhibitory == False:
        ax1.axvspan(0,stim_length, color = 'lightskyblue')
    
    if inhibitory == True:
         ax1.axvspan(0,stim_length, color = 'yellow')   
    #plt.title(name)
    plt.ylim(0,102)
    #plt.title(name)
    plt.savefig(name +'opto_stim.pdf')
    
    plt.show()
    return
def calc_fill_between(a, error):
    plusError = []
    minusError = []
    for i in range((len(a))):
        plusError.append(a[i]+error[i])
        minusError.append(a[i]-error[i])
    return plusError, minusError
def plotPercSEM(wake, wakeSEM, NREM,NREMSEM,  REM,REMSEM, sec, name):

    ax1 = plt.axes(frameon=False)
   # plt.plot(sec, wake, color ='blue', linewidth = 3, label = 'Wake')
    plt.errorbar(sec, wake,wakeSEM, c = 'blue', lw = 4, ecolor= 'dodgerblue')
    #wakePlus, wakeMinus = calc_fill_between(wake, wakeSEM)
    #plt.fill_between(sec, wakePlus, wakeMinus, facecolor = 'dodgerblue')
    #plt.plot(sec, NREM, color = 'purple', linewidth = 3, label = 'NREM')
    plt.errorbar(sec, NREM,NREMSEM, c = 'purple', lw = 4, ecolor= 'thistle')

    #plt.plot(sec, REM, color = 'gray', linewidth = 3, label = 'REM')
    plt.errorbar(sec, REM,REMSEM, c = 'gray', lw = 4, ecolor= 'lightgrey')
    
    plt.xlabel('time (m)',fontsize = globalFont)
    plt.ylabel('%',fontsize = globalFont)    
    plt.ylim(0,100)
    
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.tick_params(labelsize=globalFont*.8)  
    plt.xticks(ticks = [-180,-120,-60, 0, 60, 120, 180, 240],labels = ['-3','-2','-1','0','1','2','3', '4'])
    
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))  
    #ax1.legend(loc = 'best', frameon = False, prop = {'size':20})
    if inhibitory == False:
        ax1.axvspan(0,stim_length, color = 'lightskyblue')
    
    if inhibitory == True:
         ax1.axvspan(0,stim_length, color = 'yellow')       #plt.title(name)
    plt.ylim(0,102)
    #plt.title(name)
    plt.savefig(name +'opto_stim.pdf')
    
    plt.show()
    return

def extract_single_stage_before(a, name):
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
    # print(len(data))
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
    # print(num_trials)
    b = []

    for i in range(num_trials):
        index = int(np.random.rand()*num_trials)
        b.append(a[index])
    return b
    
    
def combine_state_prob_from_all_animals(wake, NREM, REM):
    all_wake = []
    all_NREM = []
    all_REM = []
    # print(len(wake))
    # print(len(wake[0]))
    for i in range(len(wake[0])):
        curr_wake = extract_column(wake, i)
        curr_NREM = extract_column(NREM, i)
        curr_REM = extract_column(REM, i)
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
    # print(len(curr_bootstrap))
    wake, NREM, REM = perc_per_animal_variable(curr_bootstrap, trialOcc)
    wake_tot, NREM_tot, REM_tot =  calc_percent_tot(curr_bootstrap)
    # print(len(wake[0]))
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
                
    
# def state_probabilities_all_bootstrap(a, trialOcc, iterations = 200):
#     wake_all = []
#     NREM_all = []
#     REM_all = []
#     for i in range(iterations):
#         curr_wake, curr_NREM, curr_REM = state_probabilities_single_bootstrap(a, trialOcc)
        
#         # curr_wake = [np.mean(curr_wake)]
#         wake_all = combineIterations(wake_all, curr_wake)
#         NREM_all = combineIterations(NREM_all, curr_NREM)
#         REM_all = combineIterations(REM_all, curr_REM)
#     return wake_all, NREM_all, REM_all

def state_probabilities_all_bootstrap_better(a,trialOcc, iterations = ITERATIONS):
    wake_all = []
    NREM_all = []
    REM_all = []
    for i in range(iterations):
        curr_wake, curr_NREM, curr_REM = state_probabilities_single_bootstrap_better(a, trialOcc)
        wake_all.append(curr_wake)
        NREM_all.append(curr_NREM)
        REM_all.append(curr_REM)
        
        # NREM_all = combineIterations(NREM_all, curr_NREM)
        # REM_all = combineIterations(REM_all, curr_REM)
    return wake_all, NREM_all, REM_all    


def dataRealign(a):
    c = []
    for i in range(len(a[0])):
        b = extract_column(a,i)
        c.append(b)
    return c

# def state_probabilities_all_bootstrap_test(a, trialOcc, iterations = 200):
#     wake_all = []
#     NREM_all = []
#     REM_all = []
#     for i in range(iterations):
#         curr_wake, curr_NREM, curr_REM = state_probabilities_single_bootstrap(a, trialOcc)
#         curr_wake_avg = []
#         curr_NREM_avg = []
#         curr_REM_avg = []
        
#         for j in range(len(curr_wake)):
#             curr_wake_avg.append(np.mean(curr_wake[j]))
#             curr_NREM_avg.append(np.mean(curr_NREM[j]))
#             curr_REM_avg.append(np.mean(curr_REM[j]))

#         wake_all.append(curr_wake_avg)
        
#         NREM_all.append(curr_NREM_avg)
#         REM_all.append(curr_REM_avg)       
#         #     # # curr_wake = [np.mean(curr_wake)]
#         # wake_all = combineIterations_test(wake_all, curr_wake)
#         # NREM_all = combineIterations_test(NREM_all, curr_NREM)
#         # REM_all = combineIterations_test(REM_all, curr_REM)
#             # wake_all.append(curr_wake_avg)
#     wake_all = dataRealign(wake_all)
#     NREM_all = dataRealign(NREM_all)
#     REM_all = dataRealign(REM_all)

#     return wake_all, NREM_all, REM_all
def extractConfidenceIntervals_single(a, index, lower = 0.025, upper = 0.975):

    # bounds = st.t.interval(alpha =.95, df = len(a)-1, loc = np.mean(a), scale = st.sem(a))
    # lower_val = bounds[0]
    # upper_val = bounds[1]
    curr_a = extract_column(a, index)
    a_sorted = curr_a[:]
    a_sorted.sort()
    # print(np.min(a_sorted))
    lower_val = a_sorted[int(len(a_sorted)*lower)]
    upper_val = a_sorted[int(len(a_sorted)*upper)]
    
    return lower_val, upper_val

# def getAvgPerBout_single(a, start, numMice):
#     new_a= []
#     for i in range(numMice):
#         index = start +i
#         curr = extract_column(a, index)


def extractConfidenceIntervals_all(a, lower = 0.025, upper = 0.975):
    lower_all =[]
    upper_all = []
    # num_mice = len(trialOcc)
    
    for i in range(len(a[0])):
        
        low_curr, up_curr = extractConfidenceIntervals_single(a,i)
        lower_all.append(float(low_curr))
        upper_all.append(float(up_curr))
    lower_all = np.array(lower_all)
    # lower_all = 1*lower_all
    upper_all = np.array(upper_all)
    # upper_all = 1*upper_all

    CI = [lower_all, upper_all]

    return CI
        
def plotPercCI(wake, wakeCI, NREM ,NREMCI,  REM,REMCI, sec, name):

    ax1 = plt.axes(frameon=False)
    plt.errorbar(sec, wake,yerr= wakeCI, c = 'blue', lw = 2, ecolor= 'dodgerblue',elinewidth =4)
    plt.errorbar(sec, NREM,NREMCI, c = 'purple', lw = 2, ecolor= 'thistle',elinewidth = 4)
    plt.errorbar(sec, REM,REMCI, c = 'gray', lw = 2, ecolor= 'lightgrey', elinewidth = 4)
    
    plt.xlabel('Time (min)',fontsize = globalFont)
    plt.ylabel('%',fontsize = globalFont)    
    plt.ylim(0,100)
    
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.tick_params(labelsize=globalFont*.8)  
    plt.xticks(ticks = [-180,-120,-60, 0, 60, 120, 180, 240],labels = ['-3','-2','-1','0','1','2','3', '4'])
    
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))  
    #ax1.legend(loc = 'best', frameon = False, prop = {'size':20})
    if inhibitory == False:
        ax1.axvspan(0,stim_length, color = 'lightskyblue')
    
    if inhibitory == True:
         ax1.axvspan(0,stim_length, color = 'yellow')       #plt.title(name)
    plt.ylim(0,102)
    #plt.title(name)
    plt.savefig(name +'opto_stim.pdf')
    
    plt.show()
    return

def generatePlotableCI(a, err):
    for i in range(len(a)):
        err[0][i] = a[i] - err[0][i]
        err[1][i] =err[1][i] - a[i]
    return err

def wakeMaintenance_single(wake,NREM, REM):
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


def wakeMaintenance_all(wake, NREM, REM):
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
    print(wakeb4)
    plotDots_ttest(wakeb4, '3 min before stim', wake_after, '3 min after stim', '% Wake', 'first min compared to last min_wake')
    plotDots_ttest(NREMb4, 'first min', NREM_after, 'last min', '% NREM', 'first min compared to last min_NREM')
    plotDots_ttest(REMb4, 'first min', REM_after, 'last min', '% REM', 'first min compared to last min_REM')
    return wakeb4, wake_after

    
    

for i in range(len(data)-1):
    data[i+1][0] = data[i+1][0].split(",")
new_data = [0 for i in range(len(data))]
new_data[0] = data[0]
for i in range(len(data)-1):
    new_data[i+1] = data[i+1][0]
new_data = scrub_formating(new_data)
new_data[0][0]


names = extract_column(new_data, 0)
names[0] = names[0][0:len(names[0])]
trialOccurances = determineTrialsPerAnimal(names)
justData = just_scores(new_data)



wake_total, NREM_total, REM_total = perc_per_animal_variable(justData, trialOccurances)


wake_avg, wake_sem = avg_animals(wake_total)
NREM_avg, NREM_sem = avg_animals(NREM_total)
REM_avg, REM_sem = avg_animals(REM_total)

NREMbefore = extract_single_stage_before(justData, 'NREM')
REMbefore = extract_single_stage_before(justData, 'REM')
wakebefore = extract_single_stage_before(justData, 'Wake')



wakeb4, wake_after = wakeMaintenance_all(wake_total, NREM_total, REM_total)


wakePer, NREMper, REMper = calc_percent_tot(justData)
sec = create_sec()
NREMbefore_wakePer, NREMbefore_NREMper, NREMbefore_REMper = calc_percent_tot(NREMbefore)
REMbefore_wakePer, REMbefore_NREMper, REMbefore_REMper = calc_percent_tot(REMbefore)
wakebefore_wakePer, wakebefore_NREMper, wakebefore_REMper = calc_percent_tot(wakebefore)



trials4bootstrap = trials_by_animal(justData,trialOccurances)
trials4bootstrap_realigned = undueBootstrapFormatting(trials4bootstrap)



# first_bootstrap = trials4bootstrap[0]
# sample_all_bootstrap = all_mouse_single_bootstrap(trials4bootstrap)
# BSw, BSn, BSr = state_probabilities_single_bootstrap(justData, trialOccurances)
# bootstrap_w, bootstrap_n, bootstrap_r = state_probabilities_all_bootstrap(justData, trialOccurances)

# bootstrap_w, bootstrap_n, bootstrap_r = state_probabilities_all_bootstrap_test(justData, trialOccurances)






# plotPerc(wakePer, NREMper, REMper, sec, global_name)
# plotPerc(NREMbefore_wakePer, NREMbefore_NREMper, NREMbefore_REMper, sec, NREM_before)
# plotPerc(REMbefore_wakePer, REMbefore_NREMper, REMbefore_REMper, sec, REM_before)
# plotPerc(wakebefore_wakePer, wakebefore_NREMper, wakebefore_REMper, sec, wake_before)




bootstrap_w, bootstrap_n, bootstrap_r = state_probabilities_all_bootstrap_better(trials4bootstrap_realigned, trialOccurances)

wake_CI = extractConfidenceIntervals_all(bootstrap_w)
NREM_CI = extractConfidenceIntervals_all(bootstrap_n)
REM_CI = extractConfidenceIntervals_all(bootstrap_r)


# NREMbefore_wake_tot, NREMbefore_NREM_tot, NREMbefore_REM_tot, NREMbefore_totaltrials = perc_per_animal_single_stage_variable(justData, trialOccurances,'NREM')
# NREMbefore_wake_avg, NREMbefore_wake_sem = avg_animals(NREMbefore_wake_tot)
# NREMbefore_NREM_avg, NREMbefore_NREM_sem = avg_animals(NREMbefore_NREM_tot)
# NREMbefore_REM_avg, NREMbefore_REM_sem = avg_animals(NREMbefore_REM_tot)

# REMbefore_wake_tot, REMbefore_NREM_tot, REMbefore_REM_tot,REMbefore_totaltrials = perc_per_animal_single_stage_variable(justData, trialOccurances,'REM')
# REMbefore_wake_avg, REMbefore_wake_sem = avg_animals(REMbefore_wake_tot)
# REMbefore_NREM_avg, REMbefore_NREM_sem = avg_animals(REMbefore_NREM_tot)
# REMbefore_REM_avg, REMbefore_REM_sem = avg_animals(REMbefore_REM_tot)

# wakebefore_wake_tot, wakebefore_NREM_tot, wakebefore_REM_tot, WAKEbefore_totaltrials = perc_per_animal_single_stage_variable(justData,trialOccurances, 'Wake')
# wakebefore_wake_avg, wakebefore_wake_sem = avg_animals(wakebefore_wake_tot)
# wakebefore_NREM_avg, wakebefore_NREM_sem = avg_animals(wakebefore_NREM_tot)
# wakebefore_REM_avg, wakebefore_REM_sem = avg_animals(wakebefore_REM_tot)







# plotPercSEM(wake_avg, wake_sem, NREM_avg, NREM_sem, REM_avg, REM_sem, sec, global_name)
# plotPercSEM(NREMbefore_wake_avg, NREMbefore_wake_sem, NREMbefore_NREM_avg, NREMbefore_NREM_sem, NREMbefore_REM_avg, NREMbefore_REM_sem, sec, NREM_before)
# plotPercSEM(REMbefore_wake_avg, REMbefore_wake_sem, REMbefore_NREM_avg, REMbefore_NREM_sem, REMbefore_REM_avg, REMbefore_REM_sem, sec, REM_before)
# plotPercSEM(wakebefore_wake_avg, wakebefore_wake_sem, wakebefore_NREM_avg, wakebefore_NREM_sem, wakebefore_REM_avg, wakebefore_REM_sem, sec, wake_before)



wake_CI = generatePlotableCI(wake_avg, wake_CI)
NREM_CI = generatePlotableCI(NREM_avg, NREM_CI)
REM_CI = generatePlotableCI(REM_avg, REM_CI)


plotPercCI(wake_avg, wake_CI, NREM_avg, NREM_CI, REM_avg, REM_CI, sec, global_name +'CI')



def mean_probabilites(wake, NREM, REM):
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

def mean_prob_bootstrap(wake, NREM, REM):
    wake_all = []
    NREM_all = []
    REM_all = []
    for i in range(len(wake)):
        wake_diff, NREM_diff, REM_diff = mean_probabilites(wake[i], NREM[i], REM[i])
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

def mean_probabilites_wakeMaintenance_firstAndLastMin(wake, NREM, REM):
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

def mean_prob_bootstrap_main1(wake, NREM, REM):
    wake_all = []
    NREM_all = []
    REM_all = []
    for i in range(len(wake)):
        wake_diff, NREM_diff, REM_diff = mean_probabilites_wakeMaintenance_firstAndLastMin(wake[i], NREM[i], REM[i])
        wake_all.append(wake_diff)
        NREM_all.append(NREM_diff)
        REM_all.append(REM_diff)
    return wake_all, NREM_all, REM_all

def mean_probabilites_wakeMaintenance_firstAndLastMin_2(wake, NREM, REM):
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

def mean_prob_bootstrap_main2(wake, NREM, REM):
    wake_all = []
    NREM_all = []
    REM_all = []
    for i in range(len(wake)):
        wake_diff, NREM_diff, REM_diff = mean_probabilites_wakeMaintenance_firstAndLastMin_2(wake[i], NREM[i], REM[i])
        wake_all.append(wake_diff)
        NREM_all.append(NREM_diff)
        REM_all.append(REM_diff)
    return wake_all, NREM_all, REM_all




def mean_probabilites_wakeMaintenance_firstAndLastMin_3(wake, NREM, REM):
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

def mean_prob_bootstrap_main3(wake, NREM, REM):
    wake_all = []
    NREM_all = []
    REM_all = []
    for i in range(len(wake)):
        wake_diff, NREM_diff, REM_diff = mean_probabilites_wakeMaintenance_firstAndLastMin_3(wake[i], NREM[i], REM[i])
        wake_all.append(wake_diff)
        NREM_all.append(NREM_diff)
        REM_all.append(REM_diff)
    return wake_all, NREM_all, REM_all








wake_diff, NREM_diff, REM_diff = mean_probabilites(wake_avg, NREM_avg, REM_avg)
wake_diff_main_1, NREM_diff_main_2, REM_diff_main_3 = mean_probabilites_wakeMaintenance_firstAndLastMin(wake_avg, NREM_avg, REM_avg)




wake_diff_bs, NREM_diff_bs, REM_diff_bs = mean_prob_bootstrap(bootstrap_w, bootstrap_n,bootstrap_r)
wake_diff_bs_main1, NREM_diff_bs_main1, REM_diff_bs_main1 = mean_prob_bootstrap_main1(bootstrap_w, bootstrap_n,bootstrap_r)
wake_diff_bs_main2, NREM_diff_bs_main2, REM_diff_bs_main2= mean_prob_bootstrap_main2(bootstrap_w, bootstrap_n,bootstrap_r)
wake_diff_bs_main3, NREM_diff_bs_main3, REM_diff_bs_main3= mean_prob_bootstrap_main3(bootstrap_w, bootstrap_n,bootstrap_r)



wake_inc, wake_dec = increaseORdecrease(wake_diff_bs)
NREM_inc, NREM_dec = increaseORdecrease(NREM_diff_bs)
REM_inc, REM_dec = increaseORdecrease(REM_diff_bs)

wake_inc_main1, wake_dec_main1 = increaseORdecrease(wake_diff_bs_main1)
NREM_inc_main1, NREM_dec_main1 = increaseORdecrease(NREM_diff_bs_main1)
REM_inc_main1, REM_dec_main1 = increaseORdecrease(REM_diff_bs_main1)

wake_inc_main2, wake_dec_main2 = increaseORdecrease(wake_diff_bs_main2)
NREM_inc_main2, NREM_dec_main2 = increaseORdecrease(NREM_diff_bs_main2)
REM_inc_main2, REM_dec_main2 = increaseORdecrease(REM_diff_bs_main2)


wake_inc_main3, wake_dec_main3 = increaseORdecrease(wake_diff_bs_main3)
NREM_inc_main3, NREM_dec_main3 = increaseORdecrease(NREM_diff_bs_main3)
REM_inc_main3, REM_dec_main3 = increaseORdecrease(REM_diff_bs_main3)





