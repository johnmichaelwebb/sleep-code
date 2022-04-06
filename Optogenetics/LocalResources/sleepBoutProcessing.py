#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:42:25 2022

@author: jwebb2020
"""

import numpy as np
from LocalResources import dataWrangling as dw

max_bout = 8640 ## the maximum number of bouts in the file 
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

def bout_len(pos, full,  name, max_bout):
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

def find_all_len(a, max_bout):
    ## essentially finds the REM, NREM and Wake bout lengths from the raw data
    REM_len = []
    NREM_len = []
    wake_len = []
    for i in range(len(a)):
        curr = a[i]
        curr_REM_loc = bout_loc(curr, 'REM')
        curr_REM_len = bout_len(curr_REM_loc, curr, 'REM', max_bout)
        REM_len.append(curr_REM_len)
    for i in range(len(a)):
        curr = a[i]
        
        curr_NREM_loc = bout_loc(curr, 'NREM')
        curr_NREM_len = bout_len(curr_NREM_loc, curr, 'NREM',max_bout)
        NREM_len.append(curr_NREM_len)        
    for i in range(len(a)):
        curr = a[i]
        curr_wake_loc = bout_loc(curr, 'Wake')
        curr_wake_len = bout_len(curr_wake_loc, curr, 'Wake', max_bout)
        wake_len.append(curr_wake_len)                
    return REM_len, NREM_len, wake_len

def avg_bout_len(a, bout_length):
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

def hour2percent(a, name,bout_length, minutes):
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

def percent_by_hour(a,  bout_length,days, minutes = 60):
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
            curr_REM = hour2percent(toAppend, 'REM', bout_length,minutes = minutes)
            curr_NREM = hour2percent(toAppend, 'NREM', bout_length, minutes = minutes)
            curr_wake = hour2percent(toAppend, 'Wake', bout_length, minutes = minutes)
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

def replaceSDwithWake(a, name, limit, bout_length):
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

def prob_single_state(a, STATE):
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

def average_days(a):
    start = 0
    stop = 2
    new_a = []
    for i in range(int(len(a)/2)):
        new_a.append(np.mean(a[start:stop]))
        start +=2
        stop+=2
    return new_a

def prob_state_all(a):
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
    NREM_to_NREM = average_days(NREM_to_NREM)
    NREM_to_wake = average_days(NREM_to_wake)
    NREM_to_REM = average_days(NREM_to_REM)
    wake_to_wake = average_days(wake_to_wake)
    wake_to_NREM = average_days(wake_to_NREM)
    wake_to_REM = average_days(wake_to_REM)
    REM_to_wake = average_days(REM_to_wake)
    REM_to_NREM = average_days(REM_to_NREM)
    REM_to_REM = average_days(REM_to_REM)    
    return wake_to_wake, wake_to_NREM, wake_to_REM, NREM_to_wake, NREM_to_NREM, NREM_to_REM, REM_to_wake, REM_to_NREM, REM_to_REM

