#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:42:59 2022

@author: jwebb2020
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plot_dir = 'Results/'

def create_x_axis(a, secPerDownsample,secBeforeAndAfter,secondsTick):
    ## creates the x-axis for the heatmap
    new_matrix = []
    num_values = int(secBeforeAndAfter*2/secondsTick)
    include = []
    for i in range(num_values):
        include.append(i*secondsTick)
    shiftedValues = []
    for i in range(len(include)):
        shiftedValues.append(include[i]-secBeforeAndAfter)
    curr_sec =[]
    for i in range(len(a)):
        curr_sec.append(i*secPerDownsample)
    counter = 0
    for i in range(len(curr_sec)):
        if curr_sec[i] in include:
            new_matrix.append(shiftedValues[counter])
            counter+=1
        else:
            new_matrix.append('')
    last = len(new_matrix) -1
    new_matrix[last] = secBeforeAndAfter
    return new_matrix

def create_y_axis(a,trialTick):
    ## creates the y-axis for the heatmap
    new_matrix = []
    num_labels = int(len(a)/trialTick)
    curr_trial = []
    include = []
    for i in range(num_labels):
        include.append((i+1)*trialTick)
    for i in range(len(a)):
        curr_trial.append(i+1)
    counter = 0
    for i in range(len(curr_trial)):
        if curr_trial[i] in include:
            new_matrix.append(include[counter])
            counter+=1
        else:
            new_matrix.append('')
    return new_matrix

def create_heatmap(a, name, globalName, secPerDownsample,secBeforeAndAfter,secondsTick,trialTick): 
    ## creates the heatmap for a single transition type. 'a' is the data; name is the savename
    sns.color_palette("Blues")
    x_labels = create_x_axis(a[1], secPerDownsample,secBeforeAndAfter,secondsTick)
    y_labels = create_y_axis(a,trialTick)
    ax = sns.heatmap(a,xticklabels = x_labels, yticklabels = y_labels, cmap = 'Blues',vmin = -2, vmax = 4, cbar_kws = {'label':'z-scored ∆F/F'})
    ax.tick_params(axis='both', which='both', length=0)
    plt.title(name)
    plt.xlabel('Time (sec)')
    plt.ylabel('Trial #')
    plt.tight_layout()
    plt.savefig(plot_dir+name + '_'+globalName+'_.pdf')
    plt.show()
    return

def zero_transitions(a, amount = 10000):
    ## ensures each trial starts at 0
    subtract = np.mean(a[0:amount])
    for i in range(len(a)):
        a[i] = a[i] - subtract
    return a 

def plot_transitions(data,sem, sec, name, before, after,secBeforeAndAfter, globalFont, globalName):
    ## plots transtions. You feed in raw data (data), error (sem), a seconds matrix, 
    ## the title (name), the before state label (before) and after state label (after)
    beforeColor = ''
    afterColor = ''
    if before == 'REM':
        beforeColor = 'lightsalmon'
    elif before == 'NREM':
        beforeColor = 'lightskyblue'
    else:
        beforeColor = 'lemonchiffon'
    if after == 'REM':
        afterColor = 'lightsalmon'
    elif after == 'NREM':
        afterColor = 'lightskyblue'
    else:
        afterColor = 'lemonchiffon'    
    data = zero_transitions(data)
    ax1 = plt.axes(frameon=False)
    plt.errorbar(sec, data,sem, c = 'black', lw = 2, ecolor= 'dimgrey')
    # plt.xlabel('Time (sec)', fontsize = globalFont*2)
    plt.title(before + '                                  '+after, fontsize = globalFont)
    ax1.axvspan(0,secBeforeAndAfter, color = beforeColor)
    end = int(secBeforeAndAfter*2)
    ax1.axvspan(secBeforeAndAfter, end,  color = afterColor)
    ax1.tick_params(axis='x', which='major', labelsize=8*2)
    ax1.axes.get_yaxis().set_ticks([])
    ax1.axes.get_yaxis().set_visible(False)
    plt.xticks(ticks = [0,15,30,45, 60],labels = ['-30','-15','0','15','30'])
    ax1.xaxis.label.set_size(globalFont*2)
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    plt.tight_layout()
    plt.savefig(plot_dir+name+globalName +'.pdf')
    plt.show()
    return