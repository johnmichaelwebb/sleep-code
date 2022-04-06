#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:50:37 2022

@author: jwebb2020
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib
import matplotlib.ticker as mticker

data_dir = 'Data/'
plot_dir = 'Results/'



def piePlot(a, name,saveName):
    ## generates of pie plot of % connected cells
    a_perc = np.sum(a)/len(a)*100
    other = 100-a_perc
    fig1, ax1 = plt.subplots()
    labels = [name, '']
    explode  = [0.1,0]
    sizes = [a_perc, other]
    ax1.pie(sizes, explode=explode, labels=labels,  shadow=True, startangle=90, colors = ['blue','lightgrey'])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(plot_dir+name + '_' +saveName +'.pdf')
    plt.show()    
    return



def create_heatmap_single(a, brainAreas, name,trialTick,saveName): 
    ## creates a heatmap of connected cells
    x_labels = brainAreas
    y_labels = create_y_axis(a, trialTick)
    sns.set(font_scale = 1.3)

    ax = sns.heatmap(a,xticklabels = x_labels, yticklabels = y_labels, linecolor = 'gray',linewidths = .005, cbar =False,cmap='YlGnBu')
    ax.tick_params(left=False, bottom=False)    
    plt.ylabel('Neuron #')
    plt.savefig(plot_dir+name +'_' + saveName +'_.pdf')
    plt.show()
    return

def create_y_axis(a, trialTick):
    ## creates y-axis for the matrix
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



def CRACM_total_plot(perc_totals, perc_names, perc_names_1):
    ##Plots the percent of cells with a give gene expression
    ax1 = plt.axes(frameon=False)
    plt.bar(perc_names, perc_totals, color = 'black')
    plt.ylim(0,102)
    plt.ylabel('Percent of cells')
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
    ax1.yaxis.set_major_locator(mticker.FixedLocator([0, 20, 40,60,80,100]))
    ax1.axhline(100, c = 'black', ls = '--')
    plt.xticks(perc_names, perc_names_1)
    plt.savefig(plot_dir+'CRACM_total.pdf') 
    plt.show()
    return