#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:29:32 2022

@author: jwebb2020
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as st
from statannot import add_stat_annotation
import matplotlib
from matplotlib.lines import Line2D


plot_dir = 'Results/'


def plot_firingRateHist(total_spike_avg):
    ax1 = plt.axes(frameon=False)
    plt.hist(total_spike_avg, color = ['black'], density = False, bins = 15)
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
    plt.ylabel('# Units')
    plt.xlabel('Firing rate (Hz)')
    plt.tight_layout()
    plt.savefig(plot_dir+"firing rate hist.pdf")
    plt.show()
    return 

def createDataFrame(frame, data, name, index):
    ## creates a pd dataframe given the input variables 
    for i in range(len(data)):
        frame[name][i+index] = data[i]
    return frame

def plotDots_3(pvalues,wake,wtName, NREM,mutName, ylabel,name, REM, mut2name,globalFont):
    labels = []
    for i in range(len(wake)):
        labels.append(wtName)
    for i in range(len(NREM)):
        labels.append(mutName)
    for i in range(len(REM)):
        labels.append(mut2name)  
    height = len(wake) + len(NREM) +len(REM)
    framee = pd.DataFrame(np.random.randint(low=0.0, high=8, size=(height, 2)), columns=['cFos', 'condition'],dtype = np.float64)
    framee = createDataFrame(framee, wake, 'cFos', 0)
    framee = createDataFrame(framee, NREM, 'cFos', len(wake))
    framee = createDataFrame(framee, REM, 'cFos', len(wake)+len(NREM))
    framee = createDataFrame(framee, labels, 'condition',0)
    sns.set_style("ticks")
    sns.set(font_scale = 1.7)
    sns.set_style("ticks")
    # order = [wtName, mutName, mut2name]
    # test_short_name = 'ttest'
    g2 = sns.boxplot(x="condition", y="cFos", data=framee, palette = ["lemonchiffon",'lightblue','lightsalmon'], linewidth = .5 )
    # test_results = add_stat_annotation(g2, data=framee, x='condition', y='cFos', order=order,
    #                                box_pairs=[(wtName, mutName),(wtName, mut2name), (mutName, mut2name)],
    #                               perform_stat_test=False, pvalues=pvalues, test_short_name=test_short_name,
    #                                text_format='star', verbose=2)
    for i in range(len(wake)):
        indexes = [i, len(wake)*1 +i, len(wake)*2+i]
        curr_line = framee.iloc[indexes]
        sns.lineplot(x = 'condition', y = 'cFos',data = curr_line, sort = False, color = 'black', linewidth = 1)
    sns.despine()
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont)
    # plt.title(name, fontsize = globalFont)
    plt.savefig(plot_dir+name +".pdf",bbox_inches="tight")
    plt.show() 
    return framee


def returnWakeStuff(wake, NREM, REM, globalFont, globalName):
    ## creates the wake-NREM and NREM-REM plot
    wakeMinusNREM = []
    REMMinusNREM = []
    for i in range(len(wake)):
        wakeMinusNREM.append(wake[i]-NREM[i])
        REMMinusNREM.append(REM[i]-NREM[i])  
    d = {'wIn':wakeMinusNREM, 'rIn':REMMinusNREM}
    df = pd.DataFrame(data=d)
    sns.set(font_scale = 1.5)
    sns.set_style('white')
    plt.tight_layout()
    
    g = sns.JointGrid(data=df, 
                      x="wIn",
                      y="rIn", )
    plt.tight_layout()
    g.plot_joint(sns.scatterplot, s = 70,color='none', edgecolor='dodgerblue', linewidth =1.5)
    # plt.xlim(0,.1)

    plt.axvline(x=0, c= 'black')
    plt.axhline(y=0, c = 'black')
    plt.xlabel('Wake - NREM (Z-score)', fontsize = globalFont)
    plt.ylabel( 'REM - NREM (Z-score)', fontsize = globalFont)
    plt.tight_layout()
    
    g.plot_marginals(sns.kdeplot, linewidth =4, color = 'dodgerblue')    
    plt.tight_layout()
    plt.yticks(ticks = [-0.05, 0, 0.05],labels = ['-0.05','0','0.05'])
    # plt.xlim(0,.1)
    plt.savefig(plot_dir+'wake vs NREM vs REM_'  +globalName +'.pdf')
    plt.show()
    return