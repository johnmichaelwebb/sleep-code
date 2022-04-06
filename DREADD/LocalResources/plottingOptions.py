#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 20:58:09 2022

@author: jwebb2020
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

plot_dir = 'Results/'

def createDataFrame(frame, data, name, index):
    for i in range(len(data)):
       # print(data[i])
        frame[name][i+index] = data[i]
    return frame

def plotDots(wt,wtName, mut,mutName, ylabel,name, globalFont, global_name):
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
    plt.savefig(plot_dir+name +global_name+".pdf")
    plt.show()
    return
    


def create_hrs(std, hrs):
    interval = hrs/len(std)
    matrix = []
    for i in range(len(std)):
        matrix.append(i*interval+1)
    return matrix

def plot_perc(wtAVG, wtSTD, mutAVG, mutSTD, tot_hrs,name, globalFont, global_name):
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
    plt.savefig(plot_dir+ title + global_name+'.pdf')
    plt.show()
    return

def plot_perc_4groups(wtAVG, wtSTD, mutAVG, mutSTD, wt2AVG, wt2STD, mut2AVG, mut2STD, tot_hrs,name, ylabel, globalFont, global_name):
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
    plt.savefig(plot_dir+title + global_name+'.pdf')
    plt.show()
    return

def plotWithLines(day1, day2, name, globalFont, global_name):
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
    plt.savefig(plot_dir+name + global_name+'.pdf')
    plt.show()
    return
    
def createHalf(a):
    stop = int(len(a[0])/2)
    for i in range(len(a)):
        a[i] = a[i][0:stop]
    return a

def plot_boxplot_4groups(pval, wt,wtName, mut,mutName, wt_2,mut_2, ylabel,name, globalFont, global_name):
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
    plt.savefig(plot_dir+name +global_name+".pdf")
    plt.show()
    return

