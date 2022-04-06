#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:42:59 2022

@author: jwebb2020
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.stats as st
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import matplotlib
from statannot import add_stat_annotation

from LocalResources import dataWrangling as dw

plot_dir = 'Results/'


def createDataFrame(frame, data, name, index):
    ## creates a pd dataframe given the input variables 
    for i in range(len(data)):
        frame[name][i+index] = data[i]
    return frame

def plotDots(wt,wtName, mut,mutName, ylabel,name, globalFont, global_name, wake = False):
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
    plt.tight_layout()
    if wake == True:
        plt.yticks(ticks = [12, 14, 16],labels = ['12','14','16'])
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont*1.2)
    plt.savefig(plot_dir+name +global_name+".pdf")
    plt.show()
    return

def plotDots_stats(wt,wtName, mut,mutName, ylabel,name,globalFont, global_name, title =False):
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
    sns.set(font_scale = 1.8)
    sns.set_style("ticks")
    if title == True:
        plt.title(name, fontsize = globalFont)
    plt.tight_layout()
    g2 = sns.boxplot(x="condition", y="cFos", data=framee, palette = ["lightgray",'dodgerblue'], linewidth =1)
    sns.swarmplot(x = "condition", y = "cFos", data = framee,palette = ["dimgray",'dimgray'], alpha = 1, s = 8)
    sns.despine()
    order = [wtName, mutName]
    test_results = add_stat_annotation(g2, data=framee, x='condition', y='cFos', order=order,
                                   box_pairs=[(wtName, mutName)],
                                   test='t-test_ind', text_format='star',
                                   loc='outside', verbose=2)
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont)
    plt.tight_layout()

    plt.savefig(plot_dir +name +global_name+".pdf")
    plt.show()
    return

def plot_barplot(wt,wtName, mut,mutName, ylabel,name, globalFont, global_name):
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
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont)
    plt.savefig(plot_dir+name +global_name+".pdf")
    plt.show()
    return 

def plot_barplot_REM(wt,wtName, mut,mutName, ylabel,name,globalFont, global_name):
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
    plt.yticks(ticks = [0, 50, 100],labels = ['0','50','100'])
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont*2)
    plt.tight_layout()
    plt.savefig(plot_dir+name +global_name+".pdf")
    plt.show()
    return 

def plot_barplot_4groups(wt,wtName, mut,mutName, wt_2,mut_2, ylabel,name,globalFont, global_name):
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
    sns.despine()
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont)
    plt.savefig(plot_dir+name +global_name+".pdf")
    plt.show()
    return

def plot_barplot_6groups(wt,wtName, mut,mutName, wt_2,mut_2,wt_3, mut_3, ylabel,name,globalFont, global_name):
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
    sns.despine()
    plt.xlabel('')
    plt.ylim(0,150)
    plt.ylabel(ylabel, fontsize = globalFont)
    plt.savefig(plot_dir+name +global_name+".pdf")
    plt.show()
    return

def bout2sec(a,bout_length):
    ## calculate bout length from number of bouts
    for i in range(len(a)):
        a[i] = a[i]*bout_length/60
    return a

def combineMatrix(a,bout_length):
    ## combine many matrixes into a single matrix
    matrix = []
    for i in range(len(a)):
        for j in range(len(a[i])):
            matrix.append(a[i][j])
    matrix = bout2sec(matrix,bout_length)
    return matrix

def plotHist(wt, mut, xname, yname, title, wtname, mutname, bout_length, globalFont, global_name, BIN = 30):
    ## plot histogram of state bout lengths
    wt = combineMatrix(wt,bout_length)
    mut = combineMatrix(mut,bout_length)
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
    plt.savefig(plot_dir+title + global_name+'.pdf')
    plt.show()
    return

def plotPerc(wtAVG, wtSTD, mutAVG, mutSTD,name, wt_name, mut_name, yname,globalFont, global_name): 
    ## plot percent by hour
    x_axis = dw.create_hrs()    
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
    plt.tight_layout()
    # plt.title(title, fontsize = globalFont*.5)
    plt.savefig(plot_dir+name + '_' + global_name+'.pdf')
    plt.show()
    return

def ttest_by_hr(wt, mut):
    ## generate the t test by hour
    t = []
    for i in range(len(wt[0])):
        wt_curr = dw.extract_column(wt, i)
        mut_curr = dw.extract_column(mut,i)
        curr_t = st.ttest_ind(wt_curr,mut_curr, equal_var = False)
        curr_t = curr_t[1]
        t.append(curr_t)
    return t
def create_hrs(std):
    ## creates 24 hour matrix
    interval = 24/len(std)
    matrix = []
    for i in range(len(std)):
        matrix.append(i*interval)
    return matrix
def plot_perc(ylabel, wtAVG, wtSTD, mutAVG, mutSTD,WT_NAME,MUT_NAME, name,globalFont, global_name):
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
    plt.savefig(plot_dir+title + global_name+'.pdf')
    plt.show()
    return