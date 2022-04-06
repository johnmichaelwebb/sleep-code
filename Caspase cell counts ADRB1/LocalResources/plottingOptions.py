#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:32:11 2022

@author: jwebb2020
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from scipy.stats import linregress

plot_dir = 'Results/'

def combineMatrix(a,b):
    ## combine many matrixes into a single matrix    
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(a[i])
    for i in range(len(b)):
        new_matrix.append(b[i])
    return new_matrix

def createDataFrame(frame, data, name, index):
    ## creates a pd dataframe given the input variables 
    for i in range(len(data)):
       # print(data[i])
        frame[name][i+index] = data[i]
    return frame

def plotDots(wt,wtName, mut,mutName, ylabel,name, globalFont, global_name):
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
    plt.tight_layout()
    g2 = sns.boxplot(x="condition", y="cFos", data=framee, palette = ["lightgray",'dodgerblue'], linewidth =1)
    sns.swarmplot(x = "condition", y = "cFos", data = framee,palette = ["dimgray",'dimgray'], alpha = 1, s = 8)
    sns.despine()
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont)
    plt.tight_layout()
    plt.savefig(plot_dir+name +global_name+".pdf")
    plt.show()
    return

def plotLinearTrend(x, x_name, y, y_name, saveName, globalFont, global_name):
    ## plot linear trend with just one color for the dots
    ax1 = plt.axes(frameon=False)
    plt.scatter(x, y, s = 100, c = 'black')
    plt.xlabel(x_name, fontsize = globalFont)
    plt.ylabel(y_name,fontsize = globalFont)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    LINEAR = linregress(x, y)
    slope = LINEAR[2]
    plt.plot(x,p(x), ls = '-', c = 'black')
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=4))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=4))  
    ax1.tick_params(labelsize=globalFont*1)     
    plt.tight_layout()
    plt.title(saveName + '_'+ str(slope), fontsize= globalFont)
    plt.savefig(plot_dir + saveName + '.pdf')
    plt.show() 
    return LINEAR

def plotLinearTrend_2color(x, x_name, y, y_name, saveName, x2, y2, globalFont, global_name):
    ##plot 2 groups of mice with a linear trendline
    ax1 = plt.axes(frameon=False)
    plt.scatter(x, y, s = 100, c = 'darkgray')
    plt.scatter(x2, y2, s = 100, c = 'dodgerblue')
    plt.xlabel(x_name, fontsize = globalFont)
    plt.ylabel(y_name,fontsize = globalFont)
    plt.tight_layout()
    x = combineMatrix(x, x2)
    y = combineMatrix(y, y2)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    LINEAR = linregress(x, y)
    plt.plot(x,p(x), ls = '-', c = 'black')
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=4))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=4))  
    ax1.tick_params(labelsize=globalFont*1)     
    plt.savefig(plot_dir+ saveName + '.pdf')
    plt.show() 
    return LINEAR