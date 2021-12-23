#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:10:31 2020

@author: jwebb2020
"""
############################################
############################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from scipy.stats import linregress
from statannot import add_stat_annotation
import matplotlib
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
##############################################
##############################################

wt_cells = [241, 269,309, 310, 216, 189] ##ADRB1+ cells in WT mice
casp_cells = [146, 83, 93, 63, 69] ##ADRB1+ cells in ablated mice
casp_wake = [63.75,  54.1, 58.85, 59.95,52.55] ## wake % for ablated mice
wt_wake = [47.35,46.8,47.6, 45.15, 44.8, 45.5] ## wake % for WT mice
wt_sleep = [] ##sleep percentage for WT mice
casp_sleep = [] ## sleep percentage for experimental mice
globalFont = 16 # fontsize for figures
global_name = ''
    
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

def plotDots(wt,wtName, mut,mutName, ylabel,name):
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
    order = [wtName, mutName]
    # test_results = add_stat_annotation(g2, data=framee, x='condition', y='cFos', order=order,
    #                                box_pairs=[(wtName, mutName)],
    #                                test='t-test_ind', text_format='star',
    #                                loc='outside', verbose=2)
    plt.xlabel('')

    plt.ylabel(ylabel, fontsize = globalFont)
    plt.tight_layout()
    # plt.title(name, fontsize = globalFont)
    plt.savefig(name +global_name+".pdf")
    plt.show()
    return

def plotLinearTrend(x, x_name, y, y_name, saveName):
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
    plt.savefig(saveName + '.pdf')
    plt.show() 
    return LINEAR

def plotLinearTrend_2color(x, x_name, y, y_name, saveName, x2, y2):
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
    slope = LINEAR[2] ## basically the r^2 value
    plt.plot(x,p(x), ls = '-', c = 'black')
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=4))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=4))  
    ax1.tick_params(labelsize=globalFont*1)     
    # plt.title(saveName + '_'+ str(slope), fontsize= globalFont)
    plt.savefig(saveName + '.pdf')
    plt.show() 
    return LINEAR

## Calculate sleep % from the wake percent (values add up to 100%)
for i in range(len(casp_wake)):
    q = 100- casp_wake[i]
    casp_sleep.append(q)
for i in range(len(wt_wake)):
    q = 100 - wt_wake[i]
    wt_sleep.append(q)
plotDots(wt_cells, 'WT', casp_cells, 'Cre', '$ADRB1^+$ cells', 'caspase cell counts')    
plotLinearTrend(casp_sleep, 'Total sleep %', casp_cells, 'ADRB1 + cells', 'ADRB1 cell # versus sleep percent')
total_cells = combineMatrix(wt_cells, casp_cells)
total_sleep = combineMatrix(wt_sleep, casp_sleep)
LINEAR =   plotLinearTrend_2color(wt_sleep, 'Total sleep %', wt_cells, 'ADRB1 + cells', 'Total ADRB1 cell # versus sleep percent',casp_sleep, casp_cells)
   
LINEAR =   plotLinearTrend_2color(wt_cells, '$ADRB1^+$ cells',wt_sleep , 'Total sleep %', 'Sleep % versus total ADRB1 cell #', casp_cells, casp_sleep)
    
  