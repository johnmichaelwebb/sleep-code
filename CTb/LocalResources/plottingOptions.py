#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:23:39 2022

@author: jwebb2020
"""



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plot_dir = 'Results/'

def createDataFrame(frame, data, name, index):
    ## creates a pd dataframe given the input variables 
    for i in range(len(data)):
        frame[name][i+index] = data[i]
    return frame

def plot_barplot_3groups(wt,wtName, mut,mutName, over, overName,ylabel,name, globalFont, global_name):
    ## creates a barplot for 2 groups
    labels = []
    for i in range(len(wt)):
        labels.append(wtName)
    for i in range(len(mut)):
        labels.append(mutName)
    for i in range(len(over)):
        labels.append(overName)
    height = len(wt) + len(mut)
    framee = pd.DataFrame(np.random.randint(low=0.0, high= 100, size=(height, 3)), columns=['cFos','cFos_2', 'condition'],dtype = np.float64)
    framee = createDataFrame(framee, wt, 'cFos', 0)
    framee = createDataFrame(framee, mut, 'cFos', len(wt))
    framee = createDataFrame(framee, over, 'cFos', len(wt)+len(mut))
    framee = createDataFrame(framee, labels, 'condition',0)
    sns.set_style("ticks")
    sns.set(font_scale = 1.4)
    sns.set_style("ticks")
    g1=  sns.barplot(x="condition", y="cFos", data=framee,capsize=.2, errcolor = 'black',palette = ["green",'red','yellow'], linewidth =1 )
    g1.set(xlabel = None)
    g1.set(xticklabels=[])  # remove the tick labels
    g2=  sns.swarmplot(x = "condition", y = "cFos", data = framee, palette = ["dimgray",'dimgray', 'dimgray'], alpha = 1, s = 8)
    g2.set(xlabel= None)
    g2.set(xticklabels=[])  # remove the tick labels
    g2.tick_params(bottom=False)  # remove the ticks    
    sns.despine()
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont)
    plt.savefig(plot_dir +global_name+".pdf")
    plt.show()
    return 