#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:53:39 2020

@author: jwebb2020
"""
############################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import pandas as pd
##############################################
##############################################
##############################################
global_name = 'ADRB1 in situ_'
wt = [1026, 1281, 992] ## wt no sleep deprivation values
sd = [401, 1073,404] ## sleep deprivation values
globalFont = 18 ## sets font size for figures

def createDataFrame(frame, data, name, index):
    ## creates a pd dataframe given the input variables 
    for i in range(len(data)):
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
    g2 = sns.boxplot(x="condition", y="cFos", data=framee, palette = ["lightgray",'dodgerblue'], linewidth =1)
    sns.swarmplot(x = "condition", y = "cFos", data = framee,palette = ["dimgray",'dimgray'], alpha = 1, s = 8)
    sns.despine()
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont)
    plt.tight_layout()
    plt.savefig(name +global_name+".pdf")
    plt.show()
    return
    
wt_sd_ttest =st.ttest_ind(wt,sd, equal_var = False)
wt_sd_ttest= wt_sd_ttest[1]
plotDots(wt, 'No SD', sd, 'SD', 'cFos+ and ADRB1+ cells', 'cFos+ and ADRB1+ cells')
