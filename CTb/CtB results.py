#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:32:47 2021

@author: jwebb2020
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.stats as st
from matplotlib.lines import Line2D
import seaborn as sns
from statannot import add_stat_annotation
import pandas as pd
import matplotlib
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
##############################################
##############################################
##############################################
##############################################
name_score = 'ctb results.xlsx' ## user scored epochs
global_name = 'CtB all'
globalFont = 20





def get_tot_per_brain(a):
    new = []
    for i in range(int(len(a)/3)):
        curr = []
        for j in range(3):
            curr.append(a[i*3+j])
        new.append(np.sum(curr))
    return new

def createDataFrame(frame, data, name, index):
    ## creates a pd dataframe given the input variables 
    for i in range(len(data)):
        frame[name][i+index] = data[i]
    return frame

def create_perc(a, b, c):
    a_perc = []
    b_perc = []
    c_perc = []
    for i in range(len(a)):
        SUM = a[i] + b[i] +c[i]
        a_perc.append(a[i]/SUM*100)
        b_perc.append(b[i]/SUM*100)
        c_perc.append(c[i]/SUM*100)
    return a_perc, b_perc, c_perc

# def plot_barplot_3groups(wt,wtName, mut,mutName, wt_2,wt2Name, ylabel,name):
#     ## creates a barplot for 4 groups
#     labels = []

#     for i in range(len(wt)):
#         labels.append(wtName)
#     for i in range(len(mut)):
#         labels.append(mutName)
#     for i in range(len(wt_2)):
#         labels.append(wt2Name)
#     height = len(wt) + len(mut) + len(wt_2)
#     framee = pd.DataFrame(np.random.randint(low=0.0, high= 100, size=(height, 3)), columns=['cFos','cFos_2', 'condition'],dtype = np.float64)
#     curr_length = 0
#     framee = createDataFrame(framee, wt, 'cFos', curr_length)
#     curr_length = curr_length + len(wt)
#     framee = createDataFrame(framee, mut, 'cFos', curr_length)
#     curr_length = curr_length + len(mut)
#     framee = createDataFrame(framee, wt_2, 'cFos', curr_length)
#     curr_length = curr_length + len(wt_2)
#     sns.set_style("ticks")
#     sns.set(font_scale = 1.4)
#     sns.set_style("ticks")
#     print(framee)
#     g1=  sns.barplot(x="condition", y="cFos", data=framee,capsize=.1, errcolor = 'red',palette = ['green','red', 'yellow'], linewidth= 1 )
#     g1.set(xlabel = None)
#     g1.set(xticklabels=[])  # remove the tick labels
#     g2=  sns.swarmplot(x = "condition", y = "cFos", data = framee, palette = ["dimgray",'dimgray', 'dimgray'], alpha = 1, s = 5)
#     g2.set(xlabel= None)
#     g2.set(xticklabels=[])  # remove the tick labels
#     g2.tick_params(bottom=False)  # remove the ticks 
#     sns.despine()
#     plt.xlabel('')
#     plt.ylabel(ylabel, fontsize = globalFont)
#     # plt.title(name, fontsize = globalFont)
#     plt.savefig(name +global_name+".pdf")
#     plt.show()
#     return        

def plot_barplot_3groups(wt,wtName, mut,mutName, over, overName,ylabel,name):
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
    # order = [wtName, mutName]
    # test_results = add_stat_annotation(g2, data=framee, x='condition', y='cFos', order=order,
    #                                box_pairs=[(wtName, mutName)],
    #                                test='t-test_ind', text_format='star',
    #                                loc='outside', verbose=2)
    plt.xlabel('')
    plt.ylabel(ylabel, fontsize = globalFont)
    # plt.title(name, fontsize = globalFont)
    plt.savefig(name +global_name+".pdf")
    plt.show()
    return 


data1 = pd.read_excel(name_score)
name_all = data1['name']
overlap_all = data1['overlap']
lh_all = data1['lh']
pz_all = data1['pz']


overlap_all = overlap_all.values.tolist()
lh_all = lh_all.values.tolist()
pz_all = pz_all.values.tolist()



overlap = get_tot_per_brain(overlap_all)
lh = get_tot_per_brain(lh_all)
pz = get_tot_per_brain(pz_all)

overlap_perc, lh_perc, pz_perc = create_perc(overlap, lh, pz)

# plot_barplot(lh_perc, 'LH', pz_perc, 'PZ','% Signal', 'CtB all')
plot_barplot_3groups(lh_perc, 'LH', pz_perc, 'PZ', overlap_perc, 'LH + PZ', '% signal', 'CtB all')