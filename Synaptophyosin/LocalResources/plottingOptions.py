#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 08:26:11 2022

@author: jwebb2020
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from LocalResources import dataWrangling as dw

plot_dir = 'Results/'


def format_curr_cate_for_graphing(cat_index, x_label, y_label):
    curr_x = []
    curr_y = []
    for i in range(len(cat_index)):
        curr_x.append(x_label[cat_index[i]])
        curr_y.append(y_label[cat_index[i]])
    return curr_x, curr_y

def combine_scatter(x,y):
    ALL_x = []
    ALL_y = []
    for i in range(len(y[0])):
        # print(y[i])
        curr_y = dw.extract_column(y,i)
        # print(len(x))
        print(i)
        for j in range(len(x)):
            ALL_x.append(x[j])
            ALL_y.append(curr_y[j])
    return ALL_x, ALL_y



def plotAllAreas(final_val_mean, final_cat_index,final_val, final_label, savename):
    ax1 = plt.axes(frameon=False)
    labels_plot_all = []
    for i in range(len(final_val_mean)):
        labels_plot_all.append(i)
    COLORS = ['navy','deepskyblue','seagreen', 'darkorchid', 'orange','red','darkseagreen', 'y','darkgrey','mediumslateblue']
    for i in range(len(final_cat_index)):
        curr_x, curr_y = format_curr_cate_for_graphing(final_cat_index[i],labels_plot_all,final_val_mean)
        plt.bar(curr_x, curr_y, color = COLORS[i],  width = .8)
    scatter_x, scatter_y = combine_scatter(labels_plot_all, final_val)
    plt.scatter(scatter_x, scatter_y,s =2, facecolors = 'none', edgecolors='black',linewidths= .3,zorder =2 )
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.set_yticklabels([])
    ax1.get_yaxis().set_ticks([])
    plt.xticks(labels_plot_all, final_label, fontsize = 4, rotation = 90)
    ax1.tick_params(axis='both', which='both', length=0)
    ax1.tick_params(axis="x",direction="in", pad=-10)
    plt.tight_layout()
    plt.savefig(plot_dir+savename + "combined.pdf")
    plt.show()
    return