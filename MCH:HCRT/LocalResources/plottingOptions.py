#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:50:47 2022

@author: jwebb2020
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
from LocalResources import dataWrangling as dw

plot_dir = 'Results/'


def histAll(a, name,savename):
    c = []
    for i in range(len(a)):
        if len(c) == 0:
            c = a[i]
        else:
            c = dw.combineMatrix(c, a[i])
    ax1 = plt.axes(frameon=False)
       
            
    plt.hist(c, bins = 200, color = 'black')
    
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
    plt.ylabel('Number of synapses')
    plt.xlabel('Nearest synapse (µm)')
    plt.tight_layout()

    # plt.title('All synapses'+ '_' +savename)
    plt.savefig(plot_dir+ 'all_synapses' + savename + '.pdf')
    plt.show()
    
    return c


def plot_histogram(allDistances,COLOR, savename):
    ax1 = plt.axes(frameon=False)
    plt.hist(allDistances, density = True, color = COLOR,bins = 200)
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
    plt.ylabel('Number of Synapses')
    plt.xlabel('Nearest synapse (µm)')
    plt.tight_layout()
    plt.savefig(plot_dir+'different brains_fullhistogram_density' + savename +'.pdf')
    plt.show()
    
    ax1 = plt.axes(frameon=False)
    plt.hist(allDistances, density = True, color = COLOR,bins = 30, range = (0,30))
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
    plt.ylabel('Number of fibers')
    plt.xlabel('Nearest fiber (µm)')
    plt.tight_layout()
    plt.savefig(plot_dir+'different brains_shorthistogram_density' + savename +'.pdf')
    plt.show()
    return

def plot_histogram_stacked(allDistances,COLOR, savename):
    ax1 = plt.axes(frameon=False)
    # colors = ['black','blue']
    for i in range(len(allDistances)):
        plt.hist(allDistances[i], density = False, stacked = True, fill = False, color = COLOR[i],  bins = 200,edgecolor = COLOR[i])
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
    plt.ylabel('Number of fibers')
    plt.xlabel('  (µm)')
    plt.tight_layout()
    plt.savefig(plot_dir+'diff brains stacked hist_full.pdf' + savename + '.pdf')
    plt.show()
    
    
    ax1 = plt.axes(frameon=False)
    # colors = ['black','blue']
    for i in range(len(allDistances)):
        plt.hist(allDistances[i], density = False, stacked = True, fill = False, color = COLOR[i], range = (0,30), bins = 30,edgecolor = COLOR[i])
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
    plt.ylabel('Number of fibers')
    plt.xlabel('Nearest fiber (µm)')
    plt.tight_layout()
    plt.savefig(plot_dir+'diff brains stacked hist_short.pdf' + savename + '.pdf')
    plt.show()
    return

def plot_histogram_rawNum(allDistances,COLOR, savename):
    ax1 = plt.axes(frameon=False)
    plt.hist(allDistances, density = False, color = COLOR,bins = 200)
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
    plt.ylabel('Number of fibers')
    plt.xlabel('Nearest fiber (µm)')
    plt.tight_layout()
    plt.savefig(plot_dir+'different brains_fullhistogram_raw numbers' + savename +'.pdf')
    plt.show()
    
    ax1 = plt.axes(frameon=False)
    plt.hist(allDistances, density = False, color = COLOR,bins = 30, range = (0,30))
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
    plt.ylabel('Number of fibers')
    plt.xlabel('Nearest fiber (µm)')
    plt.tight_layout()
    plt.savefig(plot_dir+'different brains_shorthistogram_raw numbers' + savename +'.pdf')
    plt.show()
    return

def plot_histogram_rawNum_stacked(allDistances,COLOR, savename):
    ax1 = plt.axes(frameon=False)
    # colors = ['black','blue']
    for i in range(len(allDistances)):
        plt.hist(allDistances[i], density = False, stacked = True, fill = False, color = COLOR[i],  bins = 200,edgecolor = COLOR[i])
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
    plt.ylabel('Number of fibers')
    plt.xlabel('Nearest fiber (µm)')
    plt.tight_layout()
    plt.savefig(plot_dir+'diff brains stacked hist_full_raw numbers' + '_'+savename+'.pdf')
    plt.show()
    
    
    ax1 = plt.axes(frameon=False)
    # colors = ['black','blue']
    for i in range(len(allDistances)):
        plt.hist(allDistances[i], density = False, stacked = True, fill = False, color = COLOR[i], range = (0,30), bins = 30,edgecolor = COLOR[i])
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
    plt.ylabel('Number of fibers')
    plt.xlabel('Nearest fiber (µm)')
    plt.tight_layout()
    plt.savefig(plot_dir+'diff brains stacked hist_short_raw numbers' + '_'+savename+'.pdf' + savename + '.pdf')
    plt.show()
    return


