#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:51:32 2022

@author: jwebb2020
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn
from sklearn.decomposition import PCA

plot_dir = 'Results/'

def create_heatmap_single(a, brainAreas, name,MOUSENAME, trialTick): 
    #create heatmap for one area
    x_labels = brainAreas
    y_labels = create_y_axis(a, trialTick)
    ax = seaborn.heatmap(a,xticklabels = x_labels, yticklabels = y_labels, linecolor = 'gray',linewidths = 0, cbar =False,cmap='YlGnBu')
    ax.tick_params(left=False, bottom=False)    
    plt.ylabel('Neuron #')
    name = str(name)
    plt.savefig(plot_dir+name +'_' + MOUSENAME +'_.pdf')
    plt.show()
    return

def create_heatmap_all(a, brainAreas, MOUSENAME, trialTick):
    ## create heatmap for all areas
    for i in range(len(a)):
        create_heatmap_single(a[i], brainAreas, brainAreas[i],MOUSENAME, trialTick)
    return

def create_y_axis(a, trialTick):
    ## create y-axis for heatmap
    new_matrix = []
    num_labels = int(len(a)/trialTick)
    curr_trial = []
    include = []
    for i in range(num_labels):
        include.append((i+1)*trialTick)
    for i in range(len(a)):
        curr_trial.append(i+1)
    counter = 0
    for i in range(len(curr_trial)):
        if curr_trial[i] in include:
            new_matrix.append(include[counter])
            counter+=1
        else:
            new_matrix.append('')
    return new_matrix

def plot_PCA(binary4PCA):
    ax1 = plt.axes(frameon=False)
    pca = PCA(n_components=2)
    # h  = pca.fit(binary4PCA)
    # X_pca = pca.transform(binary4PCA)
    pca = PCA(2)  # project from many to 2 dimensions
    projected = pca.fit_transform(binary4PCA)
    # test = projected[:, 0]
    plt.scatter(projected[:, 0], projected[:, 1], color = 'black', edgecolor='none', alpha=0.5)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=4))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=4))  
    plt.tight_layout()
    plt.savefig(plot_dir+ '_PCAall_.pdf')
    plt.show()
    return

def plot_numTotalProjectingNeurons(projectionLabels_forplot,TOTAl_Projecting_Neurons_to_area,projectionLabels_sorted):
    ax1 = plt.axes(frameon=False)
    plt.bar(projectionLabels_forplot, TOTAl_Projecting_Neurons_to_area, color = 'black')
    xmin, xmax = ax1.get_xaxis().get_view_interval()
    ymin, ymax = ax1.get_yaxis().get_view_interval()
    ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
    plt.ylabel('# Neurons')
    plt.xticks(projectionLabels_forplot, projectionLabels_sorted, fontsize = 10)
    plt.tight_layout()
    plt.savefig(plot_dir+"number projecting neurons total.pdf")
    plt.show()
    return