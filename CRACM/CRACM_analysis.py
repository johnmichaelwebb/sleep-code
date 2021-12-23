#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:52:09 2020

@author: jwebb2020
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib
import matplotlib.ticker as mticker

matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')



fileName = 'CRACM results_connected.csv' ## raw datas
saveName = 'CRACM connected' ## attached to saved files
trialTick = 5
projectionLabels = ['GAD1', 'LHX6','MCH', 'HCRT'] ## cell labels

def extract_column(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(a[i][column])
    return new_matrix    

def matrix_float(a):
    #converts a matrix of strings to floats
    ##often necessary when importing the data csv files into python
    new_matrix = []
    for i in range(len(a)):
        try:
            new_matrix.append((float(a[i])))
        except ValueError:
            print(i)
            print("matrixFloat")
    return new_matrix 

def extract_row(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a[0])):
        new_matrix.append(a[column][i])
    return new_matrix 

def piePlot(a, name):
    ## generates of pie plot of % connected cells
    a_perc = np.sum(a)/len(a)*100
    other = 100-a_perc
    fig1, ax1 = plt.subplots()
    labels = [name, '']
    explode  = [0.1,0]
    sizes = [a_perc, other]
    ax1.pie(sizes, explode=explode, labels=labels,  shadow=True, startangle=90, colors = ['blue','lightgrey'])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(name + '_' +saveName +'.pdf')
    plt.show()    
    return

def create_heatmap_single(a, brainAreas, name): 
    ## creates a heatmap of connected cells
    x_labels = brainAreas
    y_labels = create_y_axis(a)
    sns.set(font_scale = 1.3)

    ax = sns.heatmap(a,xticklabels = x_labels, yticklabels = y_labels, linecolor = 'gray',linewidths = .005, cbar =False,cmap='YlGnBu')
    ax.tick_params(left=False, bottom=False)    
    plt.ylabel('Neuron #')
    plt.savefig(name +'_' + saveName +'_.pdf')
    plt.show()
    return

def create_y_axis(a):
    ## creates y-axis for the matrix
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

def create_heatmap_matrix(a):
    ## creates the matrix for a heatmap
    a = a[1:len(a)-2]
    for i in range(len(a)):
        a[i] = a[i][1:len(a[i])-1]
        a[i] = matrix_float(a[i])
    return a

def switch_columns(a, labels, first, second):
    ## switch the order of the columns and labels
    one = extract_column(a, first)
    two = extract_column(a, second)
    label_one = labels[first]
    label_two = labels[second]
    for i in range(len(a)):
        a[i][first] = two[i]
        a[i][second] = one[i]
    labels[first] = label_two
    labels[second] = label_one
    return a, labels

def sortByColumn(a, column):
    ## makes sure that values equaling 1 rise to the top of the column
    to_sort = extract_column(a, column)
    pos = []
    neg =[]
    for i in range(len(to_sort)):
        if to_sort[i] == 1:
            pos.append(i)
        else:
            neg.append(i)
    new = []
    for i in range(len(pos)):
        curr = extract_row(a, pos[i])
        new.append(curr)
    for i in range(len(neg)):
        curr = extract_row(a, neg[i])
        new.append(curr)
    return new

def perc(a):
    ## gets the percent for a list of values
    a_perc = np.sum(a)/len(a)*100
    return a_perc

with open(fileName, newline='\n' ) as inputfile:
   data = list(csv.reader(inputfile)) 
GAD = extract_column(data[1:len(data)], 1)
GAD = matrix_float(GAD)
GAD_perc = np.sum(GAD)/len(GAD)*100
LHX6 = extract_column(data[1:len(data)], 2)
LHX6 = matrix_float(LHX6)
LHX6_perc = np.sum(LHX6)/len(LHX6)*100
MCH = extract_column(data[1:len(data)], 3)
MCH = matrix_float(MCH)
MCH_perc = np.sum(MCH)/len(MCH)*100
orexin = extract_column(data[1:len(data)], 4)
orexin = matrix_float(orexin)
OREXIN_perc = np.sum(orexin)/len(orexin)*100
heatmap_data = create_heatmap_matrix(data)
heatmap_data = sortByColumn(heatmap_data, 0)
heatmap_data = sortByColumn(heatmap_data, 2)
heatmap_data = sortByColumn(heatmap_data, 1)
create_heatmap_single(heatmap_data, projectionLabels, 'heatmap')
piePlot(GAD, 'GAD')
piePlot(MCH, 'MCH')
piePlot(LHX6, 'LHX6')
piePlot(orexin, 'HCRT')
GAD_perc = perc(GAD)
LHX6_perc = perc(LHX6)
MCH_perc = perc(MCH)
orexin_perc = perc(orexin)
perc_totals = [GAD_perc, MCH_perc, LHX6_perc,  orexin_perc]

perc_names_1 = ['GAD','MCH','LHX6','HCRT']
perc_names  = [1,2,3,4]
plt.show()
ax1 = plt.axes(frameon=False)
plt.bar(perc_names, perc_totals, color = 'black')
plt.ylim(0,102)
plt.ylabel('Percent of cells')
# plt.yticks([0, 20, 40,60,80,100], ['0','20','40','60','80','100'])

xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
ax1.yaxis.set_major_locator(mticker.FixedLocator([0, 20, 40,60,80,100]))

ax1.axhline(100, c = 'black', ls = '--')
# plt.yticks([0, 20, 40,60,80,100], ['0','20','40','60','80','100'])
plt.xticks(perc_names, perc_names_1)

# ax1.tick_params(axis = 'y',direction = 'out')
plt.savefig('CRACM_total.pdf') 
plt.show()