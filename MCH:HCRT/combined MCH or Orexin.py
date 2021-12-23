#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 09:37:42 2021

@author: jwebb2020
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from math import atan2
from math import pi
from scipy import signal
from sklearn import metrics
from sklearn import feature_selection
#from pyitlib import discrete_random_variable as drv
import h5py
import scipy.io as sio
from math import*
import math
import sys
from scipy import signal
from sklearn import metrics
from sklearn import feature_selection
#from pyitlib import discrete_random_variable as drv
import h5py
import scipy.io as sio
from matplotlib.lines import Line2D
from six import iteritems
import seaborn as sns
import scipy.stats as st
import matplotlib
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')


gross_name = 'mch 4 mice gross instructions.csv'
savename = gross_name[0:len(gross_name)-4]

matplotlib.rcParams.update({'font.size': 22})


def extract_row(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a[0])):
        new_matrix.append(a[column][i])
    return new_matrix 


def extract_column(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(a[i][column])
    return new_matrix 

def extractColumnByLabel(a, columnName):
    names = extract_row(a, 0)
    index = names.index(columnName)
    column = extract_column(a, index)
    
    return column


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

def singMouseID(a, label):
    ID = []
    for i in range(len(a)):
        if a[i] == label:
            ID.append(i)
    return ID

def mouse_indexes(a):
    mouseID = []
    for i in range(len(a)-1):
        if a[i+1] not in mouseID:
            mouseID.append(a[i+1])
    mouseIndex = []
    for i in range(len(mouseID)):
        currIndex = singMouseID(a, mouseID[i])
        mouseIndex.append(currIndex)
    return mouseIndex

def combineMatrix(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i])
    for i in range(len(b)):
        c.append(b[i])  
    return c


def importSingMouse(a,filenames):
    total_distances = []
    for i in range(len(a)):
        index = a[i]
        
        with open(filenames[index], newline='\n' ) as inputfile:
            data = list(csv.reader(inputfile))
        for i in range(len(data)):
            data[i] = data[i][0]
        data = matrix_float(data)
        if len(total_distances) == 0:
            total_distances = data
        else:
            total_distances = combineMatrix(total_distances, data)
    
    return total_distances
    
def importAllMice(a, filenames):
    distPerMouse = []
    for i in range(len(a)):
        currDist = importSingMouse(a[i], filenames)
        distPerMouse.append(currDist)
    return distPerMouse

def histAll(a, name):
    c = []
    for i in range(len(a)):
        if len(c) == 0:
            c = a[i]
        else:
            c = combineMatrix(c, a[i])
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
    plt.savefig('all_synapses' + savename + '.pdf')
    plt.show()
    
    return c


with open(gross_name, newline='\n' ) as inputfile:
    data = list(csv.reader(inputfile)) 
    data[0][0] = data[0][0][1:len(data[0][0])]





filename = extractColumnByLabel(data, 'filename')
mouse_column = extractColumnByLabel(data, 'mouse')
mouse_index = mouse_indexes(mouse_column)

# singMouseTest = importSingMouse(mouse_index[0], filename)
allDistances = importAllMice(mouse_index, filename)
test = histAll(allDistances, 'all dist')

if len(allDistances) == 1:
    COLOR = 'black'
elif len(allDistances) == 2:
    COLOR = ['black', 'blue']
elif len(allDistances) == 3:
    COLOR = ['black', 'blue','green']
elif len(allDistances) == 4:
    
    COLOR = ['black', 'blue','green','orange']





ax1 = plt.axes(frameon=False)
plt.hist(allDistances, density = True, color = COLOR,bins = 200)
xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
plt.ylabel('Number of Synapses')
plt.xlabel('Nearest synapse (µm)')
plt.tight_layout()

# plt.title('different brains_fullhistogram_density'+ '_' +savename)
plt.savefig('different brains_fullhistogram_density' + savename +'.pdf')
plt.show()

ax1 = plt.axes(frameon=False)
plt.hist(allDistances, density = True, color = COLOR,bins = 30, range = (0,30))
xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
plt.ylabel('Number of synapses')
plt.xlabel('Nearest synapse (µm)')
plt.tight_layout()

# plt.title('different brains_shorthistogram_density'+ '_' +savename)
plt.savefig('different brains_shorthistogram_density' + savename +'.pdf')
plt.show()






ax1 = plt.axes(frameon=False)
colors = ['black','blue']
for i in range(len(allDistances)):
    plt.hist(allDistances[i], density = False, stacked = True, fill = False, color = COLOR[i],  bins = 200,edgecolor = COLOR[i])




# plt.hist(allDistances, density = False, stacked = True, label = COLOR, fill = True, color = COLOR,lw= 0)
xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
plt.ylabel('Number of fibers')
plt.xlabel('  (µm)')
plt.tight_layout()

# plt.title('All synapses_stacked'+ '_' +savename)
plt.savefig('diff brains stacked hist_full.pdf' + savename + '.pdf')
plt.show()


ax1 = plt.axes(frameon=False)
colors = ['black','blue']
for i in range(len(allDistances)):
    plt.hist(allDistances[i], density = False, stacked = True, fill = False, color = COLOR[i], range = (0,30), bins = 30,edgecolor = COLOR[i])


# plt.hist(allDistances[0], density = True, stacked = True, fill = False, color = COLOR[1], range = (0,30), bins = 30,edgecolor = COLOR[1])
# plt.hist(allDistances[1], density = False, stacked = True, fill = False, color = COLOR[0], range = (0,30), bins = 30,edgecolor = COLOR[0])


xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
plt.ylabel('Number of fibers')
plt.xlabel('Nearest fiber (µm)')
plt.tight_layout()

# plt.title('short synapses_stacked'+ '_' +savename)
plt.savefig('diff brains stacked hist_short.pdf' + savename + '.pdf')
plt.show()
######
######
######
######
######
######


ax1 = plt.axes(frameon=False)
plt.hist(allDistances, density = False, color = COLOR,bins = 200)
xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
plt.ylabel('Number of fibers')
plt.xlabel('Nearest fiber (µm)')
plt.tight_layout()

# plt.title('different brains_fullhistogram_raw numbers'+ '_' +savename)
plt.savefig('different brains_fullhistogram_raw numbers' + savename +'.pdf')
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

# plt.title('different brains_shorthistogram_raw numbers'+ '_' +savename)
plt.savefig('different brains_shorthistogram_raw numbers' + savename +'.pdf')
plt.show()






ax1 = plt.axes(frameon=False)
colors = ['black','blue']
for i in range(len(allDistances)):
    plt.hist(allDistances[i], density = False, stacked = True, fill = False, color = COLOR[i],  bins = 200,edgecolor = COLOR[i])




# plt.hist(allDistances, density = False, stacked = True, label = COLOR, fill = True, color = COLOR,lw= 0)
xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
plt.ylabel('Number of fibers')
plt.xlabel('Nearest fiber (µm)')
plt.tight_layout()

# plt.title('All synapses_stacked_raw numbers'+ '_' +savename)
plt.savefig('diff brains stacked hist_full_raw numbers' + '_'+savename+'.pdf')
plt.show()


ax1 = plt.axes(frameon=False)
colors = ['black','blue']
for i in range(len(allDistances)):
    plt.hist(allDistances[i], density = False, stacked = True, fill = False, color = COLOR[i], range = (0,30), bins = 30,edgecolor = COLOR[i])


# plt.hist(allDistances[0], density = True, stacked = True, fill = False, color = COLOR[1], range = (0,30), bins = 30,edgecolor = COLOR[1])
# plt.hist(allDistances[1], density = False, stacked = True, fill = False, color = COLOR[0], range = (0,30), bins = 30,edgecolor = COLOR[0])


xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
plt.ylabel('Number of fibers')
plt.xlabel('Nearest fiber (µm)')
plt.tight_layout()

# plt.title('short synapses_stacked_raw numbers'+ '_' +savename)
plt.savefig('diff brains stacked hist_short_raw numbers' + '_'+savename+'.pdf' + savename + '.pdf')
plt.show()

def under30(a):
    new = []
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] < 30:
                new.append(a[i][j])
    return new
under_30 = under30(allDistances)
# plt.legend(loc = 'upper right')