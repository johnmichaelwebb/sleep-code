#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 09:37:42 2021

@author: jwebb2020
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

from LocalResources import dataWrangling as dw
from LocalResources import plottingOptions as po
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
matplotlib.rcParams.update({'font.size': 22})


data_dir = 'Data/'
gross_name = 'mch 4 mice gross instructions.csv'
fileName = data_dir+gross_name
savename = gross_name[0:len(gross_name)-4]








with open(fileName, newline='\n' ) as inputfile:
    data = list(csv.reader(inputfile)) 
    data[0][0] = data[0][0][1:len(data[0][0])]





filename = dw.extractColumnByLabel(data, 'filename')
mouse_column = dw.extractColumnByLabel(data, 'mouse')
mouse_index = dw.mouse_indexes(mouse_column)
allDistances = dw.importAllMice(mouse_index, filename)
test = po.histAll(allDistances, 'all dist', savename)

if len(allDistances) == 1:
    COLOR = 'black'
elif len(allDistances) == 2:
    COLOR = ['black', 'blue']
elif len(allDistances) == 3:
    COLOR = ['black', 'blue','green']
elif len(allDistances) == 4:
    COLOR = ['black', 'blue','green','orange']





po.plot_histogram(allDistances,COLOR, savename)
po.plot_histogram_stacked(allDistances,COLOR, savename)
po.plot_histogram_rawNum(allDistances,COLOR, savename)
po.plot_histogram_rawNum_stacked(allDistances,COLOR, savename)







under_30 = dw.under30(allDistances)
