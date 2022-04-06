#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:59:28 2021

@author: jwebb2020
"""

import numpy as np
import matplotlib

from LocalResources import dataWrangling as dw
from LocalResources import plottingOptions as po

matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')

data_dir = 'Data/'
gross = 'gross_syn.xlsx'
fileName = data_dir + gross
savename = 'SynTracing'
fileName_2 = data_dir +'s2.csv'


##Import data and process
final_val, final_label, category= dw.extract_category(fileName, fileName_2)
final_val_mean = []
for i in range(len(final_val)):   
    final_val_mean.append(np.mean(final_val[i]))
final_label, final_val_mean, final_val, final_category = dw.delete_zeros_all(final_label, final_val_mean, final_val, category)
final_category, final_cat_index = dw.findCat(final_category)

##plot the data
po.plotAllAreas(final_val_mean, final_cat_index,final_val, final_label,savename)







