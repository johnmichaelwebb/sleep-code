#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:32:47 2021

@author: jwebb2020
"""

import pandas as pd
import matplotlib
from LocalResources import dataWrangling as dw
from LocalResources import plottingOptions as po

matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
##############################################
##############################################
##############################################
##############################################
data_dir = 'Data/'
name_score = 'ctb results.xlsx' ## user scored epochs
fileName = data_dir + name_score
global_name = 'CtB all'
globalFont = 20



##initialize data
data1 = pd.read_excel(fileName)
name_all = data1['name']
overlap_all = data1['overlap']
lh_all = data1['lh']
pz_all = data1['pz']
overlap_all = overlap_all.values.tolist()
lh_all = lh_all.values.tolist()
pz_all = pz_all.values.tolist()



overlap = dw.get_tot_per_brain(overlap_all)
lh = dw.get_tot_per_brain(lh_all)
pz = dw.get_tot_per_brain(pz_all)
overlap_perc, lh_perc, pz_perc = dw.create_perc(overlap, lh, pz)
po.plot_barplot_3groups(lh_perc, 'LH', pz_perc, 'PZ', overlap_perc, 'LH + PZ', '% signal', 'CtB all', globalFont, global_name)