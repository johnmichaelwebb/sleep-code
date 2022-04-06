#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:52:09 2020

@author: jwebb2020
"""
import numpy as np
import csv
import matplotlib
from LocalResources import dataWrangling as dw
from LocalResources import plottingOptions as po

data_dir = 'Data/'
plot_dir = 'Results/'

matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')


fileName = 'CRACM results_connected.csv' ## raw datas
saveName = 'CRACM connected' ## attached to saved files
trialTick = 5
projectionLabels = ['GAD1', 'LHX6','MCH', 'HCRT'] ## cell labels

filePath = data_dir + fileName


with open(filePath, newline='\n' ) as inputfile:
   data = list(csv.reader(inputfile)) 
GAD = dw.extract_column(data[1:len(data)], 1)
GAD = dw.matrix_float(GAD)
GAD_perc = np.sum(GAD)/len(GAD)*100
LHX6 = dw.extract_column(data[1:len(data)], 2)
LHX6 = dw.matrix_float(LHX6)
LHX6_perc = np.sum(LHX6)/len(LHX6)*100
MCH = dw.extract_column(data[1:len(data)], 3)
MCH = dw.matrix_float(MCH)
MCH_perc = np.sum(MCH)/len(MCH)*100
orexin = dw.extract_column(data[1:len(data)], 4)
orexin = dw.matrix_float(orexin)
heatmap_data = dw.create_heatmap_matrix(data)
heatmap_data = dw.sortByColumn(heatmap_data, 0)
heatmap_data = dw.sortByColumn(heatmap_data, 2)
heatmap_data = dw.sortByColumn(heatmap_data, 1)
po.create_heatmap_single(heatmap_data, projectionLabels, 'heatmap', trialTick, saveName)

po.piePlot(GAD, 'GAD',saveName)
po.piePlot(MCH, 'MCH',saveName)
po.piePlot(LHX6, 'LHX6',saveName)
po.piePlot(orexin, 'HCRT',saveName)


GAD_perc = dw.perc(GAD)
LHX6_perc = dw.perc(LHX6)
MCH_perc = dw.perc(MCH)
orexin_perc = dw.perc(orexin)
perc_totals = [GAD_perc, MCH_perc, LHX6_perc,  orexin_perc]
perc_names_1 = ['GAD','MCH','LHX6','HCRT']
perc_names  = [1,2,3,4]
po.CRACM_total_plot(perc_totals, perc_names,perc_names_1)


