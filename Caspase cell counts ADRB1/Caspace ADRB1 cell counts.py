#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:10:31 2020

@author: jwebb2020
"""
############################################
############################################
import matplotlib
from LocalResources import plottingOptions as po
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
##############################################
##############################################

wt_cells = [241, 269,309, 310, 216, 189] ##ADRB1+ cells in WT mice
casp_cells = [146, 83, 93, 63, 69] ##ADRB1+ cells in ablated mice
casp_wake = [63.75,  54.1, 58.85, 59.95,52.55] ## wake % for ablated mice
wt_wake = [47.35,46.8,47.6, 45.15, 44.8, 45.5] ## wake % for WT mice
wt_sleep = [] ##sleep percentage for WT mice
casp_sleep = [] ## sleep percentage for experimental mice
globalFont = 16 # fontsize for figures
global_name = ''


## Calculate sleep % from the wake percent (values add up to 100%)
for i in range(len(casp_wake)):
    q = 100- casp_wake[i]
    casp_sleep.append(q)
for i in range(len(wt_wake)):
    q = 100 - wt_wake[i]
    wt_sleep.append(q)
po.plotDots(wt_cells, 'WT', casp_cells, 'Cre', '$ADRB1^+$ cells', 'caspase cell counts', globalFont, global_name)    
po.plotLinearTrend(casp_sleep, 'Total sleep %', casp_cells, 'ADRB1 + cells', 'ADRB1 cell # versus sleep percent', globalFont, global_name)
total_cells = po.combineMatrix(wt_cells, casp_cells)
total_sleep = po.combineMatrix(wt_sleep, casp_sleep)
po.plotLinearTrend_2color(wt_sleep, 'Total sleep %', wt_cells, 'ADRB1 + cells', 'Total ADRB1 cell # versus sleep percent',casp_sleep, casp_cells, globalFont, global_name)
po.plotLinearTrend_2color(wt_cells, '$ADRB1^+$ cells',wt_sleep , 'Total sleep %', 'Sleep % versus total ADRB1 cell #', casp_cells, casp_sleep, globalFont, global_name)
    
  