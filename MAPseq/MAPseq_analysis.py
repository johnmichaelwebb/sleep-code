# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

############################################
import numpy as np
import matplotlib
from LocalResources import dataWrangling as dw
from LocalResources import plottingOptions as po
from LocalResources import MAPseqProcessing as mp

matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
##############################################
data_dir = 'Data/'
plotAll = False
negColumn = 0
projectionLabels = ['VLPO', 'BF', 'PVT','LH','DMH','VTA', 'vlPAG','PZ']
trialTick = 100
numProjectionsTotal = []


mouse1filename = [data_dir + 'Brain3.csv'] ## 1
injectionColumn1 = [57] ## 1
projectionColumns1 = [[1,2,3,4,5,6,7,8]] ## 1
mouse2filename = [data_dir + 'Brain4.csv'] ## 2
injectionColumn2 = [58] ## 2
projectionColumns2 = [[10,11,12,13,14,15,16,17]] ## 2
mouse3filename = [data_dir+'Brain5.csv'] ## 3
injectionColumn3 = [59] ## 3
projectionColumns3 = [[19,20,21,22,23,24,25,26]] ## 3
mouse4filename = [data_dir+'Brain7.csv'] ## 4
injectionColumn4 = [60] ## 4
projectionColumns4 = [[28,29,30,31,32,33,34,35]] ## 4
global_name = mouse1filename[0:len(mouse1filename)-4]
global_name = 'all PCA'
mouseFileNameAll = dw.combine4(mouse1filename, mouse2filename, mouse3filename, mouse4filename)
injectionColumnAll = dw.combine4(injectionColumn1,injectionColumn2, injectionColumn3, injectionColumn4)
projectionColumnsAll = dw.combine4(projectionColumns1, projectionColumns2, projectionColumns3, projectionColumns4)


for i in range(len(mouseFileNameAll)):
    currNumProjections = mp.runOnce(mouseFileNameAll[i],injectionColumnAll[i],projectionColumnsAll[i],trialTick, projectionLabels)
    numProjectionsTotal.append(currNumProjections)
first_raw_values = np.asarray(projectionLabels[:])
for i in range(len(mouseFileNameAll)):
    aaa_furtherAnalysis, allareasingle = mp.runOnce_2(mouseFileNameAll[i],injectionColumnAll[i],projectionColumnsAll[i],trialTick, projectionLabels)
    if i == 0:
        tot_raw_values = aaa_furtherAnalysis
        all_area_all = allareasingle
    else:
        tot_raw_values = dw.combine_weird_matrix(tot_raw_values,aaa_furtherAnalysis)
        all_area_all = dw.combine_weird_matrix(all_area_all,allareasingle)

tot_raw_values_binary =mp.projectionNeuronBinaryAll(tot_raw_values)
binary4PCA = tot_raw_values_binary
posterior = mp.findPosterior(binary4PCA)
anterior = mp.findAnterior(binary4PCA)
projectionNeurons = mp.projectionNeuronIdentityAll(tot_raw_values)
projectionNeuronsBinary = mp.projectionNeuronBinaryAll(tot_raw_values)
projectionNeurons = mp.formatting(projectionNeuronsBinary)
allAreas = mp.singleAreaConnectivitySingleAll(projectionNeurons)
po.create_heatmap_all(allAreas,projectionLabels,'all',trialTick)

po.plot_PCA(binary4PCA)

tot_raw_values =  mp.removeZeroProjections(tot_raw_values)      
tot_bin_values = mp.projectionNeuronBinaryAll(tot_raw_values)   
TOTAl_Projecting_Neurons_to_area = mp.getNumProjPerArea_all(tot_raw_values_binary)
TOTAl_Projecting_Neurons_to_area.sort(reverse = True)
projectionLabels_sorted = ['VTA', 'vlPAG', 'PVT','PZ','LH','DMH', 'VLPO','BF']
projectionLabels_forplot = [1,2,3,4,5,6,7,8]


po.plot_numTotalProjectingNeurons(projectionLabels_forplot,TOTAl_Projecting_Neurons_to_area,projectionLabels_sorted)



