#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:51:55 2022

@author: jwebb2020
"""

import numpy as np
import csv

from LocalResources import dataWrangling as dw
from LocalResources import plottingOptions as po

def extractProjections(a, projections):
    b = []
    for i in range(len(projections)):
        currColumn = dw.extract_column(a, projections[i])
        b.append(currColumn)
    return b

def projectionNeuronIdentity(a):
    #find projection neurons within a single column
    b =[]
    for i in range(len(a)):
        if a[i] > 0:
             b.append(i)
    return b

def projectionNeuronIdentityAll(a):
    ## get the projection neurons for a list of a columns
    b = []
    for i in range(len(a)):
        curr_projection = projectionNeuronIdentity(a[i])
        b.append(curr_projection)
    return b 

def projectionNeuronBinary(a):
    ## turn projections into binary values
    b =[]
    for i in range(len(a)):
        if a[i] > 0:
             b.append(1)
        else:
            b.append(-1)
    return b

def projectionNeuronBinaryAll(a):
    b = []
    for i in range(len(a)):
        curr_projection = projectionNeuronBinary(a[i])
        b.append(curr_projection)
    return b 

def singleAreaConnectivitySingle(currProjection, a):
    ## create heatmap for one region
    total = []
    for i in range(len(currProjection)):
        single_neuron = []
        for j in range(len(a)):
            if currProjection[i] in a[j]:
                single_neuron.append(1)
            else:
                single_neuron.append(0)
        total.append(single_neuron)
    return total

def singleAreaConnectivitySingleAll(a):
    ## generate connectivity for all areas
    b = []
    for i in range(len(a)):
        curr_projection = singleAreaConnectivitySingle(a[i],a)
        b.append(curr_projection)
    return b

def countNumProjections(a):
    ## count the number of projection neurons
    neurons = []
    numAreas = []
    for i in range(len(a[0])):
        curr_neuron = dw.extract_column(a, i)
        neurons.append(curr_neuron)
    for i in range(len(neurons)):
        numAreas.append(np.sum(neurons[i]))
    return numAreas

def extractRawValues(data, projections, neuronIdent):
    ## get the raw sequecning values from the raw data
    curr_brain = []
    for i in range(len(projections)):
        curr_neuron = []
        for j in range((len(neuronIdent))):
            curr_value = data[i][j]
            curr_neuron.append(curr_value)
        curr_neuron = dw.matrix_float(curr_neuron)
        curr_brain.append(curr_neuron)
    np.asarray(curr_brain)
    return curr_brain

def projections_raw_values(a):
    ## get the number of neurons tha project to every projection area
    neurons = []
    for i in range(len(a[0])):
        curr_neuron = dw.extract_column(a,i)
        neurons.append(curr_neuron)
    return neurons

def findPosterior(a):
    ## find posterior projecting neurons
    indexes = []
    for i in range(len(a)):
        before = a[i][0:6]
        after = a[i][6:8]
        if np.sum(before) == 0 and np.sum(after) >0:
            indexes.append(i)
    just_posterior = []
    for i in range(len(indexes)):
        just_posterior.append(a[indexes[i]])
    return just_posterior

def findAnterior(a):
    ## find anterior-projecting neurons
    indexes = []
    for i in range(len(a)):
        before = a[i][0:6]
        after = a[i][6:8]
        if np.sum(before) >0 and np.sum(after) == 0:
            indexes.append(i)
    just_anterior = []
    for i in range(len(indexes)):
        just_anterior.append(a[indexes[i]])
    return just_anterior

def formatting(a):
    ## fix a data formatting issue
    all_neurons = []
    for i in range(len(a[0])):
        curr = dw.extract_column(a, i)
        curr_neuron = []
        for j in range(len(curr)):
            if curr[j] == 1:
                curr_neuron.append(j)
        all_neurons.append(curr_neuron)
    return all_neurons

def to_delete_byIndex(a, delete):
    ## delete items from a list and index of things to delete
    counter = 0
    for i in range(len(delete)):
        index = delete[i] -counter
        a.pop(index)
        counter+=1
    return a

def removeZeroProjections(a):
    ## remove projections with zero
    to_delete = []
    for i in range(len(a)):
        if np.sum(a[i]) ==0:
            to_delete.append(i)
    a = to_delete_byIndex(a, to_delete)
    return a

def getNumProjPerArea_single(a):
    g = []
    for i in range(len(a)):
        if a[i] == 1:
            g.append(1)
    return np.sum(g)

def getNumProjPerArea_all(a):
    ALL = []
    for i in range(len(a[0])):
        b = dw.extract_column(a,i)
        c = getNumProjPerArea_single(b)
        ALL.append(c)
    return ALL

def runOnce(fileName, injection, projection,trialTick, projectionLabels,plotAll = False):
    ## generate heatmap of all projection areas
    with open(fileName, newline='\n' ) as inputfile:
       data = list(csv.reader(inputfile)) 
       data[0][0] = data[0][0][1:len(data[0][0])]   
    CURRglobal_name = fileName[0:len(fileName)-4]
    injection = dw.extract_column(data, injection)
    projections = extractProjections(data, projection)
    projectionNeurons = projectionNeuronIdentityAll(projections)
    projectionNeuronsBinary = projectionNeuronBinaryAll(projections)
    numProjections = countNumProjections(projectionNeuronsBinary)
    allAreas = singleAreaConnectivitySingleAll(projectionNeurons)
    if plotAll == True:
        po.create_heatmap_all(allAreas,projectionLabels,CURRglobal_name,trialTick)
    return numProjections

def runOnce_2(fileName, injection, projection,trialTick, projectionLabels,plotAll = False):
    ## generate heatmap of all projection areas
    with open(fileName, newline='\n' ) as inputfile:
       data = list(csv.reader(inputfile)) 
       data[0][0] = data[0][0][1:len(data[0][0])]   
    CURRglobal_name = fileName[0:len(fileName)-4]
    injection = dw.extract_column(data, injection)
    projections = extractProjections(data, projection)
    projectionNeurons = projectionNeuronIdentityAll(projections)
    raw_values = projections_raw_values(projections)
    allAreas = singleAreaConnectivitySingleAll(projectionNeurons)
    if plotAll == True:
        dw.create_heatmap_all(allAreas,projectionLabels,CURRglobal_name )
    return raw_values, projections