#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:42:14 2022

@author: jwebb2020
"""

import numpy as np
import scipy.stats as st

def single_row_cleanup(a):
    ## returns floats for every entry in a single 1 x n matrix
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(float(a[i]))
    return new_matrix

def extract_column(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(a[i][column])
    return new_matrix 

def clean_up_imports(a,zScore):
    ## esentially returns floats for every entry in an n x n matrix 
    ## z-scores the results
    for i in range(len(a)):
        a[i] = single_row_cleanup(a[i])
    if zScore == True:
        for i in range(len(a)):
            a[i] = st.zscore(a[i])
    return a

def avg_animals(a):
    ##creates an average and standard deviation for matrix
    avg_matrix = []
    std_matrix = []
    for i in range(len(a[0])):
        curr_total = []
        for j in range(len(a)):
            curr_total.append(a[j][i])
        avg_matrix.append(np.mean(curr_total))
        std_matrix.append(np.std(curr_total))
    return avg_matrix, std_matrix  

def create_sec(a, laser_freq):
    ## creates seconds based off the sampling frequency and the transition length
    new_matrix = []
    total_sec = int(laser_freq)
    for i in range(len(a)):
        new_matrix.append(i/total_sec)
    return new_matrix

def downsampleSingle(a, secBeforeAndAfter,secPerDownsample):
    ## downsamples photometry transition for a single transition
    new_matrix = []
    numTrials = secBeforeAndAfter*2/secPerDownsample
    numTrialsInt = int(numTrials)
    samplesPerBout = len(a)/numTrials
    for i in range(numTrialsInt):
        start = int(i*samplesPerBout)
        stop = int((i+1)*samplesPerBout)
        currDownsample = a[start:stop]
        currMean = np.mean(currDownsample)
        new_matrix.append(currMean)
    return new_matrix

def downsampleAll(a,secBeforeAndAfter,secPerDownsample):
    ## downsamples the photometry transitions for the transitions
    new_matrix = []
    for i in range(len(a)):
        curr_downsample = downsampleSingle(a[i],secBeforeAndAfter,secPerDownsample)
        new_matrix.append(curr_downsample)
    return new_matrix