#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:42:14 2022

@author: jwebb2020
"""

import numpy as np


hours = 24 ## the number of hours in a day


def create_hrs(hrs = hours):
    ##creates the number of hours i a day
    matrix = []
    for i in range(hrs):
        matrix.append(i+1)
    return matrix

def extract_column(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(a[i][column])
    return new_matrix 

def matrix_float(a):
    ## converts a matrix of strings to floats
    ## often necessary when importing the data csv files into python
    new_matrix = []
    for i in range(len(a)):
        try:
            new_matrix.append((float(a[i])))
        except ValueError:
            print(i)
            print("matrixFloat")
    return new_matrix

def extract_row(a, column):
    # take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a[0])):
        new_matrix.append(a[column][i])
    return new_matrix 

def create_empty_matrix(a, label,data):
    ## creates an empty matrix of length a 
    ## puts data into it and attaches label to it
    new_matrix = []
    for i in range(len(a)):
        if a[i] == label:
            column = extract_column(data, i)
            new_matrix.append(column)
    return new_matrix

def avg_animals(a):
    ## find the avg and std for a matrix of length n animals 
    avg_matrix = []
    std_matrix = []
    for i in range(len(a[0])):
        curr_total = []
        for j in range(len(a)):
            curr_total.append(a[j][i])
        avg_matrix.append(np.mean(curr_total))
        std_matrix.append(np.std(curr_total))
    return avg_matrix, std_matrix  

def avg_animals_debt(a):
    ## calculates the average and std of sleep debt given totals for every group
    avg_matrix = [0]
    std_matrix = [0]
    for i in range(len(a[0])):
        curr_total = []
        for j in range(len(a)):
            curr_total.append(a[j][i])
        avg_matrix.append(np.mean(curr_total))
        std_matrix.append(np.std(curr_total))
    return avg_matrix, std_matrix 


def extract_trials(a,beforeAndAfter, num_trials, bout_per_min):
    matrix = [0 for i in range(num_trials +1)]
    bout_per_trial = int(beforeAndAfter*2*bout_per_min +2)
    stim_start = stimStartMatrix()
    matName = a[0:2]
    a = a[2:len(a)]
    matrix[0] = matName
    for i in range(num_trials):
        sub_matrix = []
        for j in range(bout_per_trial):
            pointer = beforeAndAfter * bout_per_min 
            index = stim_start[i] - pointer +j
            sub_matrix.append(a[index])
        matrix[i +1] = sub_matrix
    return matrix

def extract_all_trials(a,beforeAndAfter, num_trials, bout_per_min):
    matrix = [0 for i in range((len(a)))]
    for i in range(len(a)):
        matrix[i] = extract_trials(a[i],beforeAndAfter, num_trials, bout_per_min)

def scrub_formating(a,new_data):
    stop = len(a[0])-1
    for i in range(len(a)):
        for j in range(len(a[0])):
            
            a[i][j] = a[i][j][2:len(a[i][j])-1]
    for i in range(len(new_data)):
        new_data[i][stop] = new_data[i][stop][0:len(new_data[i][stop])-1]
    a[0][0] = a[0][0][1:len(a[0][0])] 
    a[0][stop] = a[0][stop][0:len(a[0][stop])-1]     
    return a

def just_scores(a):
    for i in range(len(a)):
            a[i] = a[i][2:len(a[i])]
    return a

def stimStartMatrix(bout_per_min, min_before1st, num_trials, bout_length, stim_freq):
    matrix = []
    start = min_before1st*bout_per_min
    for i in range(num_trials):
        index = int(start + i*stim_freq*60/bout_length)
        matrix.append(index)
    return matrix
