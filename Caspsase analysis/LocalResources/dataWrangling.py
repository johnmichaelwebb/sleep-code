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