#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:48:27 2022

@author: jwebb2020
"""

import numpy as np



def extract_column(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(a[i][column])
    return new_matrix    

def matrix_float(a):
    #converts a matrix of strings to floats
    ##often necessary when importing the data csv files into python
    new_matrix = []
    for i in range(len(a)):
        try:
            new_matrix.append((float(a[i])))
        except ValueError:
            print(i)
            print("matrixFloat")
    return new_matrix 

def extract_row(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a[0])):
        new_matrix.append(a[column][i])
    return new_matrix 





def create_heatmap_matrix(a):
    ## creates the matrix for a heatmap
    a = a[1:len(a)-2]
    for i in range(len(a)):
        a[i] = a[i][1:len(a[i])-1]
        a[i] = matrix_float(a[i])
    return a

def switch_columns(a, labels, first, second):
    ## switch the order of the columns and labels
    one = extract_column(a, first)
    two = extract_column(a, second)
    label_one = labels[first]
    label_two = labels[second]
    for i in range(len(a)):
        a[i][first] = two[i]
        a[i][second] = one[i]
    labels[first] = label_two
    labels[second] = label_one
    return a, labels

def sortByColumn(a, column):
    ## makes sure that values equaling 1 rise to the top of the column
    to_sort = extract_column(a, column)
    pos = []
    neg =[]
    for i in range(len(to_sort)):
        if to_sort[i] == 1:
            pos.append(i)
        else:
            neg.append(i)
    new = []
    for i in range(len(pos)):
        curr = extract_row(a, pos[i])
        new.append(curr)
    for i in range(len(neg)):
        curr = extract_row(a, neg[i])
        new.append(curr)
    return new

def perc(a):
    ## gets the percent for a list of values
    a_perc = np.sum(a)/len(a)*100
    return a_perc