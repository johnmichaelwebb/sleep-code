#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:24:51 2022

@author: jwebb2020
"""

def extractColumnByLabel(a, columnName):
    ## extracts a column by the initial label in the column
    names = extract_row(a, 0)
    index = names.index(columnName)
    column = extract_column(a, index)
    return column

def extract_row(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a[0])):
        new_matrix.append(a[column][i])
    return new_matrix 

def extract_column(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(a[i][column])
    return new_matrix 

def combine_3(a, b, c):
    #combines 3 matrixes togther 
    d = []
    for i in range(len(a)):
        d.append(a[i])
    for i in range(len(b)):
        d.append(b[i])
    for i in range(len(c)):
        d.append(c[i])
    return d  