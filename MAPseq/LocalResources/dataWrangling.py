#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:49:15 2022

@author: jwebb2020
"""

plot_dir = 'Results/'

def combine4(a, b,c,d):
    # combine 4 matrixes together
    e = []
    for i in range(len(a)):
        e.append(a[i])
    for i in range(len(b)):
        e.append(b[i])
    for i in range(len(c)):
        e.append(c[i])
    for i in range(len(d)):
        e.append(d[i])
    return e

def extract_column(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(a[i][column])
    new_matrix = matrix_float(new_matrix)
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

def combine_weird_matrix(a,b):
    ## combine 2 matrixes that are originally formatted weirdly 
    c = []
    for i in range(len(a)):
        c.append(a[i])
    for i in range(len(b)):
        c.append(b[i])
    return c

def extract_row(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a[0])):
        new_matrix.append(a[column][i])
    return new_matrix 