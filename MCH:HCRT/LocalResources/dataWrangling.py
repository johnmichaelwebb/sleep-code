#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:50:26 2022

@author: jwebb2020
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

data_dir = 'Data/'

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

def extractColumnByLabel(a, columnName):
    names = extract_row(a, 0)
    index = names.index(columnName)
    column = extract_column(a, index)
    return column

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

def singMouseID(a, label):
    ID = []
    for i in range(len(a)):
        if a[i] == label:
            ID.append(i)
    return ID

def mouse_indexes(a):
    mouseID = []
    for i in range(len(a)-1):
        if a[i+1] not in mouseID:
            mouseID.append(a[i+1])
    mouseIndex = []
    for i in range(len(mouseID)):
        currIndex = singMouseID(a, mouseID[i])
        mouseIndex.append(currIndex)
    return mouseIndex

def combineMatrix(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i])
    for i in range(len(b)):
        c.append(b[i])  
    return c

def importSingMouse(a,filenames):
    total_distances = []
    for i in range(len(a)):
        index = a[i]
        with open(data_dir+filenames[index], newline='\n' ) as inputfile:
            data = list(csv.reader(inputfile))
        for i in range(len(data)):
            data[i] = data[i][0]
        data = matrix_float(data)
        if len(total_distances) == 0:
            total_distances = data
        else:
            total_distances = combineMatrix(total_distances, data)
    return total_distances
    
def importAllMice(a, filenames):
    distPerMouse = []
    for i in range(len(a)):
        currDist = importSingMouse(a[i], filenames)
        distPerMouse.append(currDist)
    return distPerMouse

def under30(a):
    new = []
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j] < 30:
                new.append(a[i][j])
    return new