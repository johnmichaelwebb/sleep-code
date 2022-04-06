#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 20:35:30 2022

@author: jwebb2020
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib
import scipy.stats as st
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

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

def create_empty_matrix(a, label,data):
    new_matrix = []
    for i in range(len(a)):
        if a[i] == label:
            column = extract_column(data, i)
            new_matrix.append(column)
    return new_matrix

