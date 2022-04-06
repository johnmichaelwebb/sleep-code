#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 08:25:14 2022

@author: jwebb2020
"""

import numpy as np
import csv
import pandas as pd


data_dir = 'Data/'



def extract_column(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a)):
        new_matrix.append(a[i][column])
        # print(i)
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

def delete_zeros(labels, values):
    values = matrix_float(values)
    delete = []
    new_values = []
    new_labels = []
    for i in range(len(values)):
        if int(values[i]) == 0:
            delete.append(i)
    # print(delete)
    for i in range(len(labels)):
        if i not in delete:
            new_values.append(values[i])
            new_labels.append(labels[i])
    return new_labels, new_values
 

def delete_zeros_all(labels, values, all_values, cat):
    values = matrix_float(values)
    delete = []
    new_values = []
    new_labels = []
    new_all_values = []
    new_cat = []
    for i in range(len(values)):
        if values[i] == 0.0:
            delete.append(i)
    # print(delete)
    for i in range(len(labels)):
        if i not in delete:
            new_values.append(values[i])
            new_labels.append(labels[i])
            new_all_values.append(all_values[i])
            new_cat.append(cat[i])
    return new_labels, new_values, new_all_values, new_cat        
            
def reorderValues(lab, val):
    new_lab = []
    new_val = []
    VALUES = val[:]
    VALUES.sort(reverse=True)
    for i in range(len(VALUES)):
        index = val.index(VALUES[i])
        new_lab.append(lab[index])
        new_val.append(val[index])
    return new_lab, new_val

def reorderValues_all(lab, val, ALL_VALUES):
    new_lab = []
    new_val = []
    new_all_values = []
    VALUES = val[:]
    VALUES.sort(reverse=True)
    for i in range(len(VALUES)):
        index = val.index(VALUES[i])
        new_lab.append(lab[index])
        new_val.append(val[index])
        new_all_values.append(ALL_VALUES[index])
    return new_lab, new_val,new_all_values

def extractColumnByLabel(a, columnName):
    names = extract_row(a, 0)
    index = names.index(columnName)
    column = extract_column(a, index)
    return column  
  
def normalize_values(a):
    new_a = []
    MINN = np.min(a)
    MAXX = np.max(a)
    for i in range(len(a)):
        val = (a[i]-MINN)/(MAXX-MINN)
        new_a.append(val)
    return new_a
    
def extract_singleDataset(a):
    with open(data_dir +a, newline='\n' ) as inputfile:
       data = list(csv.reader(inputfile)) 
       data = data[1:len(data)]
    labels = extract_column(data, 0)
    values = extract_column(data, 1)
    values = matrix_float(values)
    values = normalize_values(values)
    return labels, values

def combineLabels(lab1, lab2,val1, val2):
    new_lab = []
    new_val = []
    for i in range(len(lab1)):
        curr_val = []
        
        if lab1[i] == lab2[i]:
            # print(val2[i])
            curr_val.append(val1[i])
            curr_val.append(val2[i])
        else:
            print('Error: Mismatch between labels')
        new_val.append(curr_val)
    return lab1, new_val

def combineLabels_ugh(lab1, lab2,val1, val2):
    new_lab = []
    new_val = []
    # print(lab1)
    # print(lab2)
    
    for i in range(len(lab1)):
        curr_val = val1[i]
        
        if lab1[i] == lab2[i]:

            # print(val2[i])
            # curr_val.append(val1[i])
            curr_val.append(val2[i])
        else:
            print('Error: Mismatch between labelz')
        new_val.append(curr_val)
    return lab1, new_val


def combineDatasets_all(a):
    all_lab = []
    all_val = []
    for i in range(len(a)):
        if i == 0:
           all_lab, all_val = extract_singleDataset(a[i])
        elif i == 1:
            curr_lab, curr_val = extract_singleDataset(a[i])
            all_lab, all_val = combineLabels(all_lab, curr_lab, all_val, curr_val)
        elif i > 1:
            curr_lab, curr_val = extract_singleDataset(a[i])
            all_lab, all_val = combineLabels_ugh(all_lab, curr_lab, all_val, curr_val)

    return all_val, all_lab

def findCatIndex_single(a, name):
    index = []
    for i in range(len(a)):
        if a[i] == name:
            index.append(i)
    return index
    
def findCat(a):
    cat_index = []
    cate = []
    for i in range(len(a)):
        if a[i] not in cate:
            cate.append(a[i])
    for i in range(len(cate)):
        curr = findCatIndex_single(a, cate[i])
        cat_index.append(curr)
    return cate, cat_index
        

def extract_category(fileName, fileName_2):
 ##find the categories   
    with open(fileName_2, newline='\n' ) as inputfile:
        data = list(csv.reader(inputfile)) 
        data = data[1:len(data)]
    category = extract_column(data, 2)
    data1 = pd.read_excel(fileName)
    filename_all = data1['file']
    # data2 = data1.values.tolist()
    filename_all = filename_all.values.tolist()
    final_val, final_label = combineDatasets_all(filename_all)
    cat, cat_index = findCat(category)
    return final_val, final_label, category