#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:59:28 2021

@author: jwebb2020
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib
import matplotlib.ticker as mticker
import pandas as pd

matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')

fileName = 's2.csv'
savename = 'SynTracing'
gross = 'gross_syn.xlsx'

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
    with open(a, newline='\n' ) as inputfile:
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
        
def format_curr_cate_for_graphing(cat_index, x_label, y_label):
    curr_x = []
    curr_y = []
    for i in range(len(cat_index)):
        curr_x.append(x_label[cat_index[i]])
        curr_y.append(y_label[cat_index[i]])
    return curr_x, curr_y

def combine_scatter(x,y):
    ALL_x = []
    ALL_y = []
    for i in range(len(y[0])):
        # print(y[i])
        curr_y = extract_column(y,i)
        # print(len(x))
        print(i)
        # print(len(y[0]))
        for j in range(len(x)):
            ALL_x.append(x[j])
            ALL_y.append(curr_y[j])
    # print(ALL_x)
    # print(ALL_y)
    return ALL_x, ALL_y


with open(fileName, newline='\n' ) as inputfile:
   data = list(csv.reader(inputfile)) 
   data = data[1:len(data)]
   
labels = extract_column(data, 0)
values = extract_column(data, 1)
category = extract_column(data, 2)
df = pd.DataFrame(data)

labels_filtered, values_filtered = delete_zeros(labels, values)



labels_filtered, values_filtered = reorderValues(labels_filtered, values_filtered)
values_filtered = normalize_values(values_filtered)


labels_plot = []
for i in range(len(labels_filtered)):
    labels_plot.append(i)
    
ax1 = plt.axes(frameon=False)

plt.bar(labels_plot, values_filtered, color = 'None', edgecolor = 'black', width = .4)


xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
plt.ylabel('Relative intensity')
plt.xlabel('Projection Region')
ax1.set_yticklabels([])
ax1.get_yaxis().set_ticks([])
plt.xticks(labels_plot, labels_filtered, fontsize = 4, rotation = 90)
ax1.tick_params(axis='both', which='both', length=0)
plt.tight_layout()
plt.savefig(savename + ".pdf")
plt.show()





data1 = pd.read_excel(gross)
filename_all = data1['file']
data2 = data1.values.tolist()

filename_all = filename_all.values.tolist()
final_val, final_label = combineDatasets_all(filename_all)


cat, cat_index = findCat(category)







final_val_mean = []
for i in range(len(final_val)):   
    final_val_mean.append(np.mean(final_val[i]))






final_label, final_val_mean, final_val, final_category = delete_zeros_all(final_label, final_val_mean, final_val, category)


final_category, final_cat_index = findCat(final_category)

# final_label, final_val_mean, final_val = reorderValues_all(final_label, final_val_mean, final_val)


ax1 = plt.axes(frameon=False)


labels_plot_all = []
for i in range(len(final_val_mean)):
    labels_plot_all.append(i)
COLORS = ['navy','deepskyblue','seagreen', 'darkorchid', 'orange','red','darkseagreen', 'y','darkgrey','mediumslateblue']
for i in range(len(final_cat_index)):
    curr_x, curr_y = format_curr_cate_for_graphing(final_cat_index[i],labels_plot_all,final_val_mean)
    plt.bar(curr_x, curr_y, color = COLORS[i],  width = .8)



# plt.bar(labels_plot_all, final_val_mean, color = 'None', edgecolor = 'black', width = .4)

scatter_x, scatter_y = combine_scatter(labels_plot_all, final_val)
plt.scatter(scatter_x, scatter_y,s =2, facecolors = 'none', edgecolors='black',linewidths= .3,zorder =2 )



# for i in range(len(final_val[0])):
#     curr_val = extract_column(final_val, i)
#     plt.scatter(labels_plot_all, curr_val,s =2, facecolors = 'none', edgecolors='black',linewidths= .3)

xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
# plt.ylabel('Relative intensity')
# plt.xlabel('Projection Region')
ax1.set_yticklabels([])
ax1.get_yaxis().set_ticks([])
plt.xticks(labels_plot_all, final_label, fontsize = 4, rotation = 90)
ax1.tick_params(axis='both', which='both', length=0)
ax1.tick_params(axis="x",direction="in", pad=-10)

plt.tight_layout()
plt.savefig(savename + "combinedTEST.pdf")
plt.show()











