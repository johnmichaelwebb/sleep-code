#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:03:33 2020

@author: jwebb2020
"""


import numpy as np
import csv
import matplotlib.pyplot as plt

PATH = 'Data for sample/'
globalName = 'Test'
grossFilename = 'sample instructions.csv'

with open(PATH + grossFilename, newline='\n' ) as inputfile:
    data = list(csv.reader(inputfile))  
    # data[0][0] = data[0][0][0:len(data[0][0])]

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

def extract_row(a, row):
    #take a single column from a more complex matrix
    ## returns that column
    
    new_matrix = []
    for i in range(len(a[0])):
        new_matrix.append(a[row][i])
    return new_matrix

def inAnyCell(a, point):
    YesOrNo = False
    if len(a) != 0:
        for i in range(len(a)):
            if point in a[i]:
                YesOrNo = True
    return YesOrNo

def extractOutlineBoundary(a, point):
    curr_point = point
    boundary = [point]
    finished = False
    while finished == False:
        xUp = a[curr_point[0]+1][curr_point[1]]
        xDown = a[curr_point[0]-1][curr_point[1]]
        yUp = a[curr_point[0]][curr_point[1]+1]
        yDown = a[curr_point[0]][curr_point[1]-1]
        if xUp == 255:
            boundary.append(xUp)
            curr_point = xUp
        elif xDown == 255:
            boundary.append(xDown)
            curr_point = xDown            
        elif yUp == 255:
            boundary.append(yUp)
            curr_point = yUp          
        elif yDown == 255:
            boundary.append(yDown)
            curr_point = yDown 
        if curr_point in boundary:
            finished = True
    return boundary

def extractOutlineBoundaryTest(a, point, boundary):
    curr_point = point
    if len(boundary) == 0:
        boundary = []
        boundary.append(point)
    finished = False
    while finished == False:
        xUp = a[curr_point[1]][curr_point[0]+1]
        xUpPoint = [ curr_point[0] +1,curr_point[1]]       
        xDown = a[curr_point[1]][curr_point[0]-1]
        xDownPoint = [ curr_point[0]-1,curr_point[1]]
        yUp = a[curr_point[1]+1][curr_point[0]]
        yUpPoint = [ curr_point[0],curr_point[1]+1]
        yDown = a[curr_point[1]-1][curr_point[0]]
        yDownPoint = [ curr_point[0],curr_point[1]-11]
        
        if xUp == 255 and xUpPoint not in boundary:
            appendPoint = []
            appendPoint.append(curr_point[0]+1)
            appendPoint.append(curr_point[1])
            boundary.append(appendPoint)
            curr_point = appendPoint
        
        elif xDown == 255 and xDownPoint not in boundary:
            appendPoint = []
            appendPoint.append(curr_point[0]-1)           
            appendPoint.append(curr_point[1])
            boundary.append(appendPoint)
            curr_point = appendPoint            
       
        elif yUp == 255 and yUpPoint not in boundary:
            appendPoint = []
            appendPoint.append(curr_point[0])              
            appendPoint.append(curr_point[1]+1)
            boundary.append(appendPoint)
            curr_point = appendPoint          
        elif yDown == 255 and yDownPoint not in boundary:
            appendPoint = []
            appendPoint.append(curr_point[0])               
            appendPoint.append(curr_point[1]-1)
            boundary.append(appendPoint)
            curr_point = appendPoint
        else:
            return boundary
            finished == True
    return boundary    

def findCells(a):
    cells = []
    for i in range(len(a[0])):
        for j in range(len(a)):
            if a[j][i] == 255:
                curr_point = [i,j]                
                inCells = inAnyCell(cells, curr_point)
                if inCells == False:
                    curr_point = [i, j]
                    curr_cell = extractOutlineBoundaryTest(a, curr_point, [])
                    curr_cell = extractOutlineBoundaryTest(a, curr_cell[0], curr_cell)
                    cells.append(curr_cell)
    return cells

def distSingleCell(cell, synapses):
    min_dist = 100000000
    for i in range(len(cell)):
        a = np.array(cell[i])
        for j in range(len(synapses)):
            b = np.array(synapses[j])
            curr_dist = np.linalg.norm(a-b)
            if curr_dist < min_dist:
                min_dist = curr_dist
    return min_dist

def distAllCells(cells, synapses):
    nearestSyn = []
    for i in range(len(cells)):
        curr_dist = distSingleCell(cells[i], synapses)
        nearestSyn.append(curr_dist)
    return nearestSyn

def extractSynapses(a):
    x = extract_column(a, 2)
    x = matrix_float(x)
    y = extract_column(a, 3)
    y = matrix_float(y)
    points = []
    for i in range(len(x)):
        curr_point = []
        curr_point.append(x[i])
        curr_point.append(y[i])
        points.append(curr_point)
    return points

def plotAllCells(a, Syn, name):
    for i in range(len(a)):
        curr_x = extract_column(a[i], 0)
        curr_y = extract_column(a[i], 1)
        plt.plot(curr_x,curr_y)
    for i in range(len(Syn)):
        curr_x = extract_column(Syn, 0)
        curr_y = extract_column(Syn, 1)       
        plt.scatter(curr_x, curr_y, s= 100, c = 'black')
    plt.savefig(name + '_ALLcells_'+globalName + '.pdf')
    plt.show()
    return

def plotHist(hist, name):
    plt.hist(hist)
    plt.savefig(name + '_'+globalName + '.pdf')
    plt.show()
    return

def runOnce(cellFile, SynapseFile, path = PATH):
    with open(path + SynapseFile, newline='\n' ) as inputfile:
        data = list(csv.reader(inputfile))     
    savename = cellFile[0:len(cellFile)-4]
    cell_binary = np.genfromtxt(fname=path +cellFile)
    Cells = findCells(cell_binary)
    Synapses = extractSynapses(data)
    NearestDistances = distAllCells(Cells, Synapses)
    np.savetxt(savename+'_' + globalName + '.csv', NearestDistances, delimiter=",") 
    plotHist(NearestDistances, savename)
    plotAllCells(Cells, Synapses, savename)
    return

cells = extract_column(data, 0)
synapses = extract_column(data, 1)
cells[0] = cells[0][1:len(cells[0])]

for i in range(len(cells)):
    runOnce(cells[i], synapses[i])