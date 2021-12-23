# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

############################################
import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.lines import Line2D
import seaborn
from sklearn.decomposition import PCA
import matplotlib
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Arial')
##############################################

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

def extractProjections(a, projections):
    b = []
    for i in range(len(projections)):
        currColumn = extract_column(a, projections[i])
        b.append(currColumn)
    return b

def projectionNeuronIdentity(a):
    #find projection neurons within a single column
    b =[]
    for i in range(len(a)):
        if a[i] > 0:
             b.append(i)
    return b

def projectionNeuronIdentityAll(a):
    ## get the projection neurons for a list of a columns
    b = []
    for i in range(len(a)):
        curr_projection = projectionNeuronIdentity(a[i])
        b.append(curr_projection)
    return b 

def projectionNeuronBinary(a):
    ## turn projections into binary values
    b =[]
    for i in range(len(a)):
        if a[i] > 0:
             b.append(1)
        else:
            b.append(-1)
    return b

def projectionNeuronBinaryAll(a):
    
    b = []
    for i in range(len(a)):
        curr_projection = projectionNeuronBinary(a[i])
        b.append(curr_projection)
    return b 

def create_heatmap_single(a, brainAreas, name,MOUSENAME): 
    #create heatmap for one area
    x_labels = brainAreas
    y_labels = create_y_axis(a)
    ax = seaborn.heatmap(a,xticklabels = x_labels, yticklabels = y_labels, linecolor = 'gray',linewidths = 0, cbar =False,cmap='YlGnBu')
    ax.tick_params(left=False, bottom=False)    
    plt.ylabel('Neuron #')
    name = str(name)
    # plt.title(name +'_' + MOUSENAME)
    plt.savefig(name +'_' + MOUSENAME +'_.pdf')
    plt.show()
    return

def create_heatmap_all(a, brainAreas, MOUSENAME):
    ## create heatmap for all areas
    for i in range(len(a)):
        create_heatmap_single(a[i], brainAreas, brainAreas[i],MOUSENAME)
    return
def singleAreaConnectivitySingleAll(a):
    ## generate connectivity for all areas
    b = []
    for i in range(len(a)):
        curr_projection = singleAreaConnectivitySingle(a[i],a)
        b.append(curr_projection)
    return b

def create_y_axis(a):
    ## create y-axis for heatmap
    new_matrix = []
    num_labels = int(len(a)/trialTick)
    curr_trial = []
    include = []
    for i in range(num_labels):
        include.append((i+1)*trialTick)
    for i in range(len(a)):
        curr_trial.append(i+1)
    counter = 0
    for i in range(len(curr_trial)):
        if curr_trial[i] in include:
            new_matrix.append(include[counter])
            counter+=1
        else:
            new_matrix.append('')
    return new_matrix

def singleAreaConnectivitySingle(currProjection, a):
    ## create heatmap for one region
    total = []
    for i in range(len(currProjection)):
        single_neuron = []
        for j in range(len(a)):
            if currProjection[i] in a[j]:
                single_neuron.append(1)
            else:
                single_neuron.append(0)
        total.append(single_neuron)
    return total

def countNumProjections(a):
    ## count the number of projection neurons
    neurons = []
    numAreas = []
    for i in range(len(a[0])):
        curr_neuron = extract_column(a, i)
        neurons.append(curr_neuron)
    for i in range(len(neurons)):
        numAreas.append(np.sum(neurons[i]))
    return numAreas

def runOnce(fileName, injection, projection):
    ## generate heatmap of all projection areas
    with open(fileName, newline='\n' ) as inputfile:
       data = list(csv.reader(inputfile)) 
       data[0][0] = data[0][0][1:len(data[0][0])]   
    CURRglobal_name = fileName[0:len(fileName)-4]
    injection = extract_column(data, injection)
    projections = extractProjections(data, projection)
    projectionNeurons = projectionNeuronIdentityAll(projections)
    projectionNeuronsBinary = projectionNeuronBinaryAll(projections)
    numProjections = countNumProjections(projectionNeuronsBinary)
    allAreas = singleAreaConnectivitySingleAll(projectionNeurons)
    if plotAll == True:
        create_heatmap_all(allAreas,projectionLabels,CURRglobal_name )
    return numProjections

def extractRawValues(data, projections, neuronIdent):
    ## get the raw sequecning values from the raw data
    curr_brain = []
    for i in range(len(projections)):
        curr_neuron = []
        for j in range((len(neuronIdent))):
            curr_value = data[i][j]
            curr_neuron.append(curr_value)
        curr_neuron = matrix_float(curr_neuron)
        curr_brain.append(curr_neuron)
    np.asarray(curr_brain)
    return curr_brain

def combine_weird_matrix(a,b):
    ## combine 2 matrixes that are originally formatted weirdly 
    c = []
    for i in range(len(a)):
        c.append(a[i])
    for i in range(len(b)):
        c.append(b[i])
    return c

def projections_raw_values(a):
    ## get the number of neurons tha project to every projection area
    neurons = []
    for i in range(len(a[0])):
        curr_neuron = extract_column(a,i)
        neurons.append(curr_neuron)
    return neurons

def runOnce_2(fileName, injection, projection):
    ## generate heatmap of all projection areas
    with open(fileName, newline='\n' ) as inputfile:
       data = list(csv.reader(inputfile)) 
       data[0][0] = data[0][0][1:len(data[0][0])]   
    CURRglobal_name = fileName[0:len(fileName)-4]
    injection = extract_column(data, injection)
    projections = extractProjections(data, projection)
    projectionNeurons = projectionNeuronIdentityAll(projections)
    raw_values = projections_raw_values(projections)
    allAreas = singleAreaConnectivitySingleAll(projectionNeurons)
    if plotAll == True:
        create_heatmap_all(allAreas,projectionLabels,CURRglobal_name )
    return raw_values, projections

def findPosterior(a):
    ## find posterior projecting neurons
    indexes = []
    for i in range(len(a)):
        before = a[i][0:6]
        after = a[i][6:8]
        if np.sum(before) == 0 and np.sum(after) >0:
            indexes.append(i)
    just_posterior = []
    for i in range(len(indexes)):
        just_posterior.append(a[indexes[i]])
    return just_posterior

def findAnterior(a):
    ## find anterior-projecting neurons
    indexes = []
    for i in range(len(a)):
        before = a[i][0:6]
        after = a[i][6:8]
        if np.sum(before) >0 and np.sum(after) == 0:
            indexes.append(i)
    just_anterior = []
    for i in range(len(indexes)):
        just_anterior.append(a[indexes[i]])
    return just_anterior

def formatting(a):
    ## fix a data formatting issue
    all_neurons = []
    for i in range(len(a[0])):
        curr = extract_column(a, i)
        curr_neuron = []
        for j in range(len(curr)):
            if curr[j] == 1:
                curr_neuron.append(j)
        all_neurons.append(curr_neuron)
    return all_neurons

def extract_row(a, column):
    #take a single column from a more complex matrix
    ## returns that column
    new_matrix = []
    for i in range(len(a[0])):
        new_matrix.append(a[column][i])
    return new_matrix 

def to_delete_byIndex(a, delete):
    ## delete items from a list and index of things to delete
    counter = 0
    for i in range(len(delete)):
        index = delete[i] -counter
        a.pop(index)
        counter+=1
    return a

def removeZeroProjections(a):
    ## remove projections with zero
    to_delete = []
    for i in range(len(a)):
        if np.sum(a[i]) ==0:
            to_delete.append(i)
    a = to_delete_byIndex(a, to_delete)
    return a

def getNumProjPerArea_single(a):
    g = []
    for i in range(len(a)):
        if a[i] == 1:
            g.append(1)
    return np.sum(g)

def getNumProjPerArea_all(a):
    ALL = []
    for i in range(len(a[0])):
        b = extract_column(a,i)
        c = getNumProjPerArea_single(b)
        ALL.append(c)
    return ALL



mouse1filename = ['Brain3.csv'] ## 1
injectionColumn1 = [57] ## 1
projectionColumns1 = [[1,2,3,4,5,6,7,8]] ## 1
mouse2filename = ['Brain4.csv'] ## 2
injectionColumn2 = [58] ## 2
projectionColumns2 = [[10,11,12,13,14,15,16,17]] ## 2
mouse3filename = ['Brain5.csv'] ## 3
injectionColumn3 = [59] ## 3
projectionColumns3 = [[19,20,21,22,23,24,25,26]] ## 3
mouse4filename = ['Brain7.csv'] ## 4
injectionColumn4 = [60] ## 4
projectionColumns4 = [[28,29,30,31,32,33,34,35]] ## 4
global_name = mouse1filename[0:len(mouse1filename)-4]
global_name = 'all PCA'
mouseFileNameAll = combine4(mouse1filename, mouse2filename, mouse3filename, mouse4filename)
injectionColumnAll = combine4(injectionColumn1,injectionColumn2, injectionColumn3, injectionColumn4)
projectionColumnsAll = combine4(projectionColumns1, projectionColumns2, projectionColumns3, projectionColumns4)
plotAll = False
negColumn = 0
projectionLabels = ['VLPO', 'BF', 'PVT','LH','DMH','VTA', 'vlPAG','PZ']
trialTick = 100
numProjectionsTotal = []
for i in range(len(mouseFileNameAll)):
    currNumProjections = runOnce(mouseFileNameAll[i],injectionColumnAll[i],projectionColumnsAll[i])
    numProjectionsTotal.append(currNumProjections)
first_raw_values = np.asarray(projectionLabels[:])
for i in range(len(mouseFileNameAll)):
    aaa_furtherAnalysis, allareasingle = runOnce_2(mouseFileNameAll[i],injectionColumnAll[i],projectionColumnsAll[i])
    if i == 0:
        tot_raw_values = aaa_furtherAnalysis
        all_area_all = allareasingle
    else:
        tot_raw_values = combine_weird_matrix(tot_raw_values,aaa_furtherAnalysis)
        all_area_all = combine_weird_matrix(all_area_all,allareasingle)

tot_raw_values_binary = projectionNeuronBinaryAll(tot_raw_values)
binary4PCA = tot_raw_values_binary
posterior = findPosterior(binary4PCA)
anterior = findAnterior(binary4PCA)
projectionNeurons = projectionNeuronIdentityAll(tot_raw_values)
projectionNeuronsBinary = projectionNeuronBinaryAll(tot_raw_values)
projectionNeurons = formatting(projectionNeuronsBinary)
allAreas = singleAreaConnectivitySingleAll(projectionNeurons)
create_heatmap_all(allAreas,projectionLabels,'all' )
ax1 = plt.axes(frameon=False)
pca = PCA(n_components=2)
h  = pca.fit(binary4PCA)
X_pca = pca.transform(binary4PCA)
pca = PCA(2)  # project from many to 2 dimensions
projected = pca.fit_transform(binary4PCA)
test = projected[:, 0]
plt.scatter(projected[:, 0], projected[:, 1], color = 'black',
            edgecolor='none', alpha=0.5)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=4))
ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=4))  
plt.tight_layout()
plt.savefig( '_PCAall_.pdf')
plt.show()

tot_raw_values =  removeZeroProjections(tot_raw_values)      
tot_bin_values = projectionNeuronBinaryAll(tot_raw_values)   

TOTAl_Projecting_Neurons_to_area = getNumProjPerArea_all(tot_raw_values_binary)
TOTAl_Projecting_Neurons_to_area.sort(reverse = True)
projectionLabels_sorted = ['VTA', 'vlPAG', 'PVT','PZ','LH','DMH', 'VLPO','BF']
projectionLabels_forplot = [1,2,3,4,5,6,7,8]
ax1 = plt.axes(frameon=False)
plt.bar(projectionLabels_forplot, TOTAl_Projecting_Neurons_to_area, color = 'black')

xmin, xmax = ax1.get_xaxis().get_view_interval()
ymin, ymax = ax1.get_yaxis().get_view_interval()
ax1.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
ax1.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2)) 
plt.ylabel('# Neurons')
plt.xticks(projectionLabels_forplot, projectionLabels_sorted, fontsize = 10)

# plt.xlabel('Firing rate (Hz)')
plt.tight_layout()
plt.savefig("number projecting neurons total.pdf")
plt.show()

