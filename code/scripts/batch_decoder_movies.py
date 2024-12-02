#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 07:53:57 2022

@author: jihopark
"""

"""
This script runs SVM-decoder analysis on each movies session and saves the output as individual csv files 

"""

import os
import numpy as np
import pandas as pd
from functions import extract

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

#%% Define paths

# Home directory where the repository is cloned 
# Make sure to change the information accordingly
homepath = os.path.join('C:\\','Users','jihop','Documents','GitHub','Park_et_al_2024','')
# Directory containing data files
datapath = os.path.join(homepath,'sample-data','')
# Directory to save output files
savepath = os.path.join(homepath,'results','sample-output','')
# Directory to save plots 
plotpath = os.path.join(homepath,'results','sample-plots','')

#%% Load the data from MATLAB files in the datapath

# Load all sessions into a single dataframe (this df contains different kinds of visual stimuli sessions)
dfMaster = extract.load_data(datapath) 

# Categorize the sessions into control and experimental groups
dfMaster['Group'] = dfMaster['animalID'].apply(extract.get_group)

# Make a copy of the dfMaster as df 
df = dfMaster.copy()

#%% Extract sessions by visual stimuli (i.e.: spo, grat, mov)

# Create a new df containing only natural movies sessions
dfMov = extract.extract_mov(df)
dfMov = extract.get_dff(dfMov)

#%% Define fixed paramters 

freqNeuro = 16
nRepeats = 32 # each series of movies is repeated 32 times 
nStim = 7
tOn = 14
tOff = 3
tMovOn = 2
tRepeat = tOn + tOff 

nTrials = nStim * nRepeats

t = 544

labels = ['1','2','3','4','5','6','7'] * nRepeats

#%% SVM-decoder 

param_grid = {'C': [10**-3,10**-2,10**-1,1,10**1,10**2,10**3]}

test_size = 0.33

# Define function to fit SVM model and compute AUC score
def fit_svm(X_train, y_train, X_test, y_test):
    # Optimizing 
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=0)
    grid.fit(X_train,y_train)
    # print(grid.best_estimator_)
    svm = SVC(kernel='linear', probability=True, decision_function_shape='ovo',C=grid.best_params_['C'])
    svm.fit(X_train, y_train)
    y_pred = svm.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, y_pred, multi_class='ovo')
    # Compute predictive probabilities on test data
    return auc_score, y_pred

#%% Batch decoder analysis 

# For loop to process each session
for session in range(len(dfMov)):
    sessionName = dfMov['Session'].iloc[session]
    dff = dfMov['DFF'].iloc[session]
    data = np.transpose(dff)
    animalID = dfMov['animalID'].iloc[session]

    # Reconstruct the dataset to make the trials by labels (time window = 2)
    nUnits = np.shape(data)[1]
    matrix = np.zeros([int(nUnits), int(nRepeats), (int(tRepeat) * int(freqNeuro))])
    print('nUnits=%s'%(nUnits))

    for i in range(nUnits):
        for j in range(nRepeats):
            hold = data[:, i]
            matrix[i, j, :] = hold[j * 272:(j + 1) * 272].T

    unitByRepeat = matrix
    unitDuringOff = unitByRepeat[:, :, 0:freqNeuro * tOff]
    unitDuringOn = unitByRepeat[:, :, freqNeuro * tOff:]
    unitConcat = np.reshape(unitDuringOn, (nUnits, 32 * 224))

    # Find the average neural response for each movie (trial)
    unitByTrial = np.reshape(unitConcat, (nUnits, (freqNeuro * tOn), nRepeats))

    # For extracting data only last 1s window for decoding
    tWindow = 1
    matrix = np.zeros([int(nUnits), int(nTrials), int(tWindow * freqNeuro)])

    for i in range(nUnits):
        for j in range(nTrials):
            hold = unitByTrial[i, j, :]
            matrix[i, j, :] = hold[int(tWindow * freqNeuro):].T

    unitByTrial2 = matrix
    unitAvgPerTrial2 = np.mean(unitByTrial2, axis=2)
    samples = unitAvgPerTrial2.T

    # For loop to generate AUC scores (n_nIters, tr_nIters)
    testn = list(range(5, int(nUnits), 5))

    n_nIters = 250
    tr_nIters = 100
    results_auc = np.zeros([len(testn), n_nIters, tr_nIters])
    results_probs = np.zeros([n_nIters, tr_nIters, 74, 7])

    for nt in range(len(testn)):
        n = testn[nt]

        for j in range(n_nIters):
            if j % 10 == 0:
                print('Iterating %s th time during population size = %s' % (j, n))

            np.random.seed(j)
            idx_pre = np.random.choice(np.arange(nUnits), n)

            for i in range(tr_nIters):
                X_train, X_test, y_train, y_test = train_test_split(samples[:, idx_pre], labels, test_size=test_size,
                                                                    random_state=i, stratify=labels)
                auc_score, y_pred = fit_svm(X_train, y_train, X_test, y_test)
                results_auc[nt, j, i] = auc_score

    #Save results into a dataframe 
    resultsDF = pd.DataFrame()
    
    for nt in range(len(testn)):
        for j in range(n_nIters):
            hold = pd.DataFrame()
            hold['AUC'] = results_auc[nt,j,:] 
            hold['neuIteration'] = j
            hold['Pop Size'] = testn[nt]
            hold['trialIteration'] = np.arange(tr_nIters)
            hold['N'] = nUnits
            hold['Session'] = sessionName
            hold['Animal'] = animalID
                        
            resultsDF = pd.concat([resultsDF, hold], ignore_index=True)
            
    fileName = sessionName + '_auc_scores.csv'    
   
    resultsDF.to_csv(savepath+fileName)
        
    print('resultsDF saved for %s' % (sessionName))
