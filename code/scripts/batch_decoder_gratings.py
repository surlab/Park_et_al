#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 07:53:57 2022

@author: jihopark
"""

"""

This script runs SVM-decoder analysis on each gratings session and saves the output as individual csv files 

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
homepath = os.path.join('C:','Users','jihop','Documents','Park_et_al_2024','')
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

# Create a new df containing only gratings sessions & extract unique values 
dfGrat = extract.extract_grat(df)
dfGrat = extract.get_dff(dfGrat)

#%% Define fixed parameters 

freqNeuro = 16
nTrials = 16*8 # each grating stimulus is repeated 16 times 
timepts = 5*16 
grat_time = 5
tOn = 2;
tOff = 3;
t = 640
num_stim = 8
labels = ['0','45','90','135','180','225','270','315'] * 16

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
for session in range(len(dfGrat)):
    sessionName = dfGrat['Session'].iloc[session]
    dff = dfGrat['DFF'].iloc[session]
    data = np.transpose(dff)
    animalID = dfGrat['animalID'].iloc[session]

    # Reconstruct the dataset to make nUnits * nTrials * 80 
    nUnits = np.shape(data)[1]
    print('nUnits = %s' % (nUnits))
    samples = np.zeros([int(nUnits),int(nTrials),int(timepts)])
            
    for t in range(nTrials):
        samples[:,t,:] = data[t*80:(t+1)*80,:].T
    
    # For extracting data only last 1s window for decoding
    tWindow = 1
    
    # Extract the data during gratings ON period         
    samples = np.nanmean(samples[:,:,freqNeuro*(tOff+tWindow):], 2)
    samples = samples.T

    # For loop to generate AUC scores (n_nIters, tr_nIters)
    testn = list(range(5, int(nUnits), 5))

    n_nIters = 250
    tr_nIters = 100
    results_auc = np.zeros([len(testn), n_nIters, tr_nIters])
    # results_probs = np.zeros([n_nIters, tr_nIters, 43, 8])

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

    # Save results into a dataframe 
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
    print('Saved %s' % (sessionName))
    