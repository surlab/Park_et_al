#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 07:53:57 2022

@author: jihopark
"""

"""

"""


import os
import glob
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

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

decoderFolder = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','decoder','')
saveFolder = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','decoder','auc_gratings','')


dataFolder = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','new','')

# Search for CSV files within subfolders
csvFiles = glob.glob(dataFolder + '**/*grat_neuro.csv', recursive=True)

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

#%%

# For loop to process each CSV file
for csvFile in csvFiles:
    print('Loading %s' % (csvFile))
    # Read CSV file
    dFF = pd.read_csv(csvFile, header=None)
    dFF.drop(0, inplace=True)
    data = dFF.to_numpy()
    data = np.transpose(data)
    sessionName = csvFile[-35:-10]
    date = csvFile[-35:-29]
    animalID = csvFile[-28:-20]

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
                # results_probs[j, i, :, :] = y_pred

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
            hold['Date'] = date
            hold['Animal'] = animalID
            
            resultsDF = pd.concat([resultsDF, hold], ignore_index=True)
    
    fileName = sessionName + '_auc_scores.csv'     
    
    resultsDF.to_csv(saveFolder+fileName)
    print('Saved %s' % (sessionName))
    
#%%
resultsDF['Date'] = resultsDF['Date'].astype(str)

def get_timepoint(date_value):
    if date_value == '230116' or date_value == '230117' or date_value == '230118' or date_value == '230120' or date_value == '221005' or date_value == '221006' or date_value == '221123' or date_value == '221126':
        return 'PRE'
    elif date_value == '230202':
        return 'POST (~1wk)'
    elif date_value == '230208' or date_value == '230209' or date_value == '230211' or date_value == '221214' or date_value == '221216' or date_value == '221025' or date_value == '221026':
        return 'POST (~2wk)'
    elif date_value == '230218' or date_value == '230219' or date_value == '221222' or date_value == '221110' or date_value == '221108' or date_value == '221109':
        return 'POST (~3wk)'
    else:
        return None
    
    
def get_group(anID):
    if anID == 'mrcut316' or anID == 'mrcut318' or anID == 'mrcuts07':
        return 'Control'
    elif anID == 'mrcut317' or anID == 'mrcuts13' or anID == 'mrcuts14' or anID == 'mrcuts15' or anID == 'mrcuts16' or anID == 'mrcuts17':
        return 'Exp'
    else:
        return None

resultsDF['Timepoint'] = resultsDF['Date'].apply(get_timepoint)
resultsDF['Group'] = resultsDF['Animal'].apply(get_group)

fileName = 'grat_auc_scores_2.csv'    

os.chdir(decoderFolder)    
resultsDF.to_csv(fileName)