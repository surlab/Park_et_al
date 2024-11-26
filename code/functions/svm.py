#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:56:20 2024

SVM-decoder for visual stimuli information and behavioral data

@author: jihopark
"""


import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats

import statsmodels.formula.api as smf
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

### Define parameters ###

# SVM-decoder 

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


### Gratings ###

def decoder_gratings(df,freqNeuro,nTrials,timepts,tOn,tOff,t,nStim,labels,savepath):
    
    for n in range(len(df)):
        session = df['Session'].iloc[n]
        data = df['DFF'].iloc[n]
        nUnits, nFrames = data.shape
        
        print(f'Processing {session}...')
        
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
        resultsDF.to_csv(savepath+fileName)
        print('Saved %s'%(sessionName))
    
