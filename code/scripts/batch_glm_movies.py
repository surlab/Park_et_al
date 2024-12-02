# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:02:53 2024

@author: jihop

Script for running single neuron encoding model of population activity on multiple sessions

"""

# Import packages needed for running the code
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from functions import extract

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

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


#%% BATCH with iterations: Now do the same but with iterations 

# Numbner of frames in a single movies session
nFrames = 8704
# Set a limit on number of neurons to be sampled from each session 
nSize = 20
# Number of iterations to randomly sample from each population
n_nIters = 30
# Number of iterations to repeat train & test split and glm fitting 
tr_nIters = 20


resultsDF = pd.DataFrame(columns=['Session','N','Pop Size','R2','Beta'])

for session in range(len(dfMov)):
    sessionName = dfMov['Session'].iloc[session]
    dff = dfMov['DFF'].iloc[session]
    nUnits = dff.shape[0]

    print(f"Session: {session}")
    
    # Calculate mean and standard deviation along the frames axis
    mean_values = np.mean(dff, axis=1, keepdims=True)
    std_values = np.std(dff, axis=1, keepdims=True)

    # Calculate z-scores
    zDff = (dff - mean_values) / std_values
    df = pd.DataFrame(zDff)
    
    print(f"nUnits={nUnits}")
    
    savefits = dict()
    X_train = []
    sessR2 = []
    
    sessionDF = pd.DataFrame(columns=['Session','N','Pop Size','R2','Beta'])
    
    labelDF = pd.DataFrame()
    labelDF['frameID'] = np.arange(0,nFrames,1)

    for n in range(nUnits):
        
        print(f'Neuron={n+1}')
        
        sample = df.iloc[n]
        
        dm = df.drop(n, axis=0).reset_index(drop=True)
        
        testn = list(range(5,nSize+1,5))
        
        X_train = []
        
        # Use each frame as a trial point and split 
        y = np.arange(nFrames)
        
        for nt in range(len(testn)):
            test_size = testn[nt]
            
            R2ByN = []
            BetaByN = []
            
            for n_Iter in range(n_nIters):
                print(f'Sampling iteration: {n_Iter}')
                
                dmSampled = dm.sample(n=test_size, random_state=n_Iter).reset_index(drop=True) 
                
                R2Iteration = []
                betaIteration = []
            
                for iteration in range(tr_nIters):
                    if iteration % 5 == 0: 
                        print(f'Pop size = {test_size}; Trial iteration = {iteration}')
                
                    y_train, y_test = train_test_split(y, test_size=0.25, random_state=iteration) # This splits the pupil trace into train and test
                    
                    X_train = pd.DataFrame(index=range(test_size), columns=range(len(y_train)))
                    
                    # Use the y_train indices to generate the same frame idx for X_train 
                    for i in range(len(y_train)):
                        idx1 = y_train[i]
                        idx2 = labelDF[labelDF['frameID'] == idx1].frameID
                        idx2 = idx2[idx1]
                        X_train[i] = dmSampled[idx2][:]
                    
                    X_test = pd.DataFrame(index=range(test_size), columns=range(len(y_test)))
                    
                    for i in range(len(y_test)):
                        idx1 = y_test[i]
                        idx2 = labelDF[labelDF['frameID'] == idx1].frameID
                        idx2 = idx2[idx1]
                        X_test[i] = dmSampled[idx2][:]
                    
                    for i in range(len(y_train)):
                        idx1 = y_train[i]
                        idx2 = labelDF[labelDF['frameID'] == idx1].frameID
                        idx2 = idx2[idx1]
                        hold = sample[idx2]
                    
                        if i == 0:
                            Y_train = hold
                        else:
                            Y_train = np.hstack((Y_train, hold))
                    
                    for i in range(len(y_test)):
                        idx1 = y_test[i]
                        idx2 = labelDF[labelDF['frameID'] == idx1].frameID
                        idx2 = idx2[idx1]
                        hold = sample[idx2]
                    
                        if i == 0:
                            Y_test = hold
                        else:
                            Y_test = np.hstack((Y_test, hold))
    
                    glm = Lasso(alpha=0.00005, max_iter=10000)
                    glm.fit(X_train.T, Y_train)
                    
                    # Compute variance explained (R^2) by finding correlation coefficient between empirical data and prediction
                    y_hat = glm.predict(X_test.T)
                    corr_coeff = pearsonr(Y_test,y_hat)[0]
                    r2 = corr_coeff**2
                    # Store each neuron's beta value 
                    weights = glm.coef_
                    
                    R2Iteration.append(r2)
                    betaIteration.append(weights)
                    betaArray = np.vstack(betaIteration)
                
                # Find the average across trial iterations
                R2Iter = np.mean(R2Iteration)
                betaIter = np.mean(betaIteration,axis=0)
                betaIter2 = np.mean(betaArray,axis=0)
                
                R2ByN.append(R2Iter)
                BetaByN.append(betaIter2)
            
                print(f'For neuron {n+1}: Pop size = {test_size}: Average R2 = {R2Iter}')
                
            R2Avg = np.mean(R2ByN)
    
            sessionDF = pd.concat([sessionDF,pd.DataFrame({'Session':[session],
                                                           'N':[n],
                                                           'Pop Size':[test_size],
                                                           'R2':[R2Avg],
                                                           'Beta':[BetaByN]})],ignore_index=True)
            
    fileName = session + '_glm_results_with_weights.csv'     
    
    sessionDF.to_csv(savepath+fileName)
    print('Saved %s' % (session))
