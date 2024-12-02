# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:45:49 2024

@author: jihop

# Contains GLM functions 

"""

import re
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import  RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# For filtering data
def causal_half_gaussian_filter(data, sigma):
    # Create the causal half Gaussian filter kernel
    x = np.arange(0, 4 * sigma, dtype=float)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2)) * (x >= 0)

    # Normalize the kernel
    kernel /= np.sum(kernel)

    # Apply the causal half Gaussian filter using convolution
    filtered_data = np.convolve(data, kernel, mode='same')

    return filtered_data

# Function to convert string lists to numerical lists
def convert_str_list_to_float(lst_str):
    try:
        return ast.literal_eval(lst_str)
    except (SyntaxError, ValueError):
        return None

# Define class for temporal basis for design matrix
class NewTemporalBasis():

    # Same cosine bump spread temporally
    def __init__(self, nBases=5,width=24):

        self.centers = np.arange(48, 80, 2)
        self.widths = np.ones(nBases)*width
        self.bases = np.arange(0,80,0.25)
        
        # self.centers = np.arange(0, 40, 8)
        # self.widths = np.ones(nBases)*width
        # self.bases = np.arange(0, 40, 0.25)

        self.filters = np.full((len(self.bases), len(self.centers)), np.NaN)
        # self.filters = np.full((len(self.bases), nBases), np.NaN)
        
        for idx, val in enumerate(zip(self.centers,self.widths)):
            self.filters[:,idx] = self.cosinebumps(self.bases,val[0],val[1])
        self.filters = self.filters/np.sum(self.filters,axis = 0) # normalization: so the area under each basis is 1

    def cosinebumps(self, x, center, width):
        x_reshape = x.reshape(-1,)
        y = (np.cos(2*np.pi*(x_reshape-center)/width)*0.5+0.5)*(np.absolute(x_reshape-center)<width/2)
        return y

    def plotfilters(self):
        plt.plot(self.bases, self.filters,linewidth=3)
        plt.xlabel('Time');

# tBases_stim.plotfilters()

# GLM for gratings 
def glm_grat(df,labels,nStim,nBases,nFrames,freqNeuro,tDur,tOn,tOff,nTrials,nRep,tr_nIters,savepath):
    
    # Create a label DF
    labelDF = pd.DataFrame()
    labelDF['trialID'] = np.arange(0,128)
    labelDF['direction'] = labels
    labelDF['frameID'] = np.arange(0,10240,80)
    
    # Create a matrix containing visual stimuli
    # Grating One 
    one = np.zeros([1,nStim*tDur*freqNeuro]) # Creating an empty array of frames of one set of stimuli (nStim*tDur*freqNeuro)
    one[0][tOff*freqNeuro] = 1 # Inserting 1s at the frame where the first stimulus comes on (tOff*freqNeuro)
    one = one[0] # Reorienting into a column? 
    gratOne = np.tile(one, nRep) # Concatenating the array for one trial for all repeats 
    matrixOne = np.tile(gratOne,(nBases,1)) # Concatenating the array in rows for  nBases
    # Grating Two
    two = np.zeros([1,nStim*tDur*freqNeuro])
    two[0][(freqNeuro*tDur)+(tOff*freqNeuro)] = 1 # Inserting 1s at the frame where the SECOND stimulus comes on: (freqNeuro*tDur)+(tOff*freqNeuro)
    two = two[0]
    gratTwo = np.tile(two,nRep)
    matrixTwo = np.tile(gratTwo,(nBases,1))
    # Grating Three
    three = np.zeros([1,nStim*tDur*freqNeuro])
    three[0][2*(freqNeuro*tDur)+(tOff*freqNeuro)] = 1
    three = three[0]
    gratThree = np.tile(three,nRep)
    matrixThree = np.tile(gratThree,(nBases,1))
    # Grating Four
    four = np.zeros([1,nStim*tDur*freqNeuro])
    four[0][3*(freqNeuro*tDur)+(tOff*freqNeuro)] = 1
    four = four[0]
    gratFour = np.tile(four,nRep)
    matrixFour = np.tile(gratFour,(nBases,1))
    # Grating Five
    five = np.zeros([1,nStim*tDur*freqNeuro])
    five[0][4*(freqNeuro*tDur)+(tOff*freqNeuro)] = 1
    five = five[0]
    gratFive = np.tile(five,nRep)
    matrixFive = np.tile(gratFive,(nBases,1))
    # Grating Six
    six = np.zeros([1,nStim*tDur*freqNeuro])
    six[0][5*(freqNeuro*tDur)+(tOff*freqNeuro)] = 1
    six = six[0]
    gratSix = np.tile(six,nRep)
    matrixSix = np.tile(gratSix,(nBases,1))
    # Grating Seven
    seven = np.zeros([1,nStim*tDur*freqNeuro])
    seven[0][6*(freqNeuro*tDur)+(tOff*freqNeuro)] = 1
    seven = seven[0]
    gratSeven = np.tile(seven,nRep)
    matrixSeven = np.tile(gratSeven,(nBases,1))
    # Grating Eight
    eight = np.zeros([1,nStim*tDur*freqNeuro])
    eight[0][7*(freqNeuro*tDur)+(tOff*freqNeuro)] = 1
    eight = eight[0]
    gratEight = np.tile(eight,nRep)
    matrixEight = np.tile(gratEight,(nBases,1))

    # Combine all arrays into one final array to convolve
    finalMatrix = np.concatenate([matrixOne,matrixTwo,matrixThree,
                                 matrixFour,matrixFive,matrixSix,
                                 matrixSeven,matrixEight],axis=0)
    
    # Get the temporal basis function 
    tBases_stim = NewTemporalBasis(nBases=nBases,width=24)
    
    
    resultsDF = pd.DataFrame(columns=['Session','n','R2','R2_vis','R2_pupil','R2_wheel'])

    for j in range(len(df)):
        
        print(f"{j}: Session: {df['Session'][j]}")
        
        session = df['Session'].iloc[j]
        neuro = df['DFF'].iloc[j]
        pupil = df['Pupil'].iloc[j]
        wheel = df['Wheel'].iloc[j]
        
        nUnits = neuro.shape[0]
        dff = neuro
        
        print(f"nUnits={nUnits}")
        
        # Interpolate wheel and pupil data 
        # Create an array of indices for the original 'wheel' array
        idxOri = np.arange(wheel.size)
        
        # Create an array of indices for the desired number of frames
        idxNew = np.round(np.linspace(0, len(idxOri) - 1, nFrames)).astype(int) 
        
        # Use linear interpolation to interpolate 'wheel' data at desired indices
        wheelNew = interp1d(idxOri, wheel)(idxNew)
        # Use linear interpolation to interpolate 'pupil' data at desired indices
        pupilNew = interp1d(idxOri, pupil)(idxNew)

        # Create a design matrix for this session
        nBases = 5
        nFeatures = np.shape(finalMatrix)[0]
        designmatrix = pd.DataFrame()
        
        # Each grating is temporally spanned out into 5 temporal bases
        for n in range(nStim):
            for nFilt in range(nBases):
                thisFilt = tBases_stim.filters[:,nFilt]   
                designmatrix['Grating'+str(n+1)+'_base'+str(nFilt)] = np.convolve(finalMatrix[n,:], thisFilt, 'same') 
        
        # Stack the additional columns with the design matrix
        # designmatrix = np.hstack((designmatrix, pupilNew[:, np.newaxis], wheelNew[:, np.newaxis]))
        
        # Add 'Pupil' and 'Wheel' columns with respective arrays
        designmatrix['Pupil'] = pupilNew
        designmatrix['Wheel'] = wheelNew
        
        # Scale the x-values using StandardScaler
        scaler = StandardScaler()
        scaled_designmatrix = scaler.fit_transform(designmatrix.values)
        
        # Create a new DataFrame with the scaled values and column names
        scaled_designmatrix = pd.DataFrame(scaled_designmatrix, columns=designmatrix.columns)
        
        # Now fit the glm for each neuron
        X_train = []
        
        R2_all = []
        R2_vis = []
        R2_pupil = []
        R2_wheel = []
        
        for predictor_type in ['All','Visual stimulus','Pupil','Wheel']:

            print(f"GLM fitting for {predictor_type}")
            
            # R2 = []
            
            if predictor_type == 'All':
                dm = scaled_designmatrix
            elif predictor_type == 'Visual stimulus':
                dm = scaled_designmatrix.copy()
                dm.iloc[:, :40] = 0
            elif predictor_type == 'Pupil':
                dm = scaled_designmatrix.copy()
                dm.iloc[:,40] = 0
            elif predictor_type == 'Wheel':
                dm = scaled_designmatrix.copy()
                dm.iloc[:,41] = 0

            for x in range(nUnits):
                neuData = dff[x]
                neuData = causal_half_gaussian_filter(neuData, 5)
                y = np.arange(nTrials)
                
                R2 = []
                
                for iteration in range(tr_nIters):
                    y_train, y_test = train_test_split(y, test_size=0.25, random_state=iteration, stratify=labels)
        
                    nRows = len(y_train) * 80
                    X_train = pd.DataFrame(index=range(nRows), columns=range(nFeatures + 2))
                
                    for i in range(len(y_train)):
                        idx1 = y_train[i]
                        idx2 = labelDF[labelDF['trialID'] == idx1].frameID
                        idx2 = idx2[idx1]
                        hold = dm[idx2:idx2+80][:]
                        X_train[i*80:(i*80)+80] = hold
                
                    # Perform data imputation on X_train
                    imputer = SimpleImputer(strategy='mean')
                    X_train_imputed = imputer.fit_transform(X_train)
                
                    nRows = len(y_test) * 80
                    X_test = pd.DataFrame(index=range(nRows), columns=range(nFeatures + 2))
                
                    for i in range(len(y_test)):
                        idx1 = y_test[i]
                        idx2 = labelDF[labelDF['trialID'] == idx1].frameID
                        idx2 = idx2[idx1]
                        hold = dm[idx2:idx2+80][:]
                        X_test[i*80:(i*80)+80] = hold
                
                    # Perform data imputation on X_test
                    X_test_imputed = imputer.transform(X_test)
                
                    for i in range(len(y_train)):
                        idx1 = y_train[i]
                        idx2 = labelDF[labelDF['trialID'] == idx1].frameID
                        idx2 = idx2[idx1]
                        hold = neuData[idx2:idx2+80]
                
                        if i == 0:
                            Y_train = hold
                        else:
                            Y_train = np.hstack((Y_train, hold))
                
                    for i in range(len(y_test)):
                        idx1 = y_test[i]
                        idx2 = labelDF[labelDF['trialID'] == idx1].frameID
                        idx2 = idx2[idx1]
                        hold = neuData[idx2:idx2+80]
                
                        if i == 0:
                            Y_test = hold
                        else:
                            Y_test = np.hstack((Y_test, hold))
                
                    y_train = Y_train
                    y_test = Y_test
                
                    glm = RidgeCV(alphas=[1e-5, 1e-3, 5.0])
                    glm.fit(X_train_imputed, y_train)
                
                    # test_r2 = glm.score(X_test_imputed, y_test)
                    # R2.append(test_r2)
    
                    # Manually find r2 from correlation between empirical and predicted 
                    y_hat = glm.predict(X_test_imputed)
                    corr_coeff = pearsonr(y_test,y_hat)[0]
                    r2 = corr_coeff**2
                    
                    # print(f"Neuron {x}: r2:{r2}")
                    
                    R2.append(r2)
                    
                avgR2 = np.mean(R2)
                print(f"Neuron {x}: average R2 = {avgR2}")
            
                if predictor_type == 'All':
                    R2_all.append(avgR2)
                elif predictor_type == 'Visual stimulus':
                    R2_vis.append(avgR2)
                elif predictor_type == 'Pupil':
                    R2_pupil.append(avgR2)
                elif predictor_type == 'Wheel':
                    R2_wheel.append(avgR2)
            
        resultsDF = pd.concat([resultsDF, pd.DataFrame({'Session': [session],
                                                        'n': x,
                                                        'R2': [R2_all],
                                                        'R2_vis': [R2_vis],
                                                        'R2_pupil': [R2_pupil],
                                                        'R2_wheel': [R2_wheel]})],ignore_index=True) 
    
    # Label the df with animalID and date
    for i in range(len(resultsDF)):
        session = resultsDF.iloc[i]['Session']
        
        pattern = r'(\d{6}).*'
        match = re.search(pattern, session)
        
        if match:
            date = match.group(1)
            resultsDF.at[i, 'Date'] = date
            
        pattern = r'.*(an\d{3}).*'
        match = re.search(pattern, session)
        
        if match:
            anID = match.group(1)
            resultsDF.at[i, 'animalID'] = anID
            
        pattern = r'.*(mrcuts\d{2}).*'
        match = re.search(pattern, session)
        
        if match:
            anID = match.group(1)
            resultsDF.at[i, 'animalID'] = anID
    
    # # Convert the values to float 
    # resultsDF['R2'] = resultsDF['R2'].apply(convert_str_list_to_float)
    # resultsDF['R2_vis'] = resultsDF['R2_vis'].apply(convert_str_list_to_float)
    # resultsDF['R2_pupil'] = resultsDF['R2_pupil'].apply(convert_str_list_to_float)
    # resultsDF['R2_wheel'] = resultsDF['R2_wheel'].apply(convert_str_list_to_float)

    return resultsDF


### For any stimulus type ###

def glm_neuron(df,nFrames,tr_nIters,savepath):

    resultsDF = pd.DataFrame(columns=['Session','R2'])

    for j in range(len(df)):
        
        session = df['Session'].iloc[j]
        neuro = df['DFF'].iloc[j]
        nUnits = neuro.shape[0]
        dff = neuro
        
        print(f"{j}: Session: {session}")
        
        # Calculate mean and standard deviation along the frames axis
        mean_values = np.mean(dff, axis=1, keepdims=True)
        std_values = np.std(dff, axis=1, keepdims=True)

        # Calculate z-scores
        zDff = (dff - mean_values) / std_values
        dfZDff = pd.DataFrame(zDff)
        
        print(f"nUnits={nUnits}")
        
        X_train = []
        sessR2 = []
        
        labelDF = pd.DataFrame()
        labelDF['frameID'] = np.arange(0,nFrames,1)

        for n in range(nUnits):
            
            sample = dfZDff.iloc[n]
            
            dm = dfZDff.drop(n, axis=0).reset_index(drop=True)
            
            X_train = []
            R2 = []
            
            # Use each frame as a trial point and split 
            y = np.arange(nFrames)
            
            for iteration in range(tr_nIters):
                y_train, y_test = train_test_split(y, test_size=0.25, random_state=iteration) # This splits the pupil trace into train and test
                
                X_train = pd.DataFrame(index=range(nUnits-1), columns=range(len(y_train)))
                
                # Use the y_train indices to generate the same frame idx for X_train 
                for i in range(len(y_train)):
                    idx1 = y_train[i]
                    idx2 = labelDF[labelDF['frameID'] == idx1].frameID
                    idx2 = idx2[idx1]
                    X_train[i] = dm[idx2][:]
                
                X_test = pd.DataFrame(index=range(nUnits-1), columns=range(len(y_test)))
                
                for i in range(len(y_test)):
                    idx1 = y_test[i]
                    idx2 = labelDF[labelDF['frameID'] == idx1].frameID
                    idx2 = idx2[idx1]
                    X_test[i] = dm[idx2][:]
                
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
                
                y_train = Y_train
                y_test = Y_test
                
                glm = RidgeCV(alphas=[1e-5, 1e-3, 5.0])
                glm.fit(X_train.T, y_train)
                
                test_r2 = glm.score(X_test.T, y_test)
                
                R2.append(test_r2)
                
            avgR2 = np.mean(R2)
            
            print(f'For neuron {n+1}: Average R2 = {avgR2}')
            
            sessR2.append(avgR2)
        
        sessionDF = pd.DataFrame({'Session':[session],'R2':[sessR2]})
        resultsDF = pd.concat([resultsDF,pd.DataFrame({'Session':[session],
                                                       'R2':[sessR2]})],ignore_index=True)
        
        sessionDF.to_csv(savepath+f'{session}_GLM_movies.csv')
    # Label the df with animalID and date
    for i in range(len(resultsDF)):
        session = resultsDF.iloc[i]['Session']
        
        pattern = r'(\d{6}).*'
        match = re.search(pattern, session)
        
        if match:
            date = match.group(1)
            resultsDF.at[i, 'Date'] = date
            
        pattern = r'.*(an\d{3}).*'
        match = re.search(pattern, session)
        
        if match:
            anID = match.group(1)
            resultsDF.at[i, 'animalID'] = anID
            
        pattern = r'.*(mrcuts\d{2}).*'
        match = re.search(pattern, session)
        
        if match:
            anID = match.group(1)
            resultsDF.at[i, 'animalID'] = anID
    
    # Convert the values to float 
    # resultsDF['R2'] = resultsDF['R2'].apply(convert_str_list_to_float)
    
    return resultsDF

#%% Plotting functions for GLM results 

