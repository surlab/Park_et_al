# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:02:53 2024

@author: jihop
"""

import os
import sys
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import pingouin as pg
from scipy.stats import pearsonr
# from scipy.interpolate import interp1d
import re
# from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

sys.path.append('C:\\Users\\jihop\\Documents\\GitHub\\neuron-analysis\\functions\\')
sys.path.append('/Users/jihopark/Documents/GitHub/neuron-analysis/functions/')

# Change the font for plotting
plt.rcParams['font.family'] = 'Arial'

def causal_half_gaussian_filter(data, sigma):
    # Create the causal half Gaussian filter kernel
    x = np.arange(0, 4 * sigma, dtype=float)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2)) * (x >= 0)

    # Normalize the kernel
    kernel /= np.sum(kernel)

    # Apply the causal half Gaussian filter using convolution
    filtered_data = np.convolve(data, kernel, mode='same')

    return filtered_data

#%% Define variables  (MAC)

dataType = 'constitutive'

analysisDrive = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','')
dataDrive = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','data','master','')
savepath = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','GLM',dataType,'')
plotDrive = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','plots',dataType,'')

#%% Define variables  (WINDOWS)

dataType = 'constitutive'

googleDrive = os.path.join('G:','My Drive','mrcuts','')
analysisDrive = os.path.join(googleDrive,'analysis','')
dataDrive = os.path.join(googleDrive,'data','master','')
savepath = os.path.join(analysisDrive,'GLM',dataType,'')
# masterDrive = os.path.join('C:','Users','Jiho Park','Dropbox (MIT)','mrcuts','data','')
plotDrive = os.path.join(analysisDrive,'plots',dataType,'')

#%% Make a list of all master files 

filesList = []

# List of specific names to check for in the file name
specific_names = ['mrcuts07', 'mrcuts24', 'mrcuts25', 'mrcuts26', 'mrcuts27', 'mrcuts28', 'mrcuts29', 'mrcuts30']
specific_dates = ['230803','230804','230809','230810','230814','221123','221126']
filesList = []

for subdir, dirs, files in os.walk(dataDrive):
    for file in files:
        if 'mov' in file and any(name in file for name in specific_names) and any(date in file for date in specific_dates):
            file_path = os.path.join(subdir, file) 
            filesList.append(file_path)
            
# Sort the filesList by name
filesList = sorted(filesList, key=lambda x: os.path.basename(x))

# Check the sessions and delete sessions if needed

nFiles = len(filesList)
print(f"Number of movies master files = {nFiles}")

#%% Load a master file and extract the data

nFrames = 8704

# Initialize empty lists to store the extracted data
neuroList = []
wheelList = []
pupilList = []
actNumList = []
dffList = []
nameList = []

# Iterate over each master file
for file_path in filesList:
    # Load the MATLAB file
    mat = scipy.io.loadmat(file_path)

    # Check if 'pupil' and 'wheel' fields exist in the dataset
    if 'pupil' in mat['master']['data'][0][0].dtype.fields and 'wheel' in mat['master']['data'][0][0].dtype.fields:
        
        wheel = mat['master']['data'][0][0]['wheel'][0][0][0][0]['norm']  # Get the normalized wheel data
        pupil = mat['master']['data'][0][0]['pupil'][0][0][0][0]['diam']['filt_zsc'][0][0][0]  # Get the filtered pupil data
        # pupil = mat['master']['data'][0][0]['pupil'][0][0][0][0]['diam']['filt'][0][0][0]
        
        # Check if wheel and pupil data have non-zero values
        if np.any(wheel != 0) and np.any(pupil != 0):
            filename = os.path.basename(file_path)
            # Define the pattern
            pattern = r'-\d{3}_master\.mat'
            
            # Remove the pattern from each file name
            sessionName = re.sub(pattern, '', filename)
            
            filename = os.path.basename(file_path)
            # Define the pattern
            pattern = r'-\d{3}_master\.mat'
            
            # Remove the pattern from each file name
            sessionName = re.sub(pattern, '', filename)
            
            # Access the desired structure or field in the MATLAB file
            analysis = mat['master']['analysis'][0][0]['neuro'][0][0][0][0][0][0][0]  # Extract out the neuro structure
            data = mat['master']['data'][0][0]['neuro'][0][0][0][0]
            actNum = data[6]['activity_num']
            dff = mat['master']['data'][0][0]['neuro'][0][0]['dff'][0][0]  # Calcium data nUnits * nFrames
    
            # Append the extracted data to the respective lists
            nameList.append(sessionName)
            neuroList.append(dff)
            wheelList.append(wheel)
            pupilList.append(pupil)

# Verify the extracted data for each file
print(f"Number of master files = {len(filesList)}")
print(f"Number of sessions with pupil and wheel data = {len(nameList)} \n {nameList}")

#%% Z score each neuronal trace and make design matrix 

# Example session
N = 15 

dff = neuroList[N]

# Reshape dff array

nUnits = dff.shape[0]

# Calculate mean and standard deviation along the frames axis
mean_values = np.mean(dff, axis=1, keepdims=True)
std_values = np.std(dff, axis=1, keepdims=True)

# Calculate z-scores
zDff = (dff - mean_values) / std_values

# View the z scored dff in heatmap

plt.figure(figsize=(10,10))
sns.heatmap(zDff)

#%% EXAMPLE session GLM fitting of one neuron

labelDF = pd.DataFrame()
labelDF['frameID'] = np.arange(0,nFrames,1)

savefits = dict()

tr_nIters = 10

# Initialize a DataFrame to store R2 prediction values
resultsDF = pd.DataFrame(columns=['Session','R2'])
session = nameList[N]
df = pd.DataFrame(zDff)

print(f"Session: {session}")

savefits = dict()
X_train = []
sessR2 = []

labelDF = pd.DataFrame()
labelDF['frameID'] = np.arange(0,nFrames,1)

for n in range(nUnits):
    
    sample = df.iloc[n]
    
    dm = df.drop(n, axis=0).reset_index(drop=True)
    
    X_train = []
    R2 = []
    
    # Use each frame as a trial point and split 
    y = np.arange(nFrames)
    
    
    for iteration in range(tr_nIters):
        
        print(f'Trial iteration = {iteration}')
        
        # Randomly select units by nSize 
        
        dmSampled = dm.sample(n=20, random_state=iteration).reset_index(drop=True) 
        
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
        
        # y_train = Y_train
        # y_test = Y_test
        
        # glm = RidgeCV(alphas=[0.1, 1.0, 10.0])
        glm = Lasso(alpha=0.00005, max_iter=10000)
        glm.fit(X_train.T, Y_train)
        
        test_r2 = glm.score(X_test.T, Y_test)
        
        R2.append(test_r2)
        
    avgR2 = np.mean(R2)
    sessR2.append(avgR2)
    
    print(f'For neuron {n+1}: Average R2 = {avgR2}')

    if avgR2 > 0.1:
        Y_hat = glm.predict(X_test.T)
        plt.figure(figsize=(16, 3))
        plt.plot(Y_test, label='Test Set TRUE', color='k', alpha=0.5)
        plt.plot(Y_hat, label='Test Set PREDICTION', color='crimson', alpha=0.8)
        plt.legend(loc='upper right', frameon=False)
        sns.despine()
        # Add text annotation for R2 value
        text = 'R2: ' + str(test_r2)
        plt.text(0.5, 0.9, text, transform=plt.gca().transAxes, fontsize=12)
        plt.xlim([0, Y_hat.shape[0]])
        plt.title('Neuron ' + str(n))
        
        
        # plt.savefig(plotDrive+'Single neuron GLM example ('+str(x)+').svg')
        plt.suptitle('Single neuron GLM example ('+str(n)+')', fontsize=13, y=1.05)
        # plt.savefig(plotDrive+f'{session}_n={n}.svg',format='svg')
        plt.show()
    else:
        pass
    
    
resultsDF = pd.concat([resultsDF,pd.DataFrame({'Session':[session],
                                               'R2':[sessR2]})],ignore_index=True)
       


#%% Start with testn, tr_nIters...

labelDF = pd.DataFrame()
labelDF['frameID'] = np.arange(0,nFrames,1)

savefits = dict()

tr_nIters = 5
n_nIters = 5
# Set a limit on number of neurons to be used for all sessions 
nSize = 15

# Initialize a DataFrame to store R2 prediction values
sessionDF = pd.DataFrame(columns=['Session','N','Pop Size','R2','Beta'])
session = nameList[N]
df = pd.DataFrame(zDff)

print(f"Session: {session}")

for n in range(nUnits):
    
    print(f'Neuron={n+1}')
    
    sample = df.iloc[n]
    
    dm = df.drop(n, axis=0).reset_index(drop=True)
    
    testn = list(range(5,nSize+1,5))
    
    X_train = []
    
    results_R2 = np.zeros([len(testn),tr_nIters])
    
    # Use each frame as a trial point and split 
    y = np.arange(nFrames)
    
    for nt in range(len(testn)):
        test_size = testn[nt]
        
        R2ByN = []
        BetaByN = []
        # BetaBytrIteration = np.zeros([n_nIters,tr_nIters])
        
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
                
                test_r2 = glm.score(X_test.T, Y_test)
                
                # Manually find r2 from correlation between empirical and predicted 
                y_hat = glm.predict(X_test.T)
                corr_coeff = pearsonr(Y_test,y_hat)[0]
                r2 = corr_coeff**2
                weights = glm.coef_
                
                # results_R2[nt,iteration] = r2
                
                R2Iteration.append(r2)
                betaIteration.append(weights)
                betaArray = np.vstack(betaIteration)
                
            R2Iter = np.mean(R2Iteration)
            betaIter = np.mean(betaIteration,axis=0)
            betaIter2 = np.mean(betaArray,axis=0)
            
            R2ByN.append(R2Iter)
            BetaByN.append(betaIter2)
        
            print(f'For neuron {n+1}: Pop size = {test_size}: Average R2 = {R2Iter}')

    
            if R2Iter > 0.1:
                # Y_hat = glm.predict(X_test.T)
                plt.figure(figsize=(16, 3))
                plt.plot(Y_test, label='Test Set TRUE', color='k', alpha=0.5)
                plt.plot(y_hat, label='Test Set PREDICTION', color='crimson', alpha=0.8)
                plt.legend(loc='upper right', frameon=False)
                sns.despine()
                # Add text annotation for R2 value
                text = 'R2: ' + str(R2Iter)
                plt.text(0.5, 0.9, text, transform=plt.gca().transAxes, fontsize=12)
                plt.xlim([0, y_hat.shape[0]])
                plt.title('Neuron ' + str(n))
                
                plt.suptitle('Single neuron GLM example ('+str(n)+')', fontsize=13, y=1.05)
                # plt.savefig(plotDrive+f'{session}_n={n}.svg',format='svg')
                plt.show()
            else:
                pass
                        
        R2Avg = np.mean(R2ByN)

        sessionDF = pd.concat([sessionDF,pd.DataFrame({'Session':[session],
                                                       'N':[n],
                                                       'Pop Size':[test_size],
                                                       'R2':[R2Avg],
                                                       'Beta':[BetaByN]})],ignore_index=True)
                

        
fileName = session + '_glm_results_with_weights_all_pop_sizes.csv'     

sessionDF.to_csv(savepath+fileName)
print('Saved %s' % (session))


#%% BATCH with iterations: Now do the same but with iterations 

n_nIters = 30
tr_nIters = 20
# Set a limit on number of neurons to be used for all sessions 
nSize = 20

resultsDF = pd.DataFrame(columns=['Session','N','Pop Size','R2','Beta'])

for j in range(len(nameList)):
    
    print(f"{j}: Session: {nameList[j]}")
    
    session = nameList[j]
    neuro = neuroList[j]
    nUnits = neuro.shape[0]
    dff = neuro
    
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
        
        # testn = list(range(5,nSize+1,5))
        testn = [nSize]
        
        X_train = []
        
        # Use each frame as a trial point and split 
        y = np.arange(nFrames)
        
        for nt in range(len(testn)):
            test_size = testn[nt]
            
            
            R2ByN = []
            BetaByN = []
            # BetaBytrIteration = np.zeros([n_nIters,tr_nIters])
            
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
                    
                    test_r2 = glm.score(X_test.T, Y_test)
                    
                    # Manually find r2 from correlation between empirical and predicted 
                    y_hat = glm.predict(X_test.T)
                    corr_coeff = pearsonr(Y_test,y_hat)[0]
                    r2 = corr_coeff**2
                    weights = glm.coef_
                    
                    # results_R2[nt,iteration] = r2
                    
                    R2Iteration.append(r2)
                    betaIteration.append(weights)
                    betaArray = np.vstack(betaIteration)
                    
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
            
            # resultsDF = pd.concat([resultsDF,pd.DataFrame({'Session':[session],
            #                                                'N':[n],
            #                                                'Pop Size':[test_size],
            #                                                'R2':[R2Iter],
            #                                                'Beta':[betaIteration]})],ignore_index=True)
    fileName = session + '_glm_results_with_weights.csv'     
    
    sessionDF.to_csv(savepath+fileName)
    print('Saved %s' % (session))
        
# Save the DataFrame to a CSV file
# csvName = 'GLM_predictions_movie_trnIter_50_with_weights.csv'
# resultsDF.to_csv(savepath+csvName, index=False)

#%% Load data (constitutive)

os.chdir(savepath)
# dfMaster = pd.read_csv('GLM_predictions_movies_20.csv')
dfMaster = pd.read_csv('G:/My Drive/mrcuts/analysis/GLM/constitutive/GLM_predictions_movie_20.csv')

for i in range(len(dfMaster)):
    session = dfMaster.iloc[i]['Session']
    
    pattern = r'(\d{6}).*'
    match = re.search(pattern, session)
    
    if match:
        date = match.group(1)
        dfMaster.at[i, 'Date'] = date
        
    pattern = r'.*(an\d{3}).*'
    match = re.search(pattern, session)
    
    if match:
        anID = match.group(1)
        dfMaster.at[i, 'animalID'] = anID
        
    pattern = r'.*(mrcuts\d{2}).*'
    match = re.search(pattern, session)
    
    if match:
        anID = match.group(1)
        dfMaster.at[i, 'animalID'] = anID
        
def get_group(anID):
    if anID == 'mrcuts07' or anID == 'mrcuts28' or anID == 'mrcuts29' or anID == 'mrcuts30':
        return 'Control'
    elif anID == 'mrcuts24' or anID == 'mrcuts25' or anID == 'mrcuts26' or anID == 'mrcuts27':
        return 'Exp'
    else:
        return None

dfMaster['Group'] = dfMaster['animalID'].apply(get_group)
# dfMaster = dfMaster[dfMaster['animalID'] != 'mrcuts24']


dfMaster.to_csv(savepath+'GLM_movies_constitutive_20_master_df.csv')
#%% Load data

# dfMaster = pd.read_csv(savepath+'GLM_gratings_master_df.csv')
dfMaster = pd.read_csv(savepath+'GLM_movies_constitutive_master_df.csv',index_col=0)

# Filter out the problematic sessions based on heatmaps (ROI labeling seems to be off)
sessions_to_drop = ['230809_mrcuts28_fov4_mov','230804_mrcuts26_fov3_mov']
df = dfMaster[~dfMaster['Session'].isin(sessions_to_drop)].reset_index(drop=True)


#%% Plot by Pop Size

sns.catplot(df,x='Pop Size',y='R2',hue='Group')

#%% Find the Pop Size at which R2 maximizes

result = df.loc[df.groupby(['Session', 'N'])['R2'].idxmax()]

print(result)

#%% Convert the lists of strings into numerical values 

import ast 

# Function to convert string lists to numerical lists
def convert_str_list_to_float(lst_str):
    try:
        return ast.literal_eval(lst_str)
    except (SyntaxError, ValueError):
        return None

# Apply the function to each column
dfMaster['R2'] = dfMaster['R2'].apply(convert_str_list_to_float)

dfMaster = dfMaster.drop('Unnamed: 0', axis=1, errors='ignore')

# Now, the columns contain numerical lists
print(dfMaster)

#%% Explode the DF 

df = dfMaster.explode(['R2'])

#%% Visualize the data 

orderGroup = ['Control', 'Exp']

# plt.figure(figsize=(10, 10))

dfMelted = pd.melt(df, id_vars=['Group', 'Session', 'Date', 'animalID'], value_vars=['R2'], var_name='Column', value_name='Value')

#%% Compute the p values 

dfMelted['Value'] = pd.to_numeric(dfMelted['Value'], errors='coerce')
df['R2'] = pd.to_numeric(df['R2'], errors='coerce')

pVal = mannwhitneyu(df[df['Group']=='Control']['R2'],df[df['Group']=='Exp']['R2'])

#%% Visualize the data& save

orderGroup = ['Control', 'Exp']

plt.figure(figsize=(10, 10))

# Create a plot
plt.figure(figsize=(10, 10))
# sns.barplot(x='Group', y='Value', hue='Column', data=dfMelted, dodge=True,order=orderGroup)
sns.boxplot(x='Group',y='Value',hue='Column',fill=False,
            fliersize=3,gap=0.1,
            data=dfMelted,dodge=True,order=orderGroup)
plt.title('Comparison of R2 values')
plt.ylabel('R2 Value')
plt.ylim([-0.7,0.7])
# plt.text(1,0.8,f'Control: pPupil={pControlPupil}\npWheel={pControlWheel}\nExp: pPupil={pExpPupil}\npWheel={pExpWheel}\nBetween pupil={pBetweenPupil}\nBetween wheel={pBetweenWheel}')
plt.legend()
sns.despine()
# plt.savefig(plotDrive+'GLM R2 contribution of pupil and wheel.svg')
plt.show()