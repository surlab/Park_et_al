# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:50:06 2024

@author: jihop
"""

# Import the functions that I need 
import os
import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from functions import extract, compute

plt.rcParams['figure.max_open_warning'] = 0

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

#%% Load data

listNames = []
listFiles = []
listRez = []
listAdjp = []

for subdir, dirs, files in os.walk(datapath):
    for file in files:
        # All files that end with .mat
        if file.endswith('results.mat'):
            file_path = os.path.join(subdir, file) 
            listFiles.append(file_path)
            
            fileName = os.path.basename(file)
            sessionName = fileName[:-16]
            listNames.append(sessionName)
            
print(f'Number of master files = {len(listFiles)}')

zippedList = list(zip(listNames,listFiles))

for n in range(len(zippedList)):
    # Extract the file name and save in listNames
    fileName = zippedList[n][0]
    filePath = zippedList[n][1]
    print(f'Loading {fileName}...')
    
    # Load the MATLAB file
    mat = scipy.io.loadmat(filePath)
    
    rez = mat['rez']
    adjp = mat['adjp']
    
    # listNames.append(fileName)
    listRez.append(rez)
    listAdjp.append(adjp)
    
#%% Combine the data into one dataframe

# Combine into a DataFrame
df = pd.DataFrame({
    'Session': listNames,
    'Rez': listRez,
    'Adjp': listAdjp
})

df = extract.get_animalID(df)
df['Group'] = df['animalID'].apply(extract.get_group)

# Display the DataFrame to verify
print(df)

# Assuming 'Session' is the column containing session names
session_to_drop = '221123_mrcuts07_fov1_grat'

# Drop rows with the specified session name
df = df[df['Session'] != session_to_drop] 

#%% Extract R2_s values

dfR2 = df

# Initialize the 'R2' column with NaNs
dfR2['R2'] = pd.NA
dfR2['R2_vis'] = pd.NA
dfR2['R2_pupil'] = pd.NA
dfR2['R2_wheel'] = pd.NA
dfR2['pVis'] = pd.NA
dfR2['pPupil'] = pd.NA
dfR2['pWheel'] = pd.NA

for session in range(len(dfR2)):
    name = dfR2.Session.iloc[session]
    rez = dfR2.Rez.iloc[session]
    adjp = dfR2.Adjp.iloc[session]
    
    listR2 = []
    listR2vis = []
    listR2pupil = []
    listR2wheel = []
    listPvis = []
    listPpupil = []
    listPwheel = []
    
    for i in range(rez['R2_s'].shape[1]):
        
        r2 = rez['R2_s'][0,i][0]['all'][0][0]
        r2Vis = rez['R2_s'][0,i][0]['partial'][0][:,10] # without all vis stim 
        r2Pupil = rez['R2_s'][0,i][0]['partial'][0][:,9] # without pupil
        r2Wheel = rez['R2_s'][0,i][0]['partial'][0][:,8] # without wheel
        pVis = adjp[i,10]
        pPupil = adjp[i,9]
        pWheel = adjp[i,8]
        meanR2Vis = np.mean(r2Vis)
        meanR2Pupil = np.mean(r2Pupil)
        meanR2Wheel = np.mean(r2Wheel)
        meanR2 = np.mean(r2)
    
        listR2.append(meanR2)
        listR2vis.append(meanR2Vis)
        listR2pupil.append(meanR2Pupil)
        listR2wheel.append(meanR2Wheel)
        listPvis.append(pVis)
        listPpupil.append(pPupil)
        listPwheel.append(pWheel)
    

    dfR2['R2'].iloc[session] = listR2
    dfR2['R2_vis'].iloc[session] = listR2vis
    dfR2['R2_pupil'].iloc[session] = listR2pupil
    dfR2['R2_wheel'].iloc[session] = listR2wheel
    dfR2['pVis'].iloc[session] = listPvis
    dfR2['pPupil'].iloc[session] = listPpupil
    dfR2['pWheel'].iloc[session] = listPwheel


dfR2 = dfR2.explode(['R2','R2_vis','R2_pupil','R2_wheel','pVis','pPupil','pWheel'])
dfR2 = dfR2.reset_index(drop=True)

dfR2['R2'] = dfR2['R2'].astype(float)
dfR2['R2_vis'] = dfR2['R2_vis'].astype(float)
dfR2['R2_pupil'] = dfR2['R2_pupil'].astype(float)
dfR2['R2_wheel'] = dfR2['R2_wheel'].astype(float)
dfR2['pVis'] = dfR2['pVis'].astype(float)
dfR2['pPupil'] = dfR2['pPupil'].astype(float)
dfR2['pWheel'] = dfR2['pWheel'].astype(float)

dfR2 = dfR2.drop(columns=['Rez','Adjp'])
dfR2 = dfR2.fillna(0)

#%% Visualize just the R2 values 

pR2 = compute.mlm_stats(dfR2,'R2').pvalues['Group[T.Exp]']

sns.histplot(dfR2,x='R2',hue='Group',log_scale=True,bins=30)
plt.title('R2 distribution')
plt.savefig(plotpath+'R2 distribution (gratings).svg',format='svg')
plt.show()

sns.catplot(dfR2,x='Group',y='R2',kind='bar', errorbar='se')
plt.title('Average R2')
plt.savefig(plotpath+'Average R2 (gratings).svg',format='svg')
plt.show()

# Plot by session

sns.catplot(dfR2,x='Group',y='R2',hue='Session',kind='point',dodge=True,errorbar='se')
plt.title('Average R2 by session')
plt.savefig(plotpath+'Average R2 by sessions.svg',format='svg')
plt.show()

#%% Calculate percentages

# Percentage of neurons that significantly encode visual stim 

dfPerc = pd.DataFrame()
dfPerc['Session'] = df['Session']
dfPerc['Group']=df['Group']
dfPerc['animalID']=df['animalID']
dfPerc['Perc'] = pd.NA


# Group by 'Session' and count the total number of cells in each session
session_counts = dfR2.groupby('Session').size()

# Count the number of cells with 'adjp' < 0.05 in each session
significant_cells = dfR2[dfR2['Adjp'] < 0.05].groupby('Session').size()

# Calculate the percentage
percentage_significant = (significant_cells / session_counts) * 100

print(percentage_significant)

for session in range(len(dfPerc)):
    dfPerc['Perc'].iloc[session] = percentage_significant[session]
    
dfPerc['Perc'] = dfPerc['Perc'].astype(float)

sns.catplot(dfPerc,x='Group',y='Perc')

#%% Calculate the relative contribution of each variable 

# First find R2(total)-R2(partial)
dfR2['dR2_vis'] = dfR2['R2']-dfR2['R2_vis']
dfR2['dR2_pupil'] = dfR2['R2']-dfR2['R2_pupil']    
dfR2['dR2_wheel'] = dfR2['R2']-dfR2['R2_wheel']

dfR2.loc[dfR2['dR2_vis']<0, 'dR2_vis'] = 0
dfR2.loc[dfR2['dR2_pupil']<0, 'dR2_pupil'] = 0
dfR2.loc[dfR2['dR2_wheel']<0, 'dR2_wheel'] = 0

# Then find dR2/ sum of the rest of dR2 for each neuron?
dfR2['dVis'] = dfR2['dR2_vis'] / sum((dfR2['dR2_vis'],dfR2['dR2_pupil'],dfR2['dR2_wheel']))
dfR2['dPupil'] = dfR2['dR2_pupil'] / sum((dfR2['dR2_vis'],dfR2['dR2_pupil'],dfR2['dR2_wheel']))
dfR2['dWheel'] = dfR2['dR2_wheel'] / sum((dfR2['dR2_vis'],dfR2['dR2_pupil'],dfR2['dR2_wheel']))

# dfR2.loc[dfR2['dVis']<0, 'dVis'] = 0
# dfR2.loc[dfR2['dPupil']<0, 'dPupil'] = 0
# dfR2.loc[dfR2['dWheel']<0, 'dWheel'] = 0

dfR2 = dfR2.fillna(0)

dfR2['dR2_vis'] = pd.to_numeric(dfR2['dVis'], errors='coerce')
dfR2['dR2_pupil'] = pd.to_numeric(dfR2['dPupil'], errors='coerce')  
dfR2['dR2_wheel'] = pd.to_numeric(dfR2['dWheel'], errors='coerce')  

mannwhitneyu(dfR2[dfR2['Group']=='Control']['dR2_wheel'],dfR2[dfR2['Group']=='Exp']['dR2_wheel'])

# np.isinf(dfR2['dR2_vis']).any()
# compute.mlm_stats(dfR2,'dR2_vis')

from scipy.stats import zscore

# Calculate z-scores
z_scores = np.abs(zscore(dfR2.select_dtypes(include=[np.number])))

# Remove rows where any z-score is above a threshold (e.g., 3)
dfFiltered = dfR2[(z_scores < 4).all(axis=1)]

#%% Melt the df to look at how each neuron's performance is affected by a predictor 

dfMelted = pd.melt(dfR2,id_vars=['Session','Group','animalID'],
                   value_vars=['R2','R2_vis','R2_pupil','R2_wheel'],
                   var_name='Predictor',value_name='Value')

dfMelted['Value'] = pd.to_numeric(dfMelted['Value'], errors='coerce')

#%% Plot the relative contributions

orderGroup = ['Control','Exp']

pVisCont = compute.mlm_stats(dfR2,'vis').pvalues['Group[T.Exp]']
pPupilCont = compute.mlm_stats(dfR2,'pupil').pvalues['Group[T.Exp]']
pWheelCont = compute.mlm_stats(dfR2,'wheel').pvalues['Group[T.Exp]']

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
# Plot visual stimulus contribution
sns.barplot(data=dfR2, x='Group', y='dR2_vis', order=orderGroup,
              linestyle='', errorbar=('se', 1), ax=axes[0])
axes[0].set_title('Visual stimulus contribution')
axes[0].text(0.5, 0.02, f'p = {pVisCont:.3e}', ha='center')

# Plot pupil contribution
sns.barplot(data=dfR2, x='Group', y='dR2_pupil', order=orderGroup,
              linestyle='', errorbar=('se', 1), ax=axes[1])
axes[1].set_title('Pupil contribution')
axes[1].text(0.5, 0.02, f'p = {pPupilCont:.3e}', ha='center')

# Plot wheel contribution
sns.barplot(data=dfR2, x='Group', y='dR2_wheel', order=orderGroup,
              linestyle='', errorbar=('se', 1), ax=axes[2])
axes[2].set_title('Wheel contribution')
axes[2].text(0.5, 0.02, f'p = {pWheelCont:.3e}', ha='center')

# Set y-axis limits for all subplots
# [ax.set_ylim(-0.01, 0.03) for ax in axes]
# Remove redundant subplots generated by sns.catplot()
plt.close(2)
plt.close(3)
plt.close(4)

# Adjust layout
plt.tight_layout()
sns.despine()

# Show the plot
# plt.savefig(plotpath+'Predictor contributions (barplot).svg')
plt.show() 

#%% Calculate how many of them have R2 >= 0.1

labelR2 = ['< 0.1', '>= 0.1']

# Calculate the counts of cells in each OSI value range for 'Control' group
nR2Control = [len(dfR2[(dfR2['Group'] == 'Control') & (dfR2['R2'] < 0.1)]),
               len(dfR2[(dfR2['Group'] == 'Control') & (dfR2['R2'] >= 0.1)])]

# Calculate the counts of cells in each OSI value range for 'Exp' group
nR2Exp = [len(dfR2[(dfR2['Group'] == 'Exp') & (dfR2['R2'] < 0.1)]),
               len(dfR2[(dfR2['Group'] == 'Exp') & (dfR2['R2'] >= 0.1)])]

plt.figure(figsize=(12, 6))
plt.rcParams['font.size'] = 15

# Create a pie chart for 'Control' group
plt.subplot(1,2,1)
plt.pie(nR2Control, labels=labelR2, autopct='%1.1f%%')
plt.title('Control R2 >= 0.1')

# Create a pie chart for 'Control' group
plt.subplot(1,2,2)
plt.pie(nR2Exp, labels=labelR2, autopct='%1.1f%%')
plt.title('Experimental R2 >= 0.1')

plt.savefig(plotpath+'GLM R2 scores larger than 0.1 (gratings).svg', format='svg', dpi=300, bbox_inches='tight')

#%% Show the percentage of cells that encode visual stimuli

labelR2 = ['< 0.05', '>= 0.05']

# Calculate the counts of cells in each OSI value range for 'Control' group
nR2Control = [len(dfR2[(dfR2['Group'] == 'Control') & (dfR2['pVis'] < 0.05)]),
               len(dfR2[(dfR2['Group'] == 'Control') & (dfR2['pVis'] >= 0.05)])]

# Calculate the counts of cells in each OSI value range for 'Exp' group
nR2Exp = [len(dfR2[(dfR2['Group'] == 'Exp') & (dfR2['pVis'] < 0.05)]),
               len(dfR2[(dfR2['Group'] == 'Exp') & (dfR2['pVis'] >= 0.05)])]

plt.figure(figsize=(12, 6))
plt.rcParams['font.size'] = 15

# Create a pie chart for 'Control' group
plt.subplot(1,2,1)
plt.pie(nR2Control, labels=labelR2, autopct='%1.1f%%')
plt.title('Control adjp < 0.05')

# Create a pie chart for 'Control' group
plt.subplot(1,2,2)
plt.pie(nR2Exp, labels=labelR2, autopct='%1.1f%%')
plt.title('Experimental adjp < 0.05')

plt.savefig(plotpath+'Proportion of cells encoding visual stimuli (gratings).svg', format='svg', dpi=300, bbox_inches='tight')

#%% Find percentages of cells encoding each parameter 

# Percentage of neurons that significantly encode visual stim 

dfPerc = pd.DataFrame()
dfPerc['Session'] = df['Session']
dfPerc['Group']=df['Group']
dfPerc['animalID']=df['animalID']
dfPerc['percVis'] = pd.NA
dfPerc['percPupil'] = pd.NA
dfPerc['percWheel'] = pd.NA


# Group by 'Session' and count the total number of cells in each session
session_counts = dfR2.groupby('Session').size()

# Count the number of cells with 'adjp' < 0.05 in each session
significant_cells = dfR2[dfR2['pVis'] < 0.05].groupby('Session').size()

# Calculate the percentage
percentage_significant = (significant_cells / session_counts) * 100

print(percentage_significant)

for session in range(len(dfPerc)):
    dfPerc['percVis'].iloc[session] = percentage_significant[session]


# Count the number of cells with 'adjp' < 0.05 in each session
significant_cells = dfR2[dfR2['pPupil'] < 0.05].groupby('Session').size()

# Calculate the percentage
percentage_significant = (significant_cells / session_counts) * 100

print(percentage_significant)

for session in range(len(dfPerc)):
    dfPerc['percPupil'].iloc[session] = percentage_significant[session]


# Count the number of cells with 'adjp' < 0.05 in each session
significant_cells = dfR2[dfR2['pWheel'] < 0.05].groupby('Session').size()

# Calculate the percentage
percentage_significant = (significant_cells / session_counts) * 100

print(percentage_significant)

for session in range(len(dfPerc)):
    dfPerc['percWheel'].iloc[session] = percentage_significant[session]

dfPerc['percVis'] = dfPerc['percVis'].astype(float)
dfPerc['percPupil'] = dfPerc['percPupil'].astype(float)
dfPerc['percWheel'] = dfPerc['percWheel'].astype(float)

dfPerc = dfPerc.fillna(0)
#%% Plot the encoding proportions 

orderGroup = ['Control','Exp']

# pVisCont = compute.mlm_stats(dfPerc,'percVis').pvalues['Group[T.Exp]']
# pPupilCont = compute.mlm_stats(dfPerc,'percPupil').pvalues['Group[T.Exp]']
# pWheelCont = compute.mlm_stats(dfPerc,'percWheel').pvalues['Group[T.Exp]']

fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
# Plot visual stimulus contribution
sns.boxplot(data=dfPerc, x='Group', y='percVis', order=orderGroup,
            ax=axes[0])
axes[0].set_title('Visual stimulus contribution')
axes[0].text(0.5, 40, f'p = {pVisCont:.3e}', ha='center')

# Plot pupil contribution
sns.boxplot(data=dfPerc, x='Group', y='percPupil', order=orderGroup,
            ax=axes[1])
axes[1].set_title('Pupil contribution')
axes[1].text(0.5, 40, f'p = {pPupilCont:.3e}', ha='center')

# Plot wheel contribution
sns.boxplot(data=dfPerc, x='Group', y='percWheel', order=orderGroup,
            ax=axes[2])
axes[2].set_title('Wheel contribution')
axes[2].text(0.5, 40, f'p = {pWheelCont:.3e}', ha='center')

# Set y-axis limits for all subplots
# [ax.set_ylim(-0.01, 0.03) for ax in axes]
# Remove redundant subplots generated by sns.catplot()
plt.close(2)
plt.close(3)
plt.close(4)

# Adjust layout
plt.tight_layout()
sns.despine()

# Show the plot
plt.savefig(plotpath+'Proportions of cells encoding different parameters.svg')
plt.show() 
