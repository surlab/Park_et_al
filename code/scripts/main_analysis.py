# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:33:23 2024

@author: jihop

# Main analysis script for analyzing preprocessed data and plotting figures

"""

# Import packages needed for running the code
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
import palettable 
from functions import extract,compute

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
dfGrat = extract.get_osi(dfGrat)
dfGrat = extract.get_visresp(dfGrat)
dfGrat = extract.get_tc(dfGrat)
dfGrat = extract.get_pref_angle(dfGrat)

# Create a new df containing only spontaneous sessions 
dfSpo = extract.extract_spo(df)
dfSpo = extract.get_dff(dfSpo)

# Create a new df containing only natural movies sessions
dfMov = extract.extract_mov(df)
dfMov = extract.get_dff(dfMov)
dfMov = extract.get_rel(dfMov)

#%% Basic visualization: Spontaneous activity heatmap

freqNeuro = 16
nFrames = 10240

for session in range(len(dfSpo)):
    name = dfSpo['Session'].iloc[session]
    dff = dfSpo['DFF'].iloc[session]
    pupil = dfSpo['Pupil'].iloc[session]
    wheel = dfSpo['Wheel'].iloc[session]
    
    # Perform Min-Max scaling to normalize data to the range [0, 1]
    normalized_dff = (dff - np.min(dff)) / (np.max(dff) - np.min(dff))

    # Calculate the time axis based on the total number of frames and the sampling frequency
    time = [i / freqNeuro for i in range(nFrames)]

    # Plot heatmap with normalized data
    plt.figure(figsize=(10,7))
    plt.imshow(normalized_dff, cmap='Greys', aspect='auto')
    plt.colorbar(label='Normalized dFF')
    plt.xticks(np.arange(0, nFrames, 640), np.arange(0, nFrames, 640) // freqNeuro)
    plt.xlabel('Time (s)')
    plt.title(f'{name}')
    # plt.savefig(plotpath+f'{name}_heatmap.svg',format='svg')
    sns.despine()
    plt.show()
    
#%% Basic visualization: Spontaneous activity average DFF trace

freqNeuro = 16
nFrames = 10240

for session in range(len(dfSpo)):
    name = dfSpo['Session'].iloc[session]
    dff = dfSpo['DFF'].iloc[session]
    
    # Perform Min-Max scaling to normalize data to the range [0, 1]
    avgDff = np.mean(dff,axis=0)

    # Calculate the time axis based on the total number of frames and the sampling frequency
    time = [i / freqNeuro for i in range(nFrames)]

    # Plot heatmap with normalized data
    plt.figure(figsize=(10,5))
    plt.plot(avgDff)

    # Adjust x-axis labels to show only every 100th frame
    plt.xticks(np.arange(0, nFrames, 640), np.arange(0, nFrames, 640) // freqNeuro)
    plt.ylim([-0.5,1])
    plt.xlabel('Time (s)')
    plt.title(f'{name}')
    # plt.savefig(plotpath+f'{name}_avgDFF.svg',format='svg')
    sns.despine()
    plt.show()
    
#%% Basic visualization: Gratings session heatmap

for session in range(len(dfGrat)):
    name = dfGrat['Session'].iloc[session]
    dff = dfGrat['DFF'].iloc[session]
    pupil = dfGrat['Pupil'].iloc[session]
    wheel = dfGrat['Wheel'].iloc[session]
    
    # Perform Min-Max scaling to normalize data to the range [0, 1]
    normalized_dff = (dff - np.min(dff)) / (np.max(dff) - np.min(dff))

    # Plot heatmap with normalized data
    plt.figure(figsize=(10,7))
    sns.heatmap(normalized_dff, cmap=palettable.cmocean.sequential.Gray_20_r.mpl_colormap, 
                vmin=0, vmax=1)
    plt.title(f'{name}')
    # plt.savefig(plotpath+f'Gratings heatmap {name}',format='png',dpi=300)
    sns.despine()
    plt.show()
    
#%% Basic visualization: Movies session heatmap

for session in range(len(dfMov)):
    name = dfMov['Session'].iloc[session]
    dff = dfMov['DFF'].iloc[session]
    pupil = dfMov['Pupil'].iloc[session]
    wheel = dfMov['Wheel'].iloc[session]
    
    # Perform Min-Max scaling to normalize data to the range [0, 1]
    normalized_dff = (dff - np.min(dff)) / (np.max(dff) - np.min(dff))

    # Plot heatmap with normalized data
    plt.figure(figsize=(10,7))
    sns.heatmap(normalized_dff, cmap=palettable.cmocean.sequential.Gray_20_r.mpl_colormap, 
                vmin=0, vmax=1)
    plt.title(f'{name}')
    plt.savefig(plotpath+f'Gratings heatmap {name}',format='png',dpi=300)
    sns.despine()
    plt.show()
    
#%% ANALYSIS

# Define the order of the groups for plotting consistency
orderGroup = ['Control','Exp']

#%% GRATINGS: Maximum response magnitudes

# Compute the maximum response magnitude of single neurons to their preferred orientation using compute.get_maxresp function

# Define parameters
tWindow = 2
freqNeuro = 16
tOff = 3
tOn = 2
nRep = 16

dfGrat = compute.get_maxresp(dfGrat,freqNeuro,tWindow,tOff,tOn)

# Explode specific columns
dfGrat = dfGrat.explode(['maxResp','OSI','VisResp','TC'])
dfGrat['OSI'] = dfGrat['OSI'].astype(float)
dfGrat['maxResp'] = dfGrat['maxResp'].astype(float)

# Filter out only visually responsive neurons
dfVisResp = dfGrat[dfGrat['VisResp']==True]
dfVisResp.drop(columns='GratStruct')

# Plot maxResp by group
plt.figure(figsize=(7, 10))
sns.boxenplot(dfVisResp,x='Group',y='maxResp',order=orderGroup,log_scale=False)
sns.despine()
plt.savefig(plotpath+'MaxResp comparison (boxenplot).svg',format='svg',dpi=300)
plt.show()

# Compute the p-value of maxResp values between control and experimental using Linear mixed effects model 
pMlmMaxResp = compute.mlm_stats(dfVisResp, 'maxResp').pvalues['Group[T.Exp]']

#%% GRATINGS: Tuning curves 

# Extract the tuning curve of each neuron and find the average tuning curve of neurons per session
listNames = []
listSessionTC = []
listSessionSEM = []
listGroup = []

for session in range(dfVisResp['Session'].nunique()):
    name = dfVisResp['Session'][session].iloc[0]
    nUnits = dfVisResp['Session'][session].shape[0]
    group = dfVisResp['Group'][session].iloc[0]
    print(f'{name}; nUnits = {nUnits}')
    
    listNormTC = []
    for n in range(nUnits):
        tc = dfVisResp['TC'][session].iloc[n]
        min_value = np.min(tc)
        max_value = np.max(tc)
        range_value = max_value - min_value
        
        # Add a duplicate of the last value to the beginning of the array
        tc = np.insert(tc, 0, tc[-1])
        
        nTC = (tc - min_value) / range_value  # Min-max normalization
        listNormTC.append(nTC)

    avgNormTC = np.mean(listNormTC, axis=0)
    semTC = sem(listNormTC,axis=0)
    
    listNames.append(name)
    listSessionTC.append(avgNormTC)
    listSessionSEM.append(semTC)
    listGroup.append(group)

# Plot the session-average tuning curves by groups 

zippedList = list(zip(listNames,listSessionTC,listSessionSEM,listGroup))

control_data = [(name, tc, semTC) for name, tc, semTC, group in zippedList if group == 'Control']
exp_data = [(name, tc, semTC) for name, tc, semTC, group in zippedList if group == 'Exp']

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))

# Plot Control group data
for name, tc, semTC in control_data:
    axes[0].plot(tc, lw=4, label=name)
    axes[0].set_ylim([0.1, 1.1])  # Adjust ylim to accommodate normalized TC values
    # Plot the average normalized TC with SEM error bars
    axes[0].errorbar(range(len(tc)), tc, yerr=semTC*1, fmt='-o', lw=3)

axes[0].set_title('Control Group')
axes[0].set_xlabel('Angle (degrees)')
axes[0].set_ylabel('Normalized Response')
axes[0].legend(loc='upper right')

# Plot Experimental group data
for name, tc, semTC in exp_data:
    axes[1].plot(tc, lw=4, label=name)
    axes[1].set_ylim([0.1, 1.1])  # Adjust ylim to accommodate normalized TC values
    # Plot the average normalized TC with SEM error bars
    axes[1].errorbar(range(len(tc)), tc, yerr=semTC*1, fmt='-o', lw=3)

axes[1].set_title('Experimental Group')
axes[1].set_xlabel('Angle (degrees)')
axes[1].set_ylabel('Normalized Response')
axes[1].legend(loc='upper right')

# Define the desired angles in degrees
angles_deg = [-180, -135, -90, -45, 0, 45, 90, 135, 180]

# Set the x-axis tick positions and labels for both subplots
for ax in axes:
    ax.set_xticks(range(len(avgNormTC)))
    ax.set_xticklabels(angles_deg)

sns.despine()
plt.savefig(plotpath+'Average tuning curves of all sessions.svg', dpi=300)
plt.show()

#%% GRATINGS: Orientation Selectivity Index (OSI) 

# Compute the p-value of maxResp values between control and experimental using Linear mixed effects model
pMlmOSI = compute.mlm_stats(dfVisResp, 'osi').pvalues['Group[T.Exp]']
                         
# Plot the OSI values of each group
sns.catplot(dfVisResp,x='Group',y='OSI',order=orderGroup,
            dodge=True,kind='boxen',height=6,aspect=1)
plt.title('OSI distribution of all neurons')
# plt.savefig(plotpath+'OSI distribution of all neurons.svg',dpi=300)
plt.show()

#%% MOVIES: Reliability indices

dfMov = dfMov.explode(['Rel'])
dfMov['Rel'] = dfMov.Rel.astype(float)
# Compute the p-value of reliability indices between control and experimental 
pMlmRel = compute.mlm_stats(dfMov,'rel').pvalues['Group[T.Exp]']

# Plot
sns.catplot(dfMov,x='Group',y='Rel',order=orderGroup,kind='boxen',height=10,aspect=1)
plt.title('Rel indices by neurons')
# plt.savefig(plotpath+'Rel indices by neurons.svg',dpi=300)
plt.show()
