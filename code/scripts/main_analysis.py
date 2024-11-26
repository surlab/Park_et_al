# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:33:23 2024

@author: jihop

# Basic analysis using preprocessed outputs (OSI, DSI, Rel, etc....)
# For visualizing DFFs

"""

# Import the functions that I need 
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp, sem, wilcoxon
import palettable 
import pingouin as pg
from scipy.interpolate import interp1d
from functions import extract,compute,plot,glm
# from sklearn.cluster import KMeans

plt.rcParams['figure.max_open_warning'] = 0


#%% Define paths

# Windows
datapath = os.path.join('G:','My Drive','mrcuts','data','master','')
savepath = os.path.join('G:','My Drive','mrcuts','analysis','')
plotpath = os.path.join('G:','My Drive','mrcuts','analysis','plots','new','')
# plotpath = os.path.join('D:','analysis','mrcuts','plots','') # New plot path for saving plots on the computer (D:Data)
# plotpath = os.path.join('G:','My Drive','mrcuts','analysis','plots','new_tam','')

# MAC
datapath = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','data','master','')
savepath = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','')
plotpath = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','plots','new','')

#%% Load the data

dfMaster = extract.load_data(datapath)

#%% Specify the animalIDs, dates,conditions to look at 

orderGroup = ['Control','Exp']
orderCondition = ['Pre','Post']

# Just mrcuts07 + constitutive
listID = ['mrcuts07','mrcuts24','mrcuts25','mrcuts26',
          'mrcuts27','mrcuts28','mrcuts29','mrcuts30']

# Just TAM group
# listID = ['an317','mrcuts13','mrcuts15','mrcuts16']

# Filter out the df
df = dfMaster[dfMaster['animalID'].isin(listID)].reset_index(drop=True)

# Filter out the problematic sessions based on heatmaps (ROI labeling seems to be off)
sessions_to_drop = ['230809_mrcuts28_fov4_mov-000','221123_mrcuts07_fov1_grat-000','230804_mrcuts26_fov3_mov-000']
df = df[~df['Session'].isin(sessions_to_drop)].reset_index(drop=True)

# Drop certain sessions
dates_to_remove = ['221214', '221216', '221222']
df = df[~df['Date'].isin(dates_to_remove)].reset_index(drop=True)
df['Group'] = df['animalID'].apply(extract.get_group)
# Display the resulting DataFrame
print(df)

# Assign conditions by dates 

# df['Condition'] = df['Date'].apply(extract.get_condition)
# df = df.dropna(subset=['Condition']).reset_index(drop=True)

#%% Extract only the data I want (ex: spo, grat, mov)

dfGrat = extract.extract_grat(df)
dfGrat = extract.get_dff(dfGrat)
dfGrat = extract.get_osi(dfGrat)
dfGrat = extract.get_visresp(dfGrat)
dfGrat = extract.get_tc(dfGrat)
# dfGrat = extract.get_vmfit(dfGrat)
dfGrat = extract.get_pref_angle(dfGrat)

dfSpo = extract.extract_spo(df)
dfSpo = extract.get_dff(dfSpo)

dfMov = extract.extract_mov(df)
dfMov = extract.get_dff(dfMov)
dfMov = extract.get_rel(dfMov)

# Save the DFs as csv files if needed

# dfGrat.to_csv(savepath+'dfGrat.csv')
# dfSpo.to_csv(savepath+'dfSpo.csv')

#%% Plot heatmaps of spontaneous activity

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
    # sns.heatmap(normalized_dff, cmap=palettable.cmocean.sequential.Gray_20_r.mpl_colormap, 
    #             vmin=0, vmax=1, xticklabels=20)
    # sns.heatmap(normalized_dff, cmap="YlOrBr", 
    #             vmin=0, vmax=1, xticklabels=20)
    plt.imshow(normalized_dff, cmap='Greys', aspect='auto')
    plt.colorbar(label='Normalized dFF')
    # Adjust x-axis labels to show only every 100th frame
    plt.xticks(np.arange(0, nFrames, 640), np.arange(0, nFrames, 640) // freqNeuro)
    plt.xlabel('Time (s)')
    plt.title(f'{name}')
    plt.savefig(plotpath+f'{name}_heatmap.svg',format='svg')
    sns.despine()
    plt.show()
    
#%% Plot avg DFF of all neurons during spontaneous activity

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
    plt.savefig(plotpath+f'{name}_avgDFF.svg',format='svg')
    sns.despine()
    plt.show()
    
    
#%% Plot heatmaps of gratings

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
    
#%% Plot heatmaps of movies

for session in range(len(dfMov)):
    name = dfMov['Session'].iloc[session]
    dff = dfMov['DFF'].iloc[session]
    pupil = dfMov['Pupil'].iloc[session]
    wheel = dfMov['Wheel'].iloc[session]
    
    # Perform Min-Max scaling to normalize data to the range [0, 1]
    normalized_dff = (dff - np.min(dff)) / (np.max(dff) - np.min(dff))

    # Plot heatmap with normalized data
    # plt.figure(figsize=(10,7))
    # sns.heatmap(normalized_dff, cmap=palettable.cmocean.sequential.Gray_20_r.mpl_colormap, 
    #             vmin=0, vmax=1)
    # plt.title(f'{name}')
    # plt.savefig(plotpath+f'Gratings heatmap {name}',format='png',dpi=300)
    # sns.despine()
    # plt.show()
    
    ymax = np.max(normalized_dff)
    ymin = np.min(normalized_dff)
    
    for n in range(normalized_dff.shape[0]):
        
        nDff = normalized_dff[n,:]
        
        plt.figure(figsize=(15,5))
        plt.plot(nDff)
        plt.ylim([ymin,ymax])
        plt.title(f'{name} n={n+1}')
        sns.despine()
        # plt.savefig(plotpath+f'{name}_n={n+1}.svg',format='svg',dpi=300)
        plt.show()


#%% Plot dff traces of single neurons for an example session

# name = '230809_mrcuts28_fov4_mov-000'
# name = '221123_mrcuts07_fov1_grat-000'

# dff = dfGrat[dfGrat['Session']==name]['DFF'].iloc[0]

# for n in range(len(dff)):

#     dffRaw = dff[n,:]
#     dffMin = np.min(dff)
#     dffMax = np.max(dff)
#     # Plot heatmap with normalized data
#     plt.figure(figsize=(15,7))
#     plt.plot(dffRaw,color='darkgreen',lw=0.5)
#     plt.ylim([dffMin,dffMax])
#     plt.title(f'Gratings DFF n={n}')
#     # plt.savefig(plotpath+f'Gratings DFF trace n={n}.svg',format='svg',dpi=300)
#     sns.despine()
#     plt.show()
    
#%% Plot dff traces of all neurons for each session

for session in range(len(dfGrat)):
    dff = dfGrat['DFF'].iloc[session]
    name = dfGrat['Session'].iloc[session]
    dffMin = np.min(dff)
    dffMax = np.max(dff)

    plt.figure(figsize=(15,7))
    for n in range(dff.shape[0]):
        nDff = dff[n,:]
        plt.plot(nDff,lw=0.5)
        plt.ylim([dffMin,dffMax])
        plt.title(f'{name}_all_traces')
        # plt.savefig(plotpath+f'Gratings DFF trace n={n}.svg',format='svg',dpi=300)
        sns.despine()
    plt.show()
 
#%% Calculate correlation between neuronal activity and pupil? 

n = 2

dfTest = dfSpo.iloc[n]
dff = dfTest.DFF
pupil = dfTest.Pupil
nUnits, nFrames = dff.shape

# Downsample the pupil data

idxOri = np.arange(pupil.size)

# Create an array of indices for the desired number of frames
idxNew = np.round(np.linspace(0, len(idxOri) - 1, nFrames)).astype(int) 

# Use linear interpolation to interpolate 'pupil' data at desired indices
pupilNew = interp1d(idxOri, pupil)(idxNew)

# Calculate correlation for each neuron
corrCoeff = []
for i in range(nUnits):
    corrCoeff.append(np.corrcoef(dff[i], pupilNew)[0, 1])

corrCoeff = np.array(corrCoeff)

# # Cluster the data
# num_clusters = 3  # You can choose the number of clusters
# kmeans = KMeans(n_clusters=num_clusters, random_state=0)
# clusters = kmeans.fit_predict(corrCoeff.reshape(-1, 1))

# # Create a dictionary to store cluster indices and their corresponding correlation coefficients
# cluster_data = {i: [] for i in range(num_clusters)}
# for i, cluster_idx in enumerate(clusters):
#     cluster_data[cluster_idx].append(corrCoeff[i])

# # Plot each cluster separately
# plt.figure(figsize=(8, 6))
# for i in range(num_clusters):
#     plt.scatter([i]*len(cluster_data[i]), cluster_data[i], label=f'Cluster {i+1}')
# plt.title('Clustered Correlation Coefficients of Each Neuron')
# plt.xlabel('Cluster Index')
# plt.ylabel('Correlation Coefficient')
# plt.legend()
# plt.show()

# Bar Plot
plt.figure(figsize=(8, 6))
plt.bar(range(len(corrCoeff)), corrCoeff)
plt.title('Correlation Coefficients of Each Neuron')
plt.xlabel('Neuron Index')
plt.ylabel('Correlation Coefficient')
plt.show()

# Box Plot
plt.figure(figsize=(8, 6))
plt.boxplot(corrCoeff)
plt.title('Distribution of Correlation Coefficients')
plt.ylabel('Correlation Coefficient')
plt.show()

# Violin Plot
plt.figure(figsize=(8, 6))
plt.violinplot(corrCoeff, vert=False)
plt.title('Distribution of Correlation Coefficients')
plt.xlabel('Correlation Coefficient')
plt.show()

sns.ecdfplot(data=corrCoeff)


#%% Neuronal activity-Pupil correlation 

listPupilCorr = []

for session in range(len(dfSpo)):
    dff = dfSpo['DFF'].iloc[session]
    name = dfSpo['Session'].iloc[session]
    pupil = dfSpo['Pupil'].iloc[session]
    nUnits, nFrames = dff.shape
    
    idxOri = np.arange(pupil.size)
    # Create an array of indices for the desired number of frames
    idxNew = np.round(np.linspace(0, len(idxOri) - 1, nFrames)).astype(int) 
    # Use linear interpolation to interpolate 'pupil' data at desired indices
    pupilNew = interp1d(idxOri, pupil)(idxNew)

    # Calculate correlation for each neuron
    corrCoeff = []
    for i in range(nUnits):
        corrCoeff.append(np.corrcoef(dff[i], pupilNew)[0, 1])

    corrCoeff = np.array(corrCoeff)
    
    listPupilCorr.append(corrCoeff)
    
    # plt.figure(figsize=(8,5))
    # sns.ecdfplot(data=corrCoeff)
    # plt.title(f'{name}')
    
dfSpo['pupilCorr'] = listPupilCorr
    
dfSpo = dfSpo.explode(['pupilCorr'])
dfSpo['pupilCorr'] = dfSpo.pupilCorr.astype(float)
sns.displot(dfSpo,x='pupilCorr',hue='Group',kind='ecdf')
compute.mlm_stats(dfSpo,'pupilCorr')

#%% Neuronal activity-Wheel correlation 

listWheelCorr = []

for session in range(len(dfSpo)):
    dff = dfSpo['DFF'].iloc[session]
    name = dfSpo['Session'].iloc[session]
    wheel = dfSpo['Wheel'].iloc[session]
    nUnits, nFrames = dff.shape
    
    idxOri = np.arange(wheel.size)
    # Create an array of indices for the desired number of frames
    idxNew = np.round(np.linspace(0, len(idxOri) - 1, nFrames)).astype(int) 
    # Use linear interpolation to interpolate 'pupil' data at desired indices
    wheelNew = interp1d(idxOri, wheel)(idxNew)

    # Calculate correlation for each neuron
    corrCoeff = []
    for i in range(nUnits):
        corrCoeff.append(np.corrcoef(dff[i], wheelNew)[0, 1])

    corrCoeff = np.array(corrCoeff)
    
    listWheelCorr.append(corrCoeff)
    
    # plt.figure(figsize=(8,5))
    # sns.ecdfplot(data=corrCoeff)
    # plt.title(f'{name}')
    
dfSpo['wheelCorr'] = listWheelCorr
    
dfSpo = dfSpo.explode(['wheelCorr'])
dfSpo['wheelCorr'] = dfSpo.wheelCorr.astype(float)
sns.displot(dfSpo,x='wheelCorr',hue='Group',kind='ecdf')
compute.mlm_stats(dfSpo,'wheelCorr')
#%% RELIABILITY INDICES


dfMov = dfMov.explode(['Rel'])
dfMov['Rel'] = dfMov.Rel.astype(float)

pMlmRel = compute.mlm_stats(dfMov,'rel').pvalues['Group[T.Exp]']
pRel = mannwhitneyu(dfMov[dfMov['Group']=='Control']['Rel'],dfMov[dfMov['Group']=='Exp']['Rel'])

sns.catplot(dfMov,x='Group',y='Rel',order=orderGroup,kind='boxen',height=10,aspect=1)
plt.title('Rel indices by neurons')
# plt.savefig(plotpath+'Rel indices by neurons.svg',dpi=300)
plt.show()

sns.catplot(dfMov,x='Group',y='Rel',order=orderGroup,hue='Session',dodge=True,kind='point',height=6,aspect=1,errorbar=('se',1))
plt.title('Rel indices by sessions')
# plt.savefig(plotpath+'Rel indices by sessions.svg',dpi=300)
plt.show()

# sns.catplot(dfMov,x='Condition',y='Rel',order=orderCondition,kind='boxen',height=10,aspect=1)
# plt.title('Rel indices by neurons')
# # plt.savefig(plotpath+'Rel indices by neuronns.svg',dpi=300)
# plt.show()

# sns.catplot(dfMov,x='Condition',y='Rel',order=orderCondition,hue='Session',dodge=True,kind='point',height=6,aspect=1,errorbar=('se',1))
# plt.title('Rel indices by sessions')
# # plt.savefig(plotpath+'Rel indices by sessions.svg',dpi=300)
# plt.show()

# sns.catplot(dfMov,x='Condition',y='Rel',order=orderCondition,hue='animalID',dodge=True,kind='point',height=6,aspect=1,errorbar=('se',1))
# plt.title('Rel indices by animals')
# plt.show()
 
#%% MAXIUMUM RESPONSE MAGNITUDES

# For current data
tWindow = 2
freqNeuro = 16
tOff = 3
tOn = 2
nRep = 16

dfGrat = compute.get_maxresp(dfGrat,freqNeuro,tWindow,tOff,tOn)

# Explode 'maxResp' to plot
dfGrat = dfGrat.explode(['maxResp','OSI','VisResp','TC'])

dfGrat['OSI'] = dfGrat['OSI'].astype(float)
dfGrat['maxResp'] = dfGrat['maxResp'].astype(float)

# Filter out only visually responsive neurons

dfVisResp = dfGrat[dfGrat['VisResp']==True]
dfVisResp.drop(columns='GratStruct')

# dfVisResp.to_csv(savepath+'dfVisResp.csv')

#%% Plot TC of visually responsive neurons and their average for each session

for session in range(dfVisResp['Session'].nunique()):
    name = dfVisResp['Session'][session].iloc[0]
    nUnits = dfVisResp['Session'][session].shape[0]
    print(f'{name}; nUnits = {nUnits}')
    
    listNormTC = []
    plt.figure(figsize=(10,7))
    for n in range(nUnits):
        tc = dfVisResp['TC'][session].iloc[n]
        min_value = np.min(tc)
        max_value = np.max(tc)
        range_value = max_value - min_value
        
        # Add a duplicate of the last value to the beginning of the array
        tc = np.insert(tc, 0, tc[-1])
        
        nTC = (tc - min_value) / range_value  # Min-max normalization
        listNormTC.append(nTC)
        plt.plot(nTC, color='gray')
        plt.ylim([-0.1, 1.4])  # Adjust ylim to accommodate normalized TC values
        sns.despine()
    
    avgNormTC = np.mean(listNormTC, axis=0)
    semTC = sem(listNormTC,axis=0)
    plt.plot(avgNormTC, lw=5, color='red')
    
    # Plot the average normalized TC with SEM error bars
    plt.errorbar(range(len(avgNormTC)), avgNormTC, yerr=semTC*2, 
                 fmt='-o', color='red', lw=3, label='Average TC with SEM')
    
    # Define the desired angles in degrees
    angles_deg = [-180, -135, -90, -45, 0, 45, 90, 135, 180]

    # Set the x-axis tick positions and labels
    plt.xticks(range(len(avgNormTC)), angles_deg)
    plt.title(f'{name} tuning curves (min-max normalized)')
    # plt.savefig(plotpath+f'{name}_tuning_curves.svg',format='svg',dpi=300)
    plt.show()
    
#%% Calculate the average TC of visually responsive neurons from each session

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

#%% Plot the average TCs by groups 

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

#%% Plot the VM fit of visually responsive neurons        

for session in range(dfVisResp['Session'].nunique()):
    name = dfVisResp['Session'][session].iloc[0]
    nUnits = dfVisResp['Session'][session].shape[0]
    print(f'{name}; nUnits = {nUnits}')
    
    plt.figure(figsize=(10,7))
    for n in range(nUnits):
        vm = dfVisResp['VM'][session].iloc[n]
        plt.plot(vm, color='gray')
        plt.ylim([-0.5, 2])
        sns.despine()
    
    avgVM = np.mean(dfVisResp['VM'][session].iloc[0], axis=0)
    plt.plot(avgVM, lw=5, color='red')
    
    # Get the x-tick positions after plotting the data
    idxTicks = plt.xticks()[0]
    
    # Define the desired angles in degrees
    angles_deg = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    # Set the x-tick positions and labels
    plt.xticks(idxTicks, angles_deg)
    
    plt.title(f'{name} tuning curves')
    plt.show()

#%% Plot OSI

sns.catplot(dfVisResp,x='Group',y='OSI',order=orderGroup,kind='box')
sns.histplot(dfVisResp,x='OSI',hue='Group',log_scale=True)

pOSI = mannwhitneyu(x=dfVisResp[dfVisResp['Group']=='Control']['OSI'],
              y=dfVisResp[dfVisResp['Group']=='Exp']['OSI'])[1]

pMlmOSI = compute.mlm_stats(dfVisResp, 'osi').pvalues['Group[T.Exp]']
                         
sns.catplot(dfVisResp,x='Group',y='OSI',order=orderGroup,hue='animalID',
            dodge=True,kind='box',height=6,aspect=1)
plt.title('OSI distribution by animal (box)')
plt.text(1,np.max(dfVisResp['OSI']),f'p value = {pMlmOSI:0.3e}')
# plt.savefig(plotpath+'OSI distribution by animal (box).svg',dpi=300)
plt.show()

sns.catplot(dfVisResp,x='Group',y='OSI',order=orderGroup,
            dodge=True,kind='boxen',height=6,aspect=1)
plt.title('OSI distribution of all neurons')
# plt.savefig(plotpath+'OSI distribution of all neurons.svg',dpi=300)
plt.show()

x = 0.3

plot.plot_OSI_greater_than_x(dfVisResp,x,plotpath)


#%%% Plot maxResp

plt.figure(figsize=(7, 10))
# sns.barplot(dfVisResp,x='Group',y='maxResp',order=orderGroup)
sns.boxenplot(dfVisResp,x='Group',y='maxResp',order=orderGroup,log_scale=False)
# sns.violinplot(dfVisResp,x='Group',y='maxResp')
# sns.stripplot(dfVisResp,x='Group',y='maxResp',order=orderGroup,size=3)
# sns.swarmplot(dfVisResp,x='Group',y='maxResp',order=orderGroup,size=2)
# plt.show()
# sns.pointplot(dfVisResp,x='Group',y='maxResp',order=orderGroup,hue='Session',
              # legend=True,errorbar=None)
sns.despine()
plt.savefig(plotpath+'MaxResp comparison (boxplot).svg',format='svg',dpi=300)
plt.show()

plt.figure(figsize=(7,10))
sns.violinplot(dfVisResp,x='Group',y='maxResp',order=orderGroup,inner='point')
plt.show()

sns.catplot(dfVisResp,x='Group',y='maxResp',order=orderGroup,hue='animalID',
              dodge=True,kind='strip',height=10,aspect=1)
sns.despine()
# plt.savefig(plotpath+'MaxResp comparison by animal.svg',format='svg',dpi=300)
plt.show()

# pMaxResp = mannwhitneyu(x=dfVisResp[dfVisResp['Group']=='Control']['maxResp'],
#              y=dfVisResp[dfVisResp['Group']=='Exp']['maxResp'])[1]

pMlmMaxResp = compute.mlm_stats(dfVisResp, 'maxResp').pvalues['Group[T.Exp]']
#%% Extract Spks data 

# spkspath = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','data','preprocessed','')
spkspath = os.path.join('G:','My Drive','mrcuts','data','preprocessed','')

dfSpks = extract.load_spks_data(spkspath)

#%% Filter out only the sessions I want (same as above)

# Set the keyword to filter out sessions (ex. 'spo', 'grat')
keyword = 'spo'
df = extract.get_session(dfSpks,keyword)

# Filter out the df
df = df[df['animalID'].isin(listID)].reset_index(drop=True)
df['Group'] = df['animalID'].apply(extract.get_group)
# Drop certain sessions
dates_to_remove = ['221214', '221216', '221222']
df = df[~df['Date'].isin(dates_to_remove)].reset_index(drop=True)
# Filter out the problematic sessions based on heatmaps (ROI labeling seems to be off)
df = df[~df['Session'].isin(sessions_to_drop)].reset_index(drop=True)

sessions_to_drop = ['230117_mrcuts13_fov1_spo-000','230211_mrcuts13_fov1_grat-000','230219_mrcuts15_fov1_spo-000']
df = df[~df['Session'].isin(sessions_to_drop)].reset_index(drop=True)

# Assign conditions by dates 
# df['Condition'] = df['Date'].apply(extract.get_condition)
# df = df.dropna(subset=['Condition']).reset_index(drop=True)

#%% FIRING RATES

# For current data
tDur = 640  # Total duration in seconds
freqNeuro = 16  # Neuro data sampling frequency (Hz)
timeBin = 0.1  # Time bin in seconds (100 ms)
threshold = 5 

df = compute.get_fr(df,tDur,freqNeuro,timeBin,threshold)
dfFR = df.explode(['FR'])

#%% Plot the FRs

dfFR['FR'] = dfFR['FR'].astype(float)

sns.catplot(dfFR,x='Group',y='FR',order=orderGroup,
            dodge=True,kind='point',linestyle='',errorbar='se')
plt.title('Firing rates',y=1.2,fontsize=15)
plt.savefig(plotpath+'Firing rates (pointplot).svg')
plt.show()

dv = 'FR'
plottype = 'bar'
plot.plot_with_pval(dfFR,dv,plottype,orderGroup)

plt.figure(figsize=(5,5))
# sns.pointplot(dfFR,x='Group',y='FR',order=orderGroup,hue='Session',
#             dodge=False,legend=True,errorbar=None)
sns.barplot(dfFR,x='Group',y='FR',order=orderGroup)
plt.ylim([1.3,1.95])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
sns.despine()
plt.savefig(plotpath+'Firing rates average.svg')
plt.show()

sns.pointplot(dfFR,x='animalID',y='FR',hue='Session',
            dodge=False,legend=False,errorbar=None)
plt.ylim([1.3,1.95])
plt.savefig(plotpath+'Firing rates by sessions.svg')
plt.show()

pFR = ks_2samp(dfFR[dfFR['Group']=='Control']['FR'], dfFR[dfFR['Group']=='Exp']['FR']).pvalue
pMlmFR = compute.mlm_stats(dfFR, 'fr').pvalues['Group[T.Exp]']

plt.figure(figsize=(10,7))
sns.ecdfplot(dfFR,x='FR',hue='Group')
plt.text(3,1.1,f"pval={pFR:0.3E}")
sns.despine()
plt.title('Firing rates',y=1.2,fontsize=15)
plt.savefig(plotpath+'Firing rates (ECDF).svg')
plt.show()

sns.displot(dfFR,x='FR',hue='animalID',col='Group',kind='ecdf',log_scale=True)
plt.savefig(plotpath+'Firing rates (ECDF) by animal.svg')
plt.show()

sns.catplot(dfFR,x='Condition',y='FR',order=orderCondition,kind='point',hue='animalID')
plt.title('FR comparison by animal')

#%% PAIRWISE CORRELATIONS

# For current data
sessDur = 640
freqNeuro = 16
nShuffles = 100

df = compute.get_pairwise_corr(df,freqNeuro,sessDur,nShuffles)
dfCorr = df.explode(['Coeff','CoeffShuffled'])

# Save
dfCorr.to_csv(savepath+'Spo_pairwise_correlations.csv')

# Load
dfCorr = pd.read_csv(savepath + 'Spo_pairwise_correlations.csv', index_col=0)

#%% Plot the pairwise correlations

dfCorr['Coeff'] = dfCorr['Coeff'].astype(float)
dfCorr['CoeffShuffled'] = dfCorr['CoeffShuffled'].astype(float)

dv = 'Coeff'
plottype = 'bar'
plot.plot_with_pval(dfCorr,dv,plottype,orderGroup)

pCorr = ks_2samp(dfCorr[dfCorr['Group']=='Control']['Coeff'], dfCorr[dfCorr['Group']=='Exp']['Coeff']).pvalue
pCorrShuffled = ks_2samp(dfCorr[dfCorr['Group']=='Control']['CoeffShuffled'], dfCorr[dfCorr['Group']=='Exp']['CoeffShuffled']).pvalue

pMlmSpoCorr = compute.mlm_stats(dfCorr, 'coeff').pvalues['Group[T.Exp]']

# Histogram (separate)

fig, axes = plt.subplots(2,1,sharex=True,sharey=True,figsize=(12,5))

sns.histplot(dfCorr[dfCorr['Group']=='Control'],x='Coeff',color='green',log_scale=True,ax=axes[0])
sns.histplot(dfCorr[dfCorr['Group']=='Control'],x='CoeffShuffled',color='gray',log_scale=True,ax=axes[0])
axes[0].set_xlim([0,0.5])
sns.despine()
axes[0].set_title('Control neuron-to-neuron correlation with shuffled')

sns.histplot(dfCorr[dfCorr['Group']=='Exp'],x='Coeff',color='orange',log_scale=True,ax=axes[1])
sns.histplot(dfCorr[dfCorr['Group']=='Exp'],x='CoeffShuffled',color='gray',log_scale=True,ax=axes[1])
sns.despine()
axes[1].set_title('Experimental neuron-to-neuron correlation with shuffled')

# Histogram (together)

plt.figure(figsize=(10,4))

sns.histplot(dfCorr[dfCorr['Group']=='Control'],x='Coeff',
             bins=60,color='steelblue',log_scale=True)
sns.histplot(dfCorr[dfCorr['Group']=='Control'],x='CoeffShuffled',
             bins=60,color='gray',log_scale=True)

sns.histplot(dfCorr[dfCorr['Group']=='Exp'],x='Coeff',
             bins=60,color='lightpink',log_scale=True)
sns.histplot(dfCorr[dfCorr['Group']=='Exp'],x='CoeffShuffled',
             bins=60,color='lightgray',log_scale=True)
sns.despine()
plt.xlim([0.000001,0.5])

[x1,x2] = plt.gca().get_xlim()
[y1,y2] = plt.gca().get_ylim()

plt.text(0.5, 0.95, f'pActual={pCorr:0.3f}\npShuffled={pCorrShuffled:0.3f}', ha='right', va='top', transform=plt.gca().transAxes)

plt.title('Distribution of neuron-to-neuron correlation coefficients')
plt.legend(title='Groups', labels=['Control', 'Control Shuffled', 'Experimental', 'Experimental Shuffled'])
plt.savefig(plotpath+'Distribution of neuron-to-neuron correlation coefficients.svg')
plt.show()

plt.figure(figsize=(10,7))
sns.ecdfplot(dfCorr,x='Coeff',hue='Group',log_scale=True)
sns.ecdfplot(dfCorr,x='CoeffShuffled',hue='Group',log_scale=True)
sns.despine()
plt.savefig(plotpath+'ECDF plot of neuron-to-neuron correlation coefficients.svg')
plt.show()

sns.catplot(dfCorr,x='Group',y='Coeff',hue='Session',dodge=True,kind='point',height=6,aspect=1)
plt.title('Avg neuron-to-neuron coeff by session')
plt.savefig(plotpath+'Neuron-to-neuron correlation coefficients by sessions.svg')
plt.show()

plt.figure(figsize=(10,4))

sns.histplot(dfCorr[dfCorr['Condition']=='Pre'],x='Coeff',
             bins=60,color='steelblue',log_scale=True)
sns.histplot(dfCorr[dfCorr['Condition']=='Pre'],x='CoeffShuffled',
             bins=60,color='gray',log_scale=True)

sns.histplot(dfCorr[dfCorr['Condition']=='Post'],x='Coeff',
             bins=60,color='lightpink',log_scale=True)
sns.histplot(dfCorr[dfCorr['Condition']=='Post'],x='CoeffShuffled',
             bins=60,color='lightgray',log_scale=True)
sns.despine()
plt.xlim([0.000001,0.5])
#%% Now let's look at movies Spks data

stimType = 'mov'

df = extract.get_session(dfSpks,stimType)

# Filter out the df
df = df[df['animalID'].isin(listID)].reset_index(drop=True)
df['Group'] = df['animalID'].apply(extract.get_group)
# Drop certain sessions
dates_to_remove = ['221214', '221216', '221222']
df = df[~df['Date'].isin(dates_to_remove)].reset_index(drop=True)

# Assign conditions by dates 
df['Condition'] = df['Date'].apply(extract.get_condition)
df = df.dropna(subset=['Condition']).reset_index(drop=True)

#%% Get SIGNAL & NOISE correlations (MOV)

freqNeuro = 16
tOff = 3
tOn = 2
nTrials = 32
nFrames = 8704
nStim = 7
nShuffles = 100

df = compute.get_noise_corr(df,freqNeuro,tOff,tOn,nTrials,nFrames,nStim,nShuffles,plotpath)
df = compute.get_signal_corr(df,freqNeuro,tOff,tOn,nTrials,nFrames,nStim,nShuffles,plotpath)
dfCorr = df.explode(['NoiseCoeff','NoiseCoeffShuffled','SignalCoeff','SignalCoeffShuffled'])

dfCorr.to_csv(savepath+'Noise and signal correlations (mov).csv')

dfCorr = pd.read_csv(savepath+'Noise and signal correlations (mov).csv')
dfCorr = dfCorr.drop('Unnamed: 0',axis=1)

# Drop a probablematic session 
session_to_drop = '230809_mrcuts28_fov4_mov-000'
sessions_to_drop = ['230809_mrcuts28_fov4_mov-000','230804_mrcuts26_fov3_mov-000']
dfCorr = dfCorr[~dfCorr['Session'].isin(sessions_to_drop)]

dfCorr = dfCorr.drop(columns=['Spks','zSpks'])

#%% Plot noise correlations

dfCorr['NoiseCoeff'] = dfCorr['NoiseCoeff'].astype(float)
dfCorr['NoiseCoeffShuffled'] = dfCorr['NoiseCoeffShuffled'].astype(float)

sns.catplot(dfCorr,x='Group',y='NoiseCoeff',order=orderGroup,hue='Session',
            dodge=True,kind='point',errorbar=None)
plt.title('Noise correlations (mov)')
plt.savefig(plotpath+'Noise correlations (mov) by session.svg',format='svg',dpi=300)
plt.show()

plt.figure(figsize=(10,10))
sns.ecdfplot(dfCorr,x='NoiseCoeff',hue='Group',log_scale=True)
plt.title('Noise correlations (mov) ecdf')
plt.show()

dv = 'NoiseCoeff'
plottype = 'bar'
plot.plot_with_pval(dfCorr,dv,plottype,orderGroup)

pMlmNoise = compute.mlm_stats(dfCorr, 'noise').pvalues['Group[T.Exp]']
pKSNoise = ks_2samp(dfCorr[dfCorr['Group']=='Control']['NoiseCoeff'],dfCorr[dfCorr['Group']=='Exp']['NoiseCoeff']).pvalue

sns.catplot(dfCorr,x='Group',y='NoiseCoeff',order=orderGroup,height=6,aspect=1,kind='bar')
plt.ylim([0,0.01])
plt.title('Noise Correlation coefficients (by neuron)')
# plt.savefig(plotpath+'Noise Correlation coefficients (by neuron).svg',dpi=300)
plt.show()

plt.figure(figsize=(10,7))
sns.ecdfplot(dfCorr,x='NoiseCoeff',hue='Group',log_scale=True)
sns.ecdfplot(dfCorr,x='NoiseCoeffShuffled',hue='Group',log_scale=True)
sns.despine()
# plt.savefig(plotpath+'ECDF plot of noise correlation coefficients.svg')
plt.show()

#%% BOOTSTRAP for fair comparison

# Find the minimum number of coefficients in any session
min_coeffs = dfCorr.groupby('Session').size().min()

# Function to bootstrap and randomly select coefficients
def bootstrap_sample(df, n_samples):
    return df.sample(n=n_samples, replace=True)

# Apply bootstrapping for each session
bootstrapped_data = []

for session in dfCorr['Session'].unique():
    session_data = dfCorr[dfCorr['Session'] == session]
    bootstrapped_data.append(bootstrap_sample(session_data, min_coeffs))
    
# Concatenate bootstrapped samples
dfSampled = pd.concat(bootstrapped_data, ignore_index=True)

# Check the bootstrapped DataFrame
print(dfSampled)

#%% BOOTSTRAP with iterations

# Function to perform multiple bootstrap iterations for each session
def bootstrap_multiple_iterations(df, min_coeffs, n_iterations):
    bootstrapped_data = []

    for session in df['Session'].unique():
        session_data = df[df['Session'] == session]
        
        for _ in range(n_iterations):
            bootstrapped_data.append(bootstrap_sample(session_data, min_coeffs))
    
    return pd.concat(bootstrapped_data, ignore_index=True)

# Number of iterations for bootstrapping
n_iterations = 100

# Apply bootstrapping for each session
bootstrapped_data = []

for session in dfCorr['Session'].unique():
    session_data = dfCorr[dfCorr['Session'] == session]
    bootstrapped_data.append(bootstrap_multiple_iterations(session_data, min_coeffs, n_iterations))

# Apply bootstrapping with multiple iterations for each session
dfSampled = pd.concat(bootstrapped_data, ignore_index=True)

# Check the bootstrapped DataFrame
print(dfSampled)

#%% Plot signal correlations

dfCorr['SignalCoeff'] = dfCorr['SignalCoeff'].astype(float)
dfCorr['SignalCoeffShuffled'] = dfCorr['SignalCoeffShuffled'].astype(float)

sns.catplot(dfCorr,x='Group',y='SignalCoeff',order=orderGroup,hue='Session',
            dodge=True,kind='point',errorbar=None)
plt.title('Signal correlations (mov)')
plt.savefig(plotpath+'Signal correlations (mov) by session.svg',format='svg',dpi=300)
plt.show()

sns.ecdfplot(dfCorr,x='SignalCoeff',hue='Group',log_scale=True)
plt.title('Signal correlations (mov) ecdf')

dv = 'SignalCoeff'
plottype = 'bar'
plot.plot_with_pval(dfCorr,dv,plottype,orderGroup)

pMlmSignal = compute.mlm_stats(dfCorr, 'signal').pvalues['Group[T.Exp]']

sns.catplot(dfCorr,x='Group',y='SignalCoeff',order=orderGroup,height=6,aspect=1,kind='bar')
plt.title('Signal Correlation coefficients (by neuron)')
plt.savefig(plotpath+'Signal Correlation coefficients (by neuron).svg',dpi=300)
plt.show()

plt.figure(figsize=(10,7))
sns.ecdfplot(dfCorr,x='SignalCoeff',hue='Group',log_scale=True)
sns.ecdfplot(dfCorr,x='SignalCoeffShuffled',hue='Group',log_scale=True)
sns.despine()
plt.savefig(plotpath+'ECDF plot of signal correlation coefficients.svg')
plt.show()

#%% Plot Signal and Noise Correlations side by side 

fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(10,8),sharey=True)

sns.pointplot(dfCorr,x='Group',y='SignalCoeff',order=orderGroup,hue='animalID',
            dodge=False,legend=False,errorbar=None,ax=axes[0,0])
axes[0,0].set_title('Signal Correlation')
axes[0,0].yaxis.set_label_text('Correlation coefficient (R)')

sns.pointplot(dfCorr,x='Group',y='NoiseCoeff',order=orderGroup,hue='animalID',
            dodge=False,legend=False,errorbar=None,ax=axes[0,1])
axes[0,1].set_title('Noise Correlation')

sns.barplot(dfCorr,x='Group',y='SignalCoeff',order=orderGroup,
            dodge=False,ax=axes[1,0])
axes[1,0].yaxis.set_label_text('Correlation coefficient (R)')

sns.barplot(dfCorr,x='Group',y='NoiseCoeff',order=orderGroup,
            dodge=False,ax=axes[1,1])

sns.despine()
plt.tight_layout()
plt.suptitle('Signal and noise correlation coefficients (mov)')
plt.savefig(plotpath+'Signal and noise correlation coefficients (mov).svg')
plt.show()

#%% Load the decoder_auc_scores files for GRATINGS

# aucDrive = os.path.join('G:','My Drive','mrcuts','analysis','decoder','auc_gratings','')
aucDrive = os.path.join(savepath,'decoder','auc_gratings','')

# Load data
listNames = []
listFiles = []

for subdir, dirs, files in os.walk(aucDrive):
    for file in files:
        if ('grat' in file) and any(prefix in file for prefix in ['mrcuts07', 'mrcuts24', 'mrcuts25', 'mrcuts26', 'mrcuts27', 'mrcuts28', 'mrcuts29', 'mrcuts30']):
            file_path = os.path.join(subdir, file)
            filename = os.path.basename(file_path)
            listNames.append(filename)
            df = pd.read_csv(file_path)
            df = df.drop(columns=['Unnamed: 0'])  # Drop 'Unnamed: 0' column
            listFiles.append(df)
            
#%% Load all dfs into one

dfGrat = pd.concat(listFiles,ignore_index=True)

dfGrat.rename(columns={'Animal': 'animalID'}, inplace=True)

dfGrat['Group'] = dfGrat['animalID'].apply(extract.get_group)
#%% Find the mean AUC scores by population size it by session 

sessions = np.unique(dfGrat.Session)

resultsDF = pd.DataFrame()

for sesh in range(len(sessions)):
    subset = dfGrat[dfGrat.Session==sessions[sesh]]
    AUC = subset.groupby('Pop Size')['AUC'].mean()
    
    hold = pd.DataFrame()
    hold['AUC'] = AUC
    hold['Pop Size']= np.unique(subset['Pop Size'])
    hold['Group'] = subset['Group'].iloc[0]
    hold['animalID'] = subset['animalID'].iloc[0]
    hold['Date'] = subset['Date'].iloc[0]
    hold['Session'] = sessions[sesh]
    resultsDF = pd.concat([resultsDF, hold], ignore_index=True)

resultsDF.to_csv(savepath+'Decoder_gratings_AUC_results_DF_(global)_final.csv')

resultsDF = pd.read_csv(savepath+'Decoder_gratings_AUC_results_DF_final.csv')
resultsDF = resultsDF.drop('Unnamed: 0',axis=1)
resultsDF['Date'] = resultsDF['Date'].astype(int).astype(str)
# Dates I want to remove for mrcuts07 POST 
dates_to_remove = ['221214', '221216', '221222']
resultsDF = resultsDF[~resultsDF['Date'].isin(dates_to_remove)]

# Drop sessions 
sessions_to_drop = ['230803_mrcuts24_fov1_grat','230803_mrcuts24_fov2_grat','230810_mrcuts29_fov2_grat',
                    '221123_mrcuts07_fov1_grat']
resultsDF = resultsDF[~resultsDF['Session'].isin(sessions_to_drop)].reset_index(drop=True)


#%% T tests at each pop size

popsize= 25

# List of 'Pop Size' values to iterate over
pop_size_values = range(5, popsize+1, 5)

# Dictionary to store t-test results for each 'Pop Size'
t_test_results = {}

for pop_size in pop_size_values:
    
    control_post = resultsDF[(resultsDF['Group'] == 'Control') & (resultsDF['Pop Size'] == pop_size)]
    exp_post = resultsDF[(resultsDF['Group'] == 'Exp') & (resultsDF['Pop Size'] == pop_size)]

    tResult = mannwhitneyu(control_post['AUC'], exp_post['AUC'])

    # Store t-test results in the dictionary
    t_test_results[pop_size] = {'POST': tResult[1]}
    
# Print t-test results for each 'Pop Size'
for pop_size, results in t_test_results.items():
    print(f"Pop Size: {pop_size}")
    print("P value:", results['POST'])
    print("------------------------")
    
#%% ANOVA for mean AUCs of animals between Group and Pop Size

# Group by 'animalID' and 'Pop Size', calculate the mean 'AUC' for each group (averaging across the sessions)
hold = resultsDF.groupby(['animalID', 'Pop Size', 'Group'])['AUC'].mean().reset_index()

# Perform ANOVA on 'AUC' between 'Group' and 'Pop Size'
anova_result = pg.anova(data=hold[hold['Pop Size']<=25], dv='AUC', between=['Group', 'Pop Size'])

# Print ANOVA result
print(anova_result)

#%% Visualize decoder results 

pGratAUC = compute.mlm_stats(resultsDF, 'decoder').pvalues['Group[T.Exp]']
aovGrat = pg.anova(resultsDF,dv='AUC',between=['Group','Pop Size'])

plt.figure(figsize=(10,5))
popsize = 25

nSessionControl = resultsDF[resultsDF.Group=='Control']['Session'].nunique()
nSessionExp = resultsDF[resultsDF.Group=='Exp']['Session'].nunique()

sns.catplot(data=resultsDF[resultsDF['Pop Size'] <=popsize], x='Pop Size', y='AUC', kind='box', hue = 'Group',
            errorbar=None)

plt.suptitle(f'Gratings mean AUC score box plot (n<={popsize})', fontsize=13, y=1.1)
plt.ylim(0.3, 1.0)  # Set y-axis limits
plt.text(0.5, 0.25, f'nSessionControl: {nSessionControl}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.text(0.5, 0.20, f'nSessionExp: {nSessionExp}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
# plt.savefig(plotpath+'Gratings mean AUC score box plot.svg')
plt.show()

sns.catplot(data=resultsDF[resultsDF['Pop Size'] <=popsize], x='Pop Size', y='AUC', 
            kind='point', hue = 'Group',
            errorbar=None)

plt.suptitle(f'Gratings mean AUC score point plot (n<={popsize})', fontsize=13, y=1.1)
# plt.savefig(plotpath+'Gratings mean AUC score point plot.svg')
plt.show()

dfGrat = resultsDF

#%% Load the decoder_auc_scores files for MOVIES

# aucDrive = os.path.join('G:','My Drive','mrcuts','analysis','decoder','auc_movies','')
aucDrive = os.path.join(savepath,'decoder','auc_movies')

# Load data
listNames = []
listFiles = []

for subdir, dirs, files in os.walk(aucDrive):
    for file in files:
        if ('mov' in file) and any(prefix in file for prefix in ['mrcuts07', 'mrcuts24', 'mrcuts25', 'mrcuts26', 'mrcuts27', 'mrcuts28', 'mrcuts29', 'mrcuts30']):
        # if ('mov' in file) and any(prefix in file for prefix in ['snap']):
           file_path = os.path.join(subdir, file)
           filename = os.path.basename(file_path)
           listNames.append(filename)
           df = pd.read_csv(file_path)
           df = df.drop(columns=['Unnamed: 0'])  # Drop 'Unnamed: 0' column
           listFiles.append(df)
            
#%% Load all dfs into one

dfMov = pd.concat(listFiles,ignore_index=True)

dfMov.rename(columns={'Animal': 'animalID'}, inplace=True)

dfMov['Group'] = dfMov['animalID'].apply(extract.get_group)
#%% Find the mean AUC scores by population size it by session 

sessions = np.unique(dfMov.Session)

resultsDF = pd.DataFrame()

for sesh in range(len(sessions)):
    subset = dfMov[dfMov.Session==sessions[sesh]]
    AUC = subset.groupby('Pop Size')['AUC'].mean()
    
    hold = pd.DataFrame()
    hold['AUC'] = AUC
    hold['Pop Size']= np.unique(subset['Pop Size'])
    hold['Group'] = subset['Group'].iloc[0]
    hold['animalID'] = subset['animalID'].iloc[0]
    hold['Date'] = subset['Date'].iloc[0]
    hold['Session'] = sessions[sesh]
    resultsDF = pd.concat([resultsDF, hold], ignore_index=True)

resultsDF.to_csv(savepath+'Decoder_movies_AUC_results_DF_final.csv')

resultsDF = pd.read_csv(savepath+'Decoder_movies_AUC_results_DF_final.csv')
resultsDF = resultsDF.drop('Unnamed: 0',axis=1)
resultsDF['Date'] = resultsDF['Date'].astype(int).astype(str)
# Dates I want to remove for mrcuts07 POST 
dates_to_remove = ['221214', '221216', '221222']
resultsDF = resultsDF[~resultsDF['Date'].isin(dates_to_remove)]

# Drop sessions 
sessions_to_drop = ['230809_mrcuts28_fov4_mov_neuro','230810_mrcuts29_fov3_mov_neuro']
resultsDF = resultsDF[~resultsDF['Session'].isin(sessions_to_drop)].reset_index(drop=True)

resultsDF.rename(columns={'Session': 'Session'}, inplace=True)

#%% T tests at each pop size

popsize= 25

# List of 'Pop Size' values to iterate over
pop_size_values = range(5, popsize+1, 5)

# Dictionary to store t-test results for each 'Pop Size'
t_test_results = {}

for pop_size in pop_size_values:
    
    control_post = resultsDF[(resultsDF['Group'] == 'Control') & (resultsDF['Pop Size'] == pop_size)]
    exp_post = resultsDF[(resultsDF['Group'] == 'Exp') & (resultsDF['Pop Size'] == pop_size)]

    tResult = mannwhitneyu(control_post['AUC'], exp_post['AUC'])

    # Store t-test results in the dictionary
    t_test_results[pop_size] = {'POST': tResult[1]}
    
# Print t-test results for each 'Pop Size'
for pop_size, results in t_test_results.items():
    print(f"Pop Size: {pop_size}")
    print("P value:", results['POST'])
    print("------------------------")
    
#%% ANOVA for mean AUCs of animals between Group and Pop Size

# Group by 'animalID' and 'Pop Size', calculate the mean 'AUC' for each group
hold = resultsDF.groupby(['animalID', 'Pop Size', 'Group'])['AUC'].mean().reset_index()

# Perform ANOVA on 'AUC' between 'Group' and 'Pop Size'
anova_result = pg.anova(data=hold[hold['Pop Size']<=25], dv='AUC', between=['Group', 'Pop Size'])

# Print ANOVA result
print(anova_result)

#%% Visualize decoder results 

pMovAUC = compute.mlm_stats(resultsDF, 'decoder').pvalues['Group[T.Exp]']

plt.figure(figsize=(10,5))
popsize = 25

nSessionControl = resultsDF[resultsDF.Group=='Control']['Session'].nunique()
nSessionExp = resultsDF[resultsDF.Group=='Exp']['Session'].nunique()

sns.catplot(data=resultsDF[resultsDF['Pop Size'] <=popsize], x='Pop Size', y='AUC', kind='box', hue = 'Group',
            errorbar=None)

plt.suptitle(f'Movies mean AUC score box plot (n<={popsize})', fontsize=13, y=1.1)
plt.ylim(0.3, 1.0)  # Set y-axis limits
plt.text(0.5, 0.25, f'nSessionControl: {nSessionControl}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.text(0.5, 0.20, f'nSessionExp: {nSessionExp}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.savefig(plotpath+'Movies mean AUC score box plot.svg')
plt.show()

sns.catplot(data=resultsDF[resultsDF['Pop Size'] <=popsize], x='Pop Size', y='AUC', 
            kind='point', hue = 'Group',
            errorbar=None)

plt.suptitle(f'Movies mean AUC score point plot (n<={popsize})', fontsize=13, y=1.1)
plt.savefig(plotpath+'Movies mean AUC score point plot.svg')
plt.show()

dfMov = resultsDF

#%% Compare the AUC scores between gratings and movies to show that overall performance is better in movies

pControl = mannwhitneyu(dfGrat[(dfGrat['Group']=='Control')&(dfGrat['Pop Size']<=25)]['AUC'],dfMov[(dfMov['Group']=='Control')&(dfMov['Pop Size']<=25)]['AUC']).pvalue
pExp = mannwhitneyu(dfGrat[(dfGrat['Group']=='Exp')&(dfGrat['Pop Size']<=25)]['AUC'],dfMov[(dfMov['Group']=='Exp')&(dfMov['Pop Size']<=25)]['AUC']).pvalue

dfGrat['Type']='gratings'
dfMov['Type']='movies'

dfFinal = pd.concat([dfGrat,dfMov])

sns.catplot(dfFinal[dfFinal['Pop Size']<=25],x='Group',y='AUC',hue='Type',dodge=True,kind='bar')
sns.catplot(dfFinal[dfFinal['Pop Size']<=25],x='Group',y='AUC',hue='Type',dodge=True,kind='bar')
plt.savefig(plotpath+'Gratings and movies mean AUC score bar plot.svg')
plt.show()
#%% Run the GLM for gratings 

# Make an empty matrix to assign non-zero values to each grating according to time
labels = ['0', '45', '90', '135', '180', '225', '270', '315'] * 16 
nStim = 8
nBases = 5
nFrames = 10240
tDur = 5
tOn = 2
tOff = 3
nTrials = 128 
nRep = 16
nFeatures = nStim * nBases 
freqNeuro = 16
tr_nIters = 20

resultsDF = glm.glm_grat(dfGrat,labels,nStim,nBases,nFrames,freqNeuro,tDur,tOn,tOff,nTrials,nRep,tr_nIters,savepath)

# csvName = 'GLM_gratings_final.csv'
csvName = 'GLM_gratings_final_new_gaussian_filter_5.csv'
resultsDF.to_csv(savepath+csvName, index=False)

# Load
resultsDF = pd.read_csv(savepath+'GLM_gratings_final_new_gaussian_filter_5.csv')

# Convert the values to float 
resultsDF['R2'] = resultsDF['R2'].apply(glm.convert_str_list_to_float)
resultsDF['R2_vis'] = resultsDF['R2_vis'].apply(glm.convert_str_list_to_float)
resultsDF['R2_pupil'] = resultsDF['R2_pupil'].apply(glm.convert_str_list_to_float)
resultsDF['R2_wheel'] = resultsDF['R2_wheel'].apply(glm.convert_str_list_to_float)

#%% Run statistical test and visualize GLM results 

orderGroup = ['Control','Exp']
resultsDF['Group'] = resultsDF['animalID'].apply(extract.get_group)

resultsDF = resultsDF.explode(['R2','R2_vis','R2_wheel','R2_pupil'])

df = resultsDF

# Assuming 'Session' is the column containing session names
session_to_drop = '221123_mrcuts07_fov1_grat-000'

# Drop rows with the specified session name
df = df[df['Session'] != session_to_drop] 

# Convert columns to numeric type
columns_to_convert = ['R2', 'R2_vis', 'R2_wheel', 'R2_pupil']
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
#%% Plot the total R2 values in histogram (log scale)

pGratGLM = compute.mlm_stats(df, 'glm').pvalues['Group[T.Exp]']

# Create a histogram plot with log scale y-axis
_, ax = plt.subplots(figsize=(8, 6))

# Define the number of bins and the bin edges
num_bins = 30
bin_edges = np.linspace(df['R2'].min(), df['R2'].max(), num_bins + 1)

# Plot histograms for each group
for i, group in enumerate(orderGroup):
    data = df[df['Group'] == group]['R2']
    ax.hist(data, bins=bin_edges, alpha=0.5, label=group,log=True, edgecolor='gray')

# Set the axis labels
ax.set_xlabel("R2")
ax.set_ylabel("Frequency (log scale)")

# Add a legend
ax.legend(title='Group')

# Add the p-value text to the plot
xpos1, ypos = 0.7, 0.7  # Adjust the position as needed
ax.text(xpos1, ypos, f"p = {pGratGLM:.3e}", fontsize=15, transform=ax.transAxes)

# Set the title
plt.title('Distribution of GLM R2 scores (gratings)', fontsize=15)
# plt.savefig(plotpath+'Distribution of GLM R2 scores (gratings).svg',dpi=300,bbox_inches='tight')
sns.despine()
plt.show()

#%% Plot average R2 by neurons


nControl = len(df[df['Group'] == 'Control'])
nControlPositive = len(df[(df['Group'] == 'Control') & (df['R2'] > 0)])
nExp = len(df[df['Group'] == 'Exp'])
nExpPositive = len(df[(df['Group'] == 'Exp') & (df['R2'] > 0)])

fControl = nControlPositive / nControl
fExp = nExpPositive / nExp

#%% Plot by session

sns.catplot(df,x='Group',y='R2',hue='Session',order=orderGroup,dodge=True,height=10,aspect=1,kind='point',errorbar='se')
plt.title('R2 scores by session')
# plt.savefig(plotpath+'R2 scores by session (gratings).svg')
plt.show()

#%% Plot & save by neurons 

sns.catplot(df,x='Group',y='R2',order=orderGroup,dodge=True,height=5,aspect=1,kind='bar',errorbar='se')
plt.title('R2 scores by neurons')
plt.savefig(plotpath+'R2 scores by neurons (gratings).svg')
plt.show()

#%% Calculate how many of them have R2 >= 0.1

labelR2 = ['< 0.1', '>= 0.1']

# Calculate the counts of cells in each OSI value range for 'Control' group
nR2Control = [len(df[(df['Group'] == 'Control') & (df['R2'] < 0.1)]),
               len(df[(df['Group'] == 'Control') & (df['R2'] >= 0.1)])]

# Calculate the counts of cells in each OSI value range for 'Exp' group
nR2Exp = [len(df[(df['Group'] == 'Exp') & (df['R2'] < 0.1)]),
               len(df[(df['Group'] == 'Exp') & (df['R2'] >= 0.1)])]

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

# plt.savefig(plotpath+'GLM R2 scores larger than 0.1 (gratings).svg', format='svg', dpi=300, bbox_inches='tight')


#%% Plot the contributions 

# First find R2(total)-R2(partial)
df['dR2_vis'] = df['R2']-df['R2_vis']
df['dR2_pupil'] = df['R2']-df['R2_pupil']    
df['dR2_wheel'] = df['R2']-df['R2_wheel']

# Set negative contribution values to zero
df.loc[df['dR2_vis'] < 0, 'dR2_vis'] = 0
df.loc[df['dR2_pupil'] < 0, 'dR2_pupil'] = 0
df.loc[df['dR2_wheel'] < 0, 'dR2_wheel'] = 0

# Then find dR2/ sum of the rest of dR2 for each neuron?
df['cVis'] = df['dR2_vis'] / sum((df['dR2_vis'],df['dR2_pupil'],df['dR2_wheel']))
df['cPupil'] = df['dR2_pupil'] / sum((df['dR2_vis'],df['dR2_pupil'],df['dR2_wheel']))
df['cWheel'] = df['dR2_wheel'] / sum((df['dR2_vis'],df['dR2_pupil'],df['dR2_wheel']))

# Remove neurons that encode nothing (all contributions were zero)
df.dropna(subset=['cVis'], inplace=True)

df['dR2_vis'] = pd.to_numeric(df['cVis'], errors='coerce')
df['dR2_pupil'] = pd.to_numeric(df['cPupil'], errors='coerce')  
df['dR2_wheel'] = pd.to_numeric(df['cWheel'], errors='coerce')  

group = ['Control','Exp']

pVisCont = compute.get_mwu_pval(df,'dR2_vis',group)
pPupilCont = compute.get_mwu_pval(df,'dR2_pupil',group)
pWheelCont = compute.get_mwu_pval(df,'dR2_wheel',group)

pVisCont = compute.mlm_stats(df,'vis').pvalues['Group[T.Exp]']
pPupilCont = compute.mlm_stats(df,'pupil').pvalues['Group[T.Exp]']
pWheelCont = compute.mlm_stats(df,'wheel').pvalues['Group[T.Exp]']

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
# Plot visual stimulus contribution
sns.barplot(data=df, x='Group', y='dR2_vis', order=orderGroup,
              linestyle='', errorbar=('se', 1), ax=axes[0])
axes[0].set_title('Visual stimulus contribution')
axes[0].text(0.5, 0.02, f'p = {pVisCont:.3e}', ha='center')

# Plot pupil contribution
sns.barplot(data=df, x='Group', y='dR2_pupil', order=orderGroup,
              linestyle='', errorbar=('se', 1), ax=axes[1])
axes[1].set_title('Pupil contribution')
axes[1].text(0.5, 0.02, f'p = {pPupilCont:.3e}', ha='center')

# Plot wheel contribution
sns.barplot(data=df, x='Group', y='dR2_wheel', order=orderGroup,
              linestyle='', errorbar=('se', 1), ax=axes[2])
axes[2].set_title('Wheel contribution')
axes[2].text(0.5, 0.02, f'p = {pWheelCont:.3e}', ha='center')

# Plot visual stimulus contribution: BOXPLOT
sns.boxplot(data=df, x='Group', y='dR2_vis', order=orderGroup,
              ax=axes[0])
axes[0].set_title('Visual stimulus contribution')
axes[0].text(0.5, 0.02, f'p = {pVisCont:.3e}', ha='center')

# Plot pupil contribution
sns.boxplot(data=df, x='Group', y='dR2_pupil', order=orderGroup,
               ax=axes[1])
axes[1].set_title('Pupil contribution')
axes[1].text(0.5, 0.02, f'p = {pPupilCont:.3e}', ha='center')

# Plot wheel contribution
sns.boxplot(data=df, x='Group', y='dR2_wheel', order=orderGroup,
              ax=axes[2])
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

#%% Melt the df

dfMelted = pd.melt(df, id_vars=['Group', 'Session', 'Date', 'animalID'], 
                   value_vars=['R2','R2_vis', 'R2_pupil', 'R2_wheel'], 
                   var_name='Predictor', value_name='Value')
dfMelted['Value'] = pd.to_numeric(dfMelted['Value'], errors='coerce')

dfMelted2 = pd.melt(df, id_vars=['Group', 'Session', 'Date', 'animalID'], value_vars=['dR2_vis','dR2_pupil','dR2_wheel'], var_name='Predictor', value_name='dR2')
dfMelted2['dR2'] = pd.to_numeric(dfMelted2['dR2'], errors='coerce')

#%% Plot the R2s from each predictor 

sns.catplot(dfMelted,x='Predictor',y='Value',hue='Group',kind='violin',dodge=True,
            hue_order=orderGroup,log_scale=True,inner='quart',split=True,
            height=8, aspect=10/8)
plt.title('Gratings GLM performances with or without predictors')

pControlVis = wilcoxon(x=df[df['Group']=='Control']['R2'],y=df[df['Group']=='Control']['R2_vis'])[1]
pControlPupil = wilcoxon(x=df[df['Group']=='Control']['R2'],y=df[df['Group']=='Control']['R2_pupil'])[1]
pControlWheel = wilcoxon(x=df[df['Group']=='Control']['R2'],y=df[df['Group']=='Control']['R2_wheel'])[1]
pExpVis = wilcoxon(x=df[df['Group']=='Exp']['R2'],y=df[df['Group']=='Exp']['R2_vis'])[1]
pExpPupil = wilcoxon(x=df[df['Group']=='Exp']['R2'],y=df[df['Group']=='Exp']['R2_pupil'])[1]
pExpWheel = wilcoxon(x=df[df['Group']=='Exp']['R2'],y=df[df['Group']=='Exp']['R2_wheel'])[1]

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

# For each group, plot the R2 values from each predictor to see the difference 
sns.barplot(dfMelted[dfMelted['Group']=='Control'],x='Predictor',y='Value',ax=axes[0])
axes[0].set_title('Control GLM performance comparisons')
axes[0].text(0.5,1.4,f'pVis={pControlVis:.3e}\npPupil={pControlPupil:.3e}\npWheel={pControlWheel:.3e}',
         transform=axes[0].transAxes,ha='center',va='top')

sns.barplot(dfMelted[dfMelted['Group']=='Exp'],x='Predictor',y='Value',ax=axes[1])
axes[1].set_title('Exp GLM performance comparisons')
axes[1].text(0.5,1.4,f'pVis={pExpVis:.3e}\npPupil={pExpPupil:.3e}\npWheel={pExpWheel:.3e}',
         transform=axes[1].transAxes,ha='center',va='top')

sns.despine()
# plt.savefig(plotpath+'GLM performance comparisons.svg')
plt.show()
#%% Plot the contributions of each predictor 

plt.figure(figsize=(10,8))
sns.catplot(dfMelted2,x='Predictor',y='dR2',hue='Group',kind='violin',dodge=True,
            hue_order=orderGroup,log_scale=True,inner='quart',split=True,
            height=8, aspect=10/8,gap=.1)
plt.ylim([1e-5,10])
plt.title('Gratings GLM predictor contributors')
#%% Different version 

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
# Plot visual stimulus contribution
sns.violinplot(data=df, x='Group', y='dR2_vis', order=orderGroup,
               ax=axes[0],log_scale=True)
axes[0].set_title('Visual stimulus contribution')
axes[0].text(0.5, 0.02, f'p = {pVisCont:.3e}', ha='center')

# Plot pupil contribution
sns.stripplot(data=df, x='Group', y='dR2_pupil', order=orderGroup,
               ax=axes[1])
axes[1].set_title('Pupil contribution')
axes[1].text(0.5, 0.02, f'p = {pPupilCont:.3e}', ha='center')

# Plot wheel contribution
sns.boxplot(data=df, x='Group', y='dR2_wheel', order=orderGroup,
            whis=[0,100],ax=axes[2])
axes[2].set_title('Wheel contribution')
axes[2].text(0.5, 0.02, f'p = {pWheelCont:.3e}', ha='center')

# Set y-axis limits for all subplots
[ax.set_ylim(-4, 4) for ax in axes]
# Remove redundant subplots generated by sns.catplot()
plt.close(2)
plt.close(3)
plt.close(4)

# Adjust layout
plt.tight_layout()
sns.despine()

#%% Run the GLM for neurons for movies dataset 

df = extract.extract_mov(dfMaster)
df = extract.get_dff(df)
df = df[df['animalID'].isin(listID)].reset_index(drop=True)
df['Group'] = df['animalID'].apply(extract.get_group)
dates_to_remove = ['221214', '221216', '221222']
df = df[~df['Date'].isin(dates_to_remove)]

nFrames = 8704
tr_nIters = 20

df = glm.glm_neuron(df,nFrames,tr_nIters)

df['R2'] = df['R2'].apply(glm.convert_str_list_to_float)

csvName = 'GLM_movies_final.csv'
df.to_csv(savepath+csvName, index=False)


#%% Load, filter, and plot the data

from matplotlib.ticker import ScalarFormatter

resultsDF = pd.read_csv(savepath+'GLM_movies_final.csv')
resultsDF['R2'] = resultsDF['R2'].astype(float)


sessions_to_drop = ['230809_mrcuts28_fov4_mov-000','230804_mrcuts26_fov3_mov-000']
resultsDF = resultsDF[~resultsDF['Session'].isin(sessions_to_drop)]

# Detect rows with 'R2' values larger than 0.9
rows_to_drop = resultsDF[resultsDF['R2'] > 0.9]

# Drop rows with 'R2' values larger than 0.9
resultsDF = resultsDF.drop(rows_to_drop.index)

# Optionally, reset the index after dropping rows
resultsDF.reset_index(drop=True, inplace=True)

plt.figure(figsize=(10,7))
sns.histplot(resultsDF, x='R2', hue='Group', log_scale=True,bins=30)
sns.despine()
plt.title('Single neuron encoding of population activity')

# Get the current axes
ax = plt.gca()

# Set the x-axis formatter to ScalarFormatter
ax.xaxis.set_major_formatter(ScalarFormatter())

plt.savefig(plotpath+'Single neuron encoding of population activity.svg',dpi=300)
plt.show()

#%% Plot the total R2 values in histogram (log scale)

pMovGLM = compute.mlm_stats(resultsDF, 'glm').pvalues['Group[T.Exp]']

# Create a histogram plot with log scale y-axis
_, ax = plt.subplots(figsize=(8, 6))

# Define the number of bins and the bin edges
num_bins = 30
bin_edges = np.linspace(resultsDF['R2'].min(), resultsDF['R2'].max(), num_bins + 1)

# Plot histograms for each group
for i, group in enumerate(orderGroup):
    data = resultsDF[resultsDF['Group'] == group]['R2']
    ax.hist(data, bins=bin_edges, alpha=0.5, label=group,log=True, edgecolor='gray')

# Set the axis labels
ax.set_xlabel("R2")
ax.set_ylabel("Frequency (log scale)")

# Add a legend
ax.legend(title='Group')

# Add the p-value text to the plot
xpos1, ypos = 0.7, 0.7  # Adjust the position as needed
ax.text(xpos1, ypos, f"p = {pMovGLM:.3e}", fontsize=15, transform=ax.transAxes)

# Set the title
plt.title('Distribution of GLM R2 scores (movies)', fontsize=15)
sns.despine()
plt.show()


#%% Plot by session

sns.catplot(resultsDF,x='Group',y='R2',hue='Session',order=orderGroup,dodge=True,height=10,aspect=1,kind='point',errorbar='se')
plt.title('R2 scores by session')
plt.savefig(plotpath+'R2 scores by session (movies).svg')

#%% Plot average by neurons 

sns.catplot(resultsDF,x='Group',y='R2',kind='bar',order=orderGroup)
plt.title('Average R2 of all neurons (movies)')
plt.savefig(plotpath+'R2 scores by all neurons (movies).svg')
#%% Plot in ecdf plot by animal 

plt.figure(figsize=(10,7))
sns.displot(resultsDF,x='R2',hue='animalID',log_scale=True,kind='ecdf',col='Group')
sns.despine()
plt.suptitle('Single neuron encoding of population activity (ecdf)',y=1.1)
plt.savefig(plotpath+'Single neuron encoding of population activity (ecdf).svg',dpi=300)
plt.show()