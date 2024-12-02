# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:58:28 2024

@author: jihop
"""
# Import packages needed for running the code
import os
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from functions import extract,compute

#%% Define paths

# Home directory where the repository is cloned 
# Make sure to change the information accordingly
homepath = os.path.join('C:\\','Users','jihop','Documents','GitHub','Park_et_al_2024','')
# Directory containing data files
datapath = os.path.join(homepath,'sample-data','spks','')
# Directory to save output files
savepath = os.path.join(homepath,'results','sample-output','')
# Directory to save plots 
plotpath = os.path.join(homepath,'results','sample-plots','')

#%% Load deconvolved spiking data 

# Loads spike data of all sessions from datapath
dfSpks = pd.read_csv(datapath+'spks_data.csv')

dfSpks['Spks'] = dfSpks['Spks'].apply(lambda x: ast.literal_eval(x))

# Set the keyword to filter out sessions (ex. 'spo', 'grat')
keyword = 'spo'
df = extract.get_session(dfSpks,keyword)

#%% SPONTANEOUS ACTIVITY: Firing rates 

# Define parameters
tDur = 640  # Total duration in seconds
freqNeuro = 16  # Neuro data sampling frequency (Hz)
timeBin = 0.1  # Time bin in seconds (100 ms)
threshold = 5 

# Compute the firing rate of single neurons using compute.get_fr
df = compute.get_fr(df,tDur,freqNeuro,timeBin,threshold)
dfFR = df.explode(['FR'])
dfFR['FR'] = dfFR['FR'].astype(float)

#%% SPONTANEOUS ACTIVITY: Plot the firing rates of single neurons

# Define order of groups for plotting
orderGroup = ['Control','Exp']

# Compute the p value of firing rates between the two groups
pMlmFR = compute.mlm_stats(dfFR, 'fr').pvalues['Group[T.Exp]']

# Plot the average firing rates of all neurons in each group
plt.figure(figsize=(5,5))
sns.barplot(dfFR,x='Group',y='FR',order=orderGroup)
plt.ylim([1.3,1.95])
plt.text(3,1.1,f"pval={pMlmFR:0.3E}")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
sns.despine()
plt.savefig(plotpath+'Firing rates average.svg')
plt.show()

# Plot an empirical cumulative density function 
plt.figure(figsize=(10,7))
sns.ecdfplot(dfFR,x='FR',hue='Group')
sns.despine()
plt.title('Firing rates',y=1.2,fontsize=15)
plt.savefig(plotpath+'Firing rates (ECDF).svg')
plt.show()

#%% SPONTANEOUS ACTIVITY: neuron-to-neuron pairwise correlation 

# Define parameters
sessDur = 640
freqNeuro = 16
nShuffles = 100

# COmpute the pairwise correlation using compute.get_pairwise_corr 
df = compute.get_pairwise_corr(df,freqNeuro,sessDur,nShuffles)
dfCorr = df.explode(['Coeff','CoeffShuffled'])

# Save the output as a csv in savepath
dfCorr.to_csv(savepath+'Spo_pairwise_correlations.csv')

# Load
# dfCorr = pd.read_csv(savepath + 'Spo_pairwise_correlations.csv', index_col=0)

#%% Plot the pairwise correlations

dfCorr['Coeff'] = dfCorr['Coeff'].astype(float)
dfCorr['CoeffShuffled'] = dfCorr['CoeffShuffled'].astype(float)

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


#%% MOVIES: signal and noise correlation 

# Extract movies sessions from the spikes dataframe 
stimType = 'mov'
df = extract.get_session(dfSpks,stimType)

#%% MOVIES: Signal & noise correlation 


"""

Signal correlation quantifies tuning similarity between neurons

Noise correlations measures co-fluctuation of trial-to-trial variability

"""

# Define parameters
freqNeuro = 16
tOff = 3
tOn = 2
nTrials = 32
nFrames = 8704
nStim = 7
nShuffles = 100

# Compute noise and signal correlation 
df = compute.get_noise_corr(df,freqNeuro,tOff,tOn,nTrials,nFrames,nStim,nShuffles,plotpath)
df = compute.get_signal_corr(df,freqNeuro,tOff,tOn,nTrials,nFrames,nStim,nShuffles,plotpath)
dfCorr = df.explode(['NoiseCoeff','NoiseCoeffShuffled','SignalCoeff','SignalCoeffShuffled'])

# Save the output as csv
dfCorr.to_csv(savepath+'Noise and signal correlations (mov).csv')

# Load the data from saved output it needed
dfCorr = pd.read_csv(savepath+'Noise and signal correlations (mov).csv')
dfCorr = dfCorr.drop('Unnamed: 0',axis=1)

# Remove spks data for convenience  
dfCorr = dfCorr.drop(columns=['Spks','zSpks'])

#%% MOVIES: Plot noise correlations

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

#%% MOVIES: Plot signal correlations

dfCorr['SignalCoeff'] = dfCorr['SignalCoeff'].astype(float)
dfCorr['SignalCoeffShuffled'] = dfCorr['SignalCoeffShuffled'].astype(float)

sns.catplot(dfCorr,x='Group',y='SignalCoeff',order=orderGroup,hue='Session',
            dodge=True,kind='point',errorbar=None)
plt.title('Signal correlations (mov)')
plt.savefig(plotpath+'Signal correlations (mov) by session.svg',format='svg',dpi=300)
plt.show()

sns.ecdfplot(dfCorr,x='SignalCoeff',hue='Group',log_scale=True)
plt.title('Signal correlations (mov) ecdf')

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

#%% MOVIES: Plot Signal and Noise Correlations side by side 

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