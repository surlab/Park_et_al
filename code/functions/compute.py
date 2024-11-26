# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:23:08 2024

@author: jihop

# Calculate basic properties from DFF 

"""

# import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.signal import butter, filtfilt

import statsmodels.formula.api as smf
# from scipy.optimize import curve_fit


### GENERAL ###

# Smoothing filter
def causal_half_gaussian_filter(data, sigma):
    # Create the causal half Gaussian filter kernel
    x = np.arange(0, 4 * sigma, dtype=float)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2)) * (x >= 0)

    # Normalize the kernel
    kernel /= np.sum(kernel)

    # Apply the causal half Gaussian filter using convolution
    filtered_data = np.convolve(data, kernel, mode='same')

    return filtered_data

#  Smoothen via binning
def smooth_data(data, bin_size):
    # Calculate the number of bins
    num_bins = len(data) // bin_size
    
    # Reshape the data to (num_bins, bin_size)
    binned_data = data[:num_bins * bin_size].reshape(num_bins, bin_size)
    
    # Compute the mean of each bin
    smoothed_data = binned_data.mean(axis=1)
    
    return smoothed_data

# Bandpass filters

def highpass_filter(data, cutoff, fs, order=4):
    """
    Apply a high-pass filter to the data.

    Parameters:
    - data: The input signal to be filtered.
    - cutoff: The cutoff frequency (in Hz) for the high-pass filter.
    - fs: The sampling frequency of the data (in Hz).
    - order: The order of the filter.

    Returns:
    - filtered_data: The high-pass filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

def lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a low-pass filter to the data.

    Parameters:
    - data: The input signal to be filtered.
    - cutoff: The cutoff frequency (in Hz) for the low-pass filter.
    - fs: The sampling frequency of the data (in Hz).
    - order: The order of the filter.

    Returns:
    - filtered_data: The low-pass filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a band-pass filter to the data.

    Parameters:
    - data: The input signal to be filtered.
    - lowcut: The lower cutoff frequency (in Hz).
    - highcut: The upper cutoff frequency (in Hz).
    - fs: The sampling frequency of the data (in Hz).
    - order: The order of the filter.

    Returns:
    - filtered_data: The band-pass filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if low <= 0 or high >= 1:
        raise ValueError("Cutoff frequencies must be within the range (0, Nyquist frequency).")
    
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

def highpass_filter_with_nans(data, cutoff, fs, order=4):
    # Identify valid (non-NaN) data
    valid_idx = ~np.isnan(data)
    
    # Filter only valid data
    filtered_data = np.full_like(data, np.nan)  # Start with all NaNs
    filtered_data[valid_idx] = highpass_filter(data[valid_idx], cutoff, fs, order)
    
    return filtered_data

def lowpass_filter_with_nans(data, cutoff, fs, order=4):
    # Identify valid (non-NaN) data
    valid_idx = ~np.isnan(data)
    
    # Filter only valid data
    filtered_data = np.full_like(data, np.nan)  # Start with all NaNs
    filtered_data[valid_idx] = lowpass_filter(data[valid_idx], cutoff, fs, order)
    
    return filtered_data

# Statistical test
def get_mwu_pval(df,dv,group):
    pval = stats.mannwhitneyu(df[df['Group']==group[0]][dv],
                              df[df['Group']==group[1]][dv])[1]
    print(f'Mannwhitney U test p-val between {group[0]} and {group[1]} = {pval:0.3e}')
    return pval    


# Calculate firing rates using deconvoled spikes
def get_fr(df,tDur,freqNeuro,timeBin,threshold):

    listFR = []

    for n in range(df.shape[0]):
        
        hold = []
        
        session = df['Session'].iloc[n]
        spks = df['Spks'].iloc[n]
        nUnits = spks.shape[0]
        print(f'Session: {session}; nUnits: {nUnits}')
        
        spksTime = []
        
        # Identify spike times for each neuron
        for neuron_data in spks:
            spike_indices = np.where(neuron_data > threshold)[0]  # Get indices where data exceeds threshold
            spike_times = spike_indices / freqNeuro  # Convert indices to time units
            spksTime.append(spike_times)

        # Initialize a list to store firing rates for each neuron
        firingRates = []

        # Iterate through each neuron's spike times
        for neuronSpikes in spksTime:
            # Create an array of time bins
            timeBins = np.arange(0, tDur, timeBin)
            
            # Count spikes within each time bin
            spikeCounts, _ = np.histogram(neuronSpikes, bins=timeBins)
            
            # Calculate firing rate in Hz
            firingRate = spikeCounts / timeBin
            
            # Append the firing rate to the list
            firingRates.append(firingRate)

        # Iterate through the neurons
        for i, firingRate in enumerate(firingRates):
            # Calculate the mean firing rate for the neuron
            meanRate = np.mean(firingRate)  # Mean firing rate in Hz
            
            hold.append(meanRate)
        
        listFR.append(hold)
            
    df['FR'] = listFR
    # df['FR'] = df['FR'].apply(np.array)
    
    return df

# Calculate pairwise neuron-to-neuron correlation (stimulus independent)
def get_pairwise_corr(df,freqNeuro,sessDur,nShuffles):
    
    # Empty lists to save an array of correlation coefficients per session
    listCorr = []
    listCorrShuffled = []

    corr_sum = 0
    n_pairs = 0

    for n in range(len(df)):
        
        session = df['Session'].iloc[n]
        spks = df['zSpks'].iloc[n]
        nUnits, nFrames = spks.shape
        
        print(f'Processing {session}...')

        # Create a matrix to store the pairwise correlation coefficients between neurons
        corr_matrix = np.zeros((nUnits, nUnits))
        corr_matrix_shuffled = np.zeros((nUnits, nUnits))
            
        # Now calculate the correlation coefficients with the interpolated data 
        
        for i in range(nUnits):
            for j in range(i+1, nUnits):
                # Concatenate the trial activity vectors for each neuron pair
                neuOne = spks[i,:]
                neuTwo = spks[j,:]
                concatData = np.concatenate((neuOne,neuTwo))
                
                # Calculate the Pearson correlation coefficient
                corr = np.corrcoef(concatData[:nFrames],concatData[nFrames:])[0, 1]
                corr_sum += corr
                n_pairs += 1
                            
                # Store the correlation coefficient in the matrix
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                
        # Flatten the matrix and save data
        neu_corr = np.triu(corr_matrix)[np.triu(corr_matrix) != 0].flatten()
        
        listCorr.append(neu_corr)
        
        # plt.figure(figsize=(10,8))
        # sns.heatmap(corr_matrix, cmap='PuOr_r', vmin=-0.3, vmax=0.3)
        # plt.title(f"{session}")
        # plt.savefig(plotDrive+f"{filename}_noise_corr.svg", format='svg')
        # plt.show()
        
        corr_shuffled_all = []

        for shuffle_iter in range(nShuffles):
            # Shuffle the data for each neuron independently
            spksShuffled = np.array([np.random.permutation(neuron_data) for neuron_data in spks])
            
            if shuffle_iter % 20 == 0:
                print(f"Shuffling iteration {shuffle_iter}")

            for i in range(nUnits):
                for j in range(i + 1, nUnits):
                    neuOneShuffled = spksShuffled[i,:]
                    neuTwoShuffled = spksShuffled[j,:]
                    corrShuffled = np.corrcoef(neuOneShuffled, neuTwoShuffled)[0, 1]
                    corr_matrix_shuffled[i, j] = corrShuffled
                    corr_matrix_shuffled[j, i] = corrShuffled
                    
            neu_corr_shuffled = np.triu(corr_matrix_shuffled)[np.triu(corr_matrix_shuffled) != 0].flatten()
            
            corr_shuffled_all.append(neu_corr_shuffled)

        corr_shuffled_avg = np.mean(corr_shuffled_all, axis=0)     
        
        listCorrShuffled.append(corr_shuffled_avg)
        
    df['Coeff'] = listCorr
    df['CoeffShuffled'] = listCorrShuffled
    
    return df

### GRATINGS ###

# Calculate maximum response magnitudes from df 
# Make sure to define tWindow and freqNeuro before calling the function 
# Calculates maxResp for ALL cells
def get_maxresp(df,freqNeuro,tWindow,tOff,tOn):
    
    listSession = []
    listMaxResp = []
    
    for session in range(len(df)):
        sessionName = df['Session'][session]
        print(f'Processing session: {sessionName}')
        gratStruct = df['GratStruct'][session]
        nUnits = gratStruct['roi'].shape[1]
        maxRespBySess = []

        for n in range(nUnits):
            avgRespPerGrat = gratStruct['roi'][0][n]['mean_grat']
            # prefGrat = gratStruct['roi'][0][n]['tc']['mean_r'][0][0]['pref_grat'][0][0][0][0]
            # idxPrefGrat = np.where(stimTotal == prefGrat)[0][0]
                
            maxRespByGrat = np.max(avgRespPerGrat[tOff*freqNeuro:tOff*freqNeuro+(tWindow*freqNeuro)])
            maxRespBySess.append(maxRespByGrat)
                
        listSession.append(sessionName)
        listMaxResp.append(maxRespBySess)

    df['maxResp'] = listMaxResp
    df['maxResp'] = df['maxResp'].apply(np.array)
    
    return df

# Von Mises 

# def VonMisesFunction(theta, baseline, peak1_amplitude, peak2_amplitude, width, pref_rad):
#     return baseline + peak1_amplitude * np.exp(width * (np.cos(theta - pref_rad) - 1)) + peak2_amplitude * np.exp(width * (np.cos(theta - pref_rad - np.pi) - 1))

# def VMFit(observed, pref_deg):
#     observed = np.array(observed)
#     pref_rad = np.radians(pref_deg)
#     angle_num = observed.size
#     theta_deg = np.linspace(0, 360 - 360/angle_num, angle_num)
#     theta_rad = np.radians(theta_deg)

#     max_angle = np.argmax(observed)
#     coeff_init = [np.mean(observed), np.max(observed), (np.max(observed) + np.mean(observed)) / 2, 7, pref_rad]

#     # Set bounds for the parameters
#     lower_bound = [0, 0, 0, 0, 0]
#     upper_bound = [np.mean(observed), np.max(observed), np.max(observed), np.inf, 2 * np.pi]

#     # Perform curve fitting
#     coeff_set, cov_matrix = curve_fit(VonMisesFunction, theta_rad, observed, p0=coeff_init, bounds=(lower_bound, upper_bound))

#     # Calculate goodness of fit
#     residuals = observed - VonMisesFunction(theta_rad, *coeff_set)
#     ss_res = np.sum(residuals ** 2)
#     ss_tot = np.sum((observed - np.mean(observed)) ** 2)
#     r_squared = 1 - (ss_res / ss_tot)

#     return coeff_set, r_squared

### MOVIES ###

# Calculate SIGNAL correlation (stimulus dependent) (for movies specifically)
def get_signal_corr(df,freqNeuro,tOff,tOn,nTrials,nFrames,nStim,nShuffles,plotpath):
    # Empty lists to save an array of correlation coefficients per session
    listCorr = []
    listCorrShuffled = []
    for n in range(len(df)):
        session = df['Session'].iloc[n]
        spks = df['zSpks'].iloc[n]
        nUnits, time = spks.shape

        spksIntp = interp1d(np.linspace(0, 1, time), spks, axis=1)(np.linspace(0, 1, nFrames))

        # Create a matrix to store the pairwise correlation coefficients between neurons
        corr_matrix = np.zeros((nUnits, nUnits))
        corr_matrix_shuffled = np.zeros((nUnits, nUnits))

        # Reshape to extract the stim on period responses & get the average for each stimulus 
        sample = np.reshape(spksIntp,(nUnits,int(nFrames/nTrials),nTrials)) # Reshape into nUnits x nFrames/nTrials x nTrials
        spksStimOn = sample[:,tOff*freqNeuro:,:] # Only the Stim On period 
        sample2 = np.reshape(spksStimOn,(nUnits,int(tOn*freqNeuro),nStim,nTrials)) # Reshape into nUnits x frames for each (tOn*freqNeuro) x nStim x nTrials
        spksAvgPerStim = np.mean(sample2,axis=1) # Average across stimulus frames (NOT trials); nUnits x nStim x nTrials
        spksAvgAllTrials = np.reshape(spksAvgPerStim,(nUnits,224)) # A single value for each stimulus for each neuron
        
        corr_matrix = np.zeros((nUnits, nUnits))
        
        # corr_matrix = np.cov(zspksNew)
         
        for i in range(nUnits):
            for j in range(i + 1, nUnits):
                # Concatenate the trial activity vectors for each neuron pair
                neuOne = spksAvgAllTrials[i, :]
                
                neuTwo = spksAvgAllTrials[j, :]
                # concatData = np.concatenate((neuOne, neuTwo))
                
                # Calculate the Pearson correlation coefficient
                # corr = np.corrcoef(concatData[:time], concatData[time:])[0, 1]
                corr = np.corrcoef(neuOne, neuTwo)[0,1]
    
                # Store the correlation coefficient in the matrix
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                
        # Calculate the hierarchical clustering
        linkage = sch.linkage(corr_matrix, method='weighted', metric='euclidean')
        
        # Reorder the correlation matrix based on clustering
        idx = sch.leaves_list(linkage)
        corr_matrix_reordered = corr_matrix[idx, :]
        corr_matrix_reordered = corr_matrix_reordered[:, idx]
        
        # Plot the reordered correlation matrix
        plt.figure(figsize=(10, 8))
        # sns.heatmap(corr_matrix, cmap='PiYG_r', vmin=-0.3, vmax=0.3)
        sns.heatmap(corr_matrix_reordered, cmap='PiYG_r', vmin=-0.3, vmax=0.3)
        plt.title(f"Signal_{session} (Reordered based on clustering)")
        # plt.savefig(plotDrive+f'{filename}_mov signal correlation matrix.svg')
        plt.show()
        
        # # Get the order of neurons from the dendrogram
        # dendrogram = sch.dendrogram(linkage, no_plot=True)
        # neuron_order = dendrogram['leaves']
        
        # # Adjust the x and y tick labels based on the neuron order
        # plt.xticks(ticks=np.arange(len(neuron_order)), labels=neuron_order)
        # plt.yticks(ticks=np.arange(len(neuron_order)), labels=neuron_order)
        
        # plt.title(f"Signal_{session} (Reordered based on clustering)")

        # Flatten the matrix and save data
        neu_corr = np.triu(corr_matrix)[np.triu(corr_matrix) != 0].flatten()
        
        listCorr.append(neu_corr)
        
        corr_shuffled_all = []

        for shuffle_iter in range(nShuffles):
            # Shuffle the data for each neuron independently
            spksFinalShuffled = np.array([np.random.permutation(neuron_data) for neuron_data in spksAvgAllTrials])
            
            if shuffle_iter % 20 == 0:
                print(f"Shuffling iteration {shuffle_iter}")

            for i in range(nUnits):
                for j in range(i + 1, nUnits):
                    neuOneShuffled = spksFinalShuffled[i,:]
                    neuTwoShuffled = spksFinalShuffled[j,:]
                    corrShuffled = np.corrcoef(neuOneShuffled, neuTwoShuffled)[0, 1]
                    corr_matrix_shuffled[i, j] = corrShuffled
                    corr_matrix_shuffled[j, i] = corrShuffled
                    
            neu_corr_shuffled = np.triu(corr_matrix_shuffled)[np.triu(corr_matrix_shuffled) != 0].flatten()
            
            corr_shuffled_all.append(neu_corr_shuffled)

        corr_shuffled_avg = np.mean(corr_shuffled_all, axis=0)
        
        listCorrShuffled.append(corr_shuffled_avg)
        
    df['SignalCoeff'] = listCorr
    df['SignalCoeffShuffled'] = listCorrShuffled
    
    # df['SignalCoeff'] = df['SignalCoeff'].astype(float)
    # df['SignalCoeffShuffled'] = df['SignalCoeffShuffled'].astype(float)
    
    return df
        
# Calculate NOISE correlation (stimulus dependent) (for movies specifically)
def get_noise_corr(df,freqNeuro,tOff,tOn,nTrials,nFrames,nStim,nShuffles,plotpath):
    # Empty lists to save an array of correlation coefficients per session
    listCorr = []
    listCorrShuffled = []
    for n in range(len(df)):
        session = df['Session'].iloc[n]
        spks = df['zSpks'].iloc[n]
        nUnits, time = spks.shape

        spksIntp = interp1d(np.linspace(0, 1, time), spks, axis=1)(np.linspace(0, 1, nFrames))

        # Create a matrix to store the pairwise correlation coefficients between neurons
        corr_matrix = np.zeros((nUnits, nUnits))
        corr_matrix_shuffled = np.zeros((nUnits, nUnits))
    
        # Reshape to extract the stim on period responses & get the average for each stimulus 
        sample = np.reshape(spksIntp,(nUnits,int(nFrames/nTrials),nTrials)) # Reshape into nUnits x nFrames/nTrials x nTrials
        spksStimOn = sample[:,tOff*freqNeuro:,:] # Only the Stim On period 
        sample2 = np.reshape(spksStimOn,(nUnits,int(tOn*freqNeuro),nStim,nTrials)) # Reshape into nUnits x frames for each (tOn*freqNeuro) x nStim x nTrials
        spksAvgPerStim = np.mean(sample2,axis=1) # Average across stimulus frames (NOT trials); nUnits x nStim x nTrials
        spksZAcrossTrials = stats.zscore(spksAvgPerStim, axis=2) # 3rd dimension contains z scores instead of activity
        spksFinal = np.reshape(spksZAcrossTrials,(nUnits,224))
        
        corr_matrix = np.zeros((nUnits, nUnits))
         
        for i in range(nUnits):
            for j in range(i + 1, nUnits):
                # Concatenate the trial activity vectors for each neuron pair
                neuOne = spksFinal[i, :]
                
                neuTwo = spksFinal[j, :]
                # concatData = np.concatenate((neuOne, neuTwo))
                
                # Calculate the Pearson correlation coefficient
                corr = np.corrcoef(neuOne, neuTwo)[0,1]
    
                # Store the correlation coefficient in the matrix
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                
        # Calculate the hierarchical clustering
        linkage = sch.linkage(corr_matrix, method='weighted', metric='euclidean')
        
        # Reorder the correlation matrix based on clustering
        idx = sch.leaves_list(linkage)
        corr_matrix_reordered = corr_matrix[idx, :]
        corr_matrix_reordered = corr_matrix_reordered[:, idx]
        
        # Plot the reordered correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap='PiYG_r', vmin=-0.3, vmax=0.3)
        
        # # Get the order of neurons from the dendrogram
        # dendrogram = sch.dendrogram(linkage, no_plot=True)
        # neuron_order = dendrogram['leaves']
        
        # # Adjust the x and y tick labels based on the neuron order
        # plt.xticks(ticks=np.arange(len(neuron_order)), labels=neuron_order)
        # plt.yticks(ticks=np.arange(len(neuron_order)), labels=neuron_order)
        
        # plt.title(f"Noise_{session} (Reordered based on clustering)")
        plt.title(f"Noise_{session}")
        plt.savefig(plotpath+f"Noise corr matrix {session}.svg")
        plt.show()
        # Flatten the matrix and save data
        neu_corr = np.triu(corr_matrix_reordered)[np.triu(corr_matrix_reordered) != 0].flatten()
        
        listCorr.append(neu_corr)
        
        
        corr_shuffled_all = []

        for shuffle_iter in range(nShuffles):
            # Shuffle the data for each neuron independently
            spksFinalShuffled = np.array([np.random.permutation(neuron_data) for neuron_data in spksFinal])
            
            if shuffle_iter % 20 == 0:
                print(f"Shuffling iteration {shuffle_iter}")


            for i in range(nUnits):
                for j in range(i + 1, nUnits):
                    neuOneShuffled = spksFinalShuffled[i,:]
                    neuTwoShuffled = spksFinalShuffled[j,:]
                    corrShuffled = np.corrcoef(neuOneShuffled, neuTwoShuffled)[0, 1]
                    corr_matrix_shuffled[i, j] = corrShuffled
                    corr_matrix_shuffled[j, i] = corrShuffled
                    
            neu_corr_shuffled = np.triu(corr_matrix_shuffled)[np.triu(corr_matrix_shuffled) != 0].flatten()
            
            corr_shuffled_all.append(neu_corr_shuffled)

        corr_shuffled_avg = np.mean(corr_shuffled_all, axis=0)
        
        listCorrShuffled.append(corr_shuffled_avg)
        
    df['NoiseCoeff'] = listCorr
    df['NoiseCoeffShuffled'] = listCorrShuffled
    
    # df['NoiseCoeff'] = df['NoiseCoeff'].astype(float)
    # df['NoiseCoeffShuffled'] = df['NoiseCoeffShuffled'].astype(float)
    
    return df


#%% Multi-level model for statistical testing of fixed variables 

"""

Create a linear mixed effects model for different response variables

{ response_variable ~ fixed_effects + random_effects }

# response_variable = dependent variable (ex. AUROC, R2, OSI...)

# fixed_effects = variables that have systemic effect on the dv (ex. Group, condition...)

# random_effects = variables assumed to have random effects on the dv; often used for correlation or clustering within the data (ex. animalID) 

"""

def mlm_stats(df, case):
    # Define the formula based on the case
    if case == 'decoder':
        formula = 'AUC ~ 1 + Group + Q("Pop Size")'
    elif case == 'decoder2':
        formula = 'AUC ~ 1+ (Group * Q("Pop Size"))'
    elif case == 'glm':
        formula = 'R2 ~ Group'
    elif case == 'osi':  # Add more cases as needed
        formula = 'OSI ~ Group'
    elif case == 'maxResp':  
        formula = 'maxResp ~ Group'
    elif case == 'coeff':
        formula = 'Coeff ~ Group'
    elif case == 'noise':
        formula = 'NoiseCoeff ~ Group'
    elif case == 'signal':
        formula = 'SignalCoeff ~ Group'
    elif case == 'fr':
        formula = 'FR ~ Group'
    elif case == 'rel':
        formula = 'Rel ~ Group'
    elif case == 'vis':
        formula = 'dR2_vis ~ Group'
    elif case == 'pupil':
        formula = 'dR2_pupil ~ Group'
    elif case == 'wheel':
        formula = 'dR2_wheel ~ Group'
    else: 
        formula = f'{case} ~ Group'
    # else:
    #     raise ValueError("Invalid case specified.")

    # Define the variance component formula (vc_formula)
    vc_formula = '1'

    # Fit the mixed-effects model
    mlm = smf.mixedlm(formula, df, groups=df['animalID'], re_formula=vc_formula).fit()

    # Print the summary
    print(mlm.summary())
    
    # Return the fitted model object
    return mlm

# def mlm_stats_2(df, case):
#     # Define the formula based on the case
#     if case == 'decoder':
#         formula = 'AUC ~ 1 + Group + Q("Pop Size")'
#     elif case == 'decoder2':
#         formula = 'AUC ~ 1 + (Group * Q("Pop Size"))'
#     elif case == 'glm':
#         formula = 'R2 ~ Group'
#     elif case == 'osi':  # Add more cases as needed
#         formula = 'OSI ~ Group'
#     elif case == 'maxResp':  
#         formula = 'maxResp ~ Group'
#     elif case == 'coeff':
#         formula = 'Coeff ~ Group'
#     elif case == 'noise':
#         formula = 'NoiseCoeff ~ Group'
#     elif case == 'signal':
#         formula = 'SignalCoeff ~ Group'
#     elif case == 'fr':
#         formula = 'FR ~ Group'
#     else:
#         raise ValueError("Invalid case specified.")

#     # Define the variance component formula (vc_formula)
#     vc_formula = {'animalID': '0 + C(animalID)', 'Session': '0 + C(Name)'}

#     # Fit the mixed-effects model
#     mlm = smf.mixedlm(formula, df, groups=df[['animalID', 'Session']], re_formula=vc_formula).fit()

#     # Print the summary
#     print(mlm.summary())
    
#     # Return the fitted model object
#     return mlm

def mlm_stats_session(df, case):
    # Define the formula based on the case
    if case == 'decoder':
        formula = 'AUC ~ 1 + Group + Q("Pop Size")'
    elif case == 'decoder2':
        formula = 'AUC ~ 1 + (Group * Q("Pop Size"))'
    elif case == 'glm':
        formula = 'R2 ~ Group'
    elif case == 'osi':  # Add more cases as needed
        formula = 'OSI ~ Group'
    elif case == 'maxResp':  
        formula = 'maxResp ~ Group'
    elif case == 'coeff':
        formula = 'Coeff ~ Group'
    elif case == 'noise':
        formula = 'NoiseCoeff ~ Group'
    elif case == 'signal':
        formula = 'SignalCoeff ~ Group'
    elif case == 'fr':
        formula = 'FR ~ Group'
    else:
        raise ValueError("Invalid case specified.")

    # Define the variance component formula (vc_formula)
    vc_formula = '1'

    # Fit the mixed-effects model
    mlm = smf.mixedlm(formula, df, groups=df['Session'], re_formula=vc_formula).fit()

    # Print the summary
    print(mlm.summary())
    
    # Return the fitted model object
    return mlm

# MLM for SNAP-5114 dataset
def mlm_stats_snap(df, group, case):
    
    if group == 'Saline':
        df = df[df['Group']==group]
    elif group == 'SNAP-5114':
        df = df[df['Group']==group]
    
    # Define the formula based on the case
    if case == 'decoder':
        formula = 'AUC ~ 1 + Condition + Q("Pop Size")'
    elif case == 'decoder2':
        formula = 'AUC ~ 1+ (Condition * Q("Pop Size"))'
    elif case == 'glm':
        formula = 'R2 ~ Condition'
    elif case == 'osi':  # Add more cases as needed
        formula = 'OSI ~ Condition'
    elif case == 'maxResp':  
        formula = 'maxResp ~ Condition'
    elif case == 'coeff':
        formula = 'Coeff ~ Condition'
    elif case == 'noise':
        formula = 'NoiseCoeff ~ Condition'
    elif case == 'signal':
        formula = 'SignalCoeff ~ Condition'
    elif case == 'fr':
        formula = 'FR ~ Condition'
    elif case == 'rel':
        formula = 'Rel ~ Condition'
    else: 
        formula = f'{case} ~ Condition'
    
    # Define the variance component formula (vc_formula)
    vc_formula = '1'

    # Fit the mixed-effects model
    mlm = smf.mixedlm(formula, df, groups=df['animalID'], re_formula=vc_formula).fit()

    # Print the summary
    print(mlm.summary())
    
    # Return the fitted model object
    return mlm

#%% Test 
