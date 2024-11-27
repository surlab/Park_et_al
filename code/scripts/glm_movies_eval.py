# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:05:31 2024

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

from sklearn.linear_model import Ridge, TweedieRegressor, Lasso, RidgeCV
from sklearn.model_selection import train_test_split

sys.path.append('C:\\Users\\jihop\\Documents\\GitHub\\neuron-analysis\\functions\\')
sys.path.append('/Users/jihopark/Documents/GitHub/neuron-analysis/functions/')
import extract
import compute


# Change the font for plotting
plt.rcParams['font.family'] = 'Arial'

#%% Define variables  (MAC)

dataType = 'constitutive'

analysisDrive = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','')
dataDrive = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','data','master','')
glmDrive = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','GLM',dataType,'')
plotDrive = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','plots',dataType,'')


#%% Define variables  (Windows)

dataType = 'constitutive'

analysisDrive = os.path.join('G:','My Drive','mrcuts','analysis','')
dataDrive = os.path.join('G:','My Drive','mrcuts','data','master','')
glmDrive = os.path.join('G:','My Drive','mrcuts','analysis','GLM',dataType,'')
plotDrive = os.path.join('G:','My Drive','mrcuts','analysis','plots',dataType,'')

#%% Read all files

glmFiles = []

for subdir, dirs, files in os.walk(glmDrive):
    for file in files:
        if file.endswith('_glm_results_with_weights.csv'):
            file_path = os.path.join(subdir, file) 
            glmFiles.append(file_path)

#%% Load the data & make sure to import beta in numerical format


# Define the converter function
def str_to_array(s):
    # Remove unwanted characters and split into components
    array_str = s.replace('array(', '').replace('[', '').replace(']', '').replace(')', '').replace('\n', '').strip()
    # Convert the cleaned string to a list of floats
    beta_list = list(map(float, array_str.split(',')))
    # Convert the list to a numpy array
    beta_array = np.array(beta_list)
    # Reshape the array to [50, 20]
    reshaped_array = beta_array.reshape(30, 20) # different combination of 30 neurons x avg beta weights of 20 neurons used in each combination
    return reshaped_array
    # return beta_array

glmList = []

# loop through the file paths and load each csv file into a dataframe
for file in glmFiles:
    # df = pd.read_csv(file)
    df = pd.read_csv(file, converters={'Beta': str_to_array})
    glmList.append(df)
    
#%% Load all dfs into one

dfCombined = pd.concat(glmList,ignore_index=True)

dfMaster = dfCombined.drop('Unnamed: 0',axis=1)

dfMaster = extract.get_animalID(dfMaster)

dfMaster['Group'] = dfMaster['animalID'].apply(extract.get_group)

# Filter out the problematic sessions based on heatmaps (ROI labeling seems to be off)
sessions_to_drop = ['230809_mrcuts28_fov4_mov','230804_mrcuts26_fov3_mov']
df = dfMaster[~dfMaster['Session'].isin(sessions_to_drop)].reset_index(drop=True)

#%% Treat each beta as an individual point 

dfSample = df.explode(['Beta'])

#%% Let's get the average or (?) of the beta weights across the trial iterations for each neuron

df['BetaAvg'] = None

for n in range(len(df)):
    betaArray = df['Beta'].iloc[n]
    betaSorted = np.sort(betaArray,axis=1)
    betaMean = np.mean(betaSorted,axis=0)
    df.at[n, 'BetaAvg'] = betaMean
    
#%% Explode BetaAvg

dfBeta = df.explode(['BetaAvg'])
dfBeta = dfBeta.drop(columns='Beta')
dfBeta['BetaAvg'] = dfBeta['BetaAvg'].astype(float)

#%% For each neuron, get how many neurons had beta values above a certain value

# Set the threshold value
threshold = 0.05

# Initialize a list to store counts
counts = []

# For each neuron (row) in the DataFrame
for neuron in df.index:
    # Retrieve the array from 'BetaAvg'
    beta_array = df.loc[neuron, 'Beta']
    
    # Count values above the threshold
    count_above_threshold = np.sum(np.array(beta_array) > threshold)
    count_by_trial = count_above_threshold / 30
    
    # Append the count to the list
    # counts.append(count_above_threshold)
    counts.append(count_by_trial)

# Optionally, add counts to DataFrame or print them
df['Counts'] = counts

print(df)

#%% For each neuron, find the maximum beta value from all values 

betaMaxList = []

# For each neuron (row) in the DataFrame
for neuron in df.index:
    # Retrieve the array from 'BetaAvg'
    betaMax = np.max(df['Beta'].iloc[neuron])
    betaMaxList.append(betaMax)
    
df['BetaMax'] = betaMaxList
    

print(df)

#%% For each neuron, find the top 5 values from Beta

# Create an empty list to store the top 5 values for each neuron
betaTop5List = []

# For each neuron (row) in the DataFrame
for neuron in df.index:
    # Retrieve the array from 'Beta'
    betaArray = df['Beta'].iloc[neuron].flatten()  # Flatten the array to 1D
    
    # Get the top 5 values
    top_5_values = np.partition(betaArray, -5)[-5:]
    
    # Sort the top 5 values in descending order
    top_5_values = np.sort(top_5_values)[::-1]
    
    # Append the top 5 values to the list
    betaTop5List.append(top_5_values)

# Add the top 5 values list as a new column in the DataFrame
df['BetaTop5'] = betaTop5List

print(df)

dfTop5 = df.explode(['BetaTop5'])
sns.catplot(dfTop5,x='Group',y='BetaTop5',hue='Session',dodge=True,kind='point')

#%% Plot the number of cells (counts) with beta weights ahove the threshold for each neuron

sns.pointplot(df,x='Group',y='Counts',hue='Session',legend=False,dodge=True)
sns.catplot(df,x='Group',y='Counts',legend=False,dodge=True,size=5)
sns.catplot(df,x='Group',y='BetaMax',hue='animalID',dodge=True)


#%% Plot the proportion of betaMax values

# Define the bins and labels for 3 ranges
bins = [0, 0.05, 0.1, np.inf]
labels = ['0 to 0.05', '0.05 to 0.1', 'Above 0.1']

# Bin the 'Counts' data
df['Range'] = pd.cut(df['BetaMax'], bins=bins, labels=labels)

# Define the new consistent order for the 3-bin system
consistent_order = ['0 to 0.05', '0.05 to 0.1', 'Above 0.1']

# Reindex the value counts for control group to ensure consistent label order
range_counts_control = df[df['Group']=='Control']['Range'].value_counts().reindex(consistent_order, fill_value=0)

# Reindex the value counts for exp group to ensure consistent label order
range_counts_exp = df[df['Group']=='Exp']['Range'].value_counts().reindex(consistent_order, fill_value=0)

# Plot pie charts 
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].pie(range_counts_control, labels=range_counts_control.index, autopct='%1.1f%%', startangle=90, counterclock=False)
axes[0].set_title('CONTROL')

axes[1].pie(range_counts_exp, labels=range_counts_exp.index, autopct='%1.1f%%', startangle=90, counterclock=False)
axes[1].set_title('EXP')

fig.suptitle('Proportion of cells by BetaMax')
# fig.savefig(plotDrive+'Proportion of cells by BetaMax (GLM movies).svg', format='svg')
plt.show()


#%% Different ways to show percentage differences 

# Create a grouped bar chart
range_counts_control = df[df['Group'] == 'Control']['Range'].value_counts(normalize=True).reindex(consistent_order, fill_value=0)
range_counts_exp = df[df['Group'] == 'Exp']['Range'].value_counts(normalize=True).reindex(consistent_order, fill_value=0)

# Combine into a DataFrame for plotting
combined_df = pd.DataFrame({'Control': range_counts_control, 'Exp': range_counts_exp})

# Plot the grouped bar chart
combined_df.plot(kind='bar', figsize=(10, 6))
plt.title('Proportion of Cells by BetaMax (Control vs Exp)')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.legend(title='Group')
plt.savefig(plotDrive+'Proportion of cells by BetaMax (GLM movies) (individual bars).svg', format='svg')
plt.show()

# Normalize the data to sum to 1 for each group
combined_df_normalized = combined_df.div(combined_df.sum(axis=1), axis=0)

# Plot 100% stacked bar chart
combined_df_normalized.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Proportion of Cells by BetaMax (Control vs Exp)')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.legend(title='Group')
plt.show()


# Combine into a DataFrame for plotting
combined_df = pd.DataFrame({'Control': range_counts_control, 'Exp': range_counts_exp})

# Create the stacked bar plot
fig, ax = plt.subplots(figsize=(8, 6))
combined_df.T.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# Add labels and title
ax.set_ylabel('Proportion')
ax.set_title('Proportion of Cells by BetaMax Range and Group')
ax.legend(title='BetaMax Range')
plt.xticks(rotation=0)
fig.savefig(plotDrive+'Proportion of cells by BetaMax (GLM movies) (stacked bar).svg', format='svg')
fig.show()

#%% Plot (if Pop Size exists)

sns.catplot(df,x='Pop Size',y='R2',hue='Group',kind='point')
sns.catplot(df,x='Group',y='R2',kind='boxen')
sns.catplot(df,x='Pop Size',y='R2',hue='animalID',kind='point',col='Group')
plt.show()
sns.lineplot(df,x='Pop Size',y='R2',hue='Group')
plt.show()



#%% Plot (no Pop Size)

sns.catplot(df,x='Group',y='R2',kind='point')
sns.catplot(df,x='Group',y='R2',kind='boxen')
sns.catplot(df,x='Group',y='R2',hue='animalID',kind='point')
plt.show()
sns.lineplot(df,x='Group',y='R2')
plt.show()

# sns.relplot(df,x='Pop Size',y='R2',hue='Session',kind='line',col='Group',legend=False)
# plt.show()
#%% Find the Pop Size at which R2 maximizes

result = df.loc[df.groupby(['Session', 'N'])['R2'].idxmax()]

print(result)

pop_size_counts_control = result[result['Group']=='Control']['Pop Size'].value_counts()
pop_size_counts_exp = result[result['Group']=='Exp']['Pop Size'].value_counts()
print(pop_size_counts_control)
print(pop_size_counts_exp)

#%% Stats test for each pop size

p5 = compute.mlm_stats(df[df['Pop Size']==5],'R2').pvalues['Group[T.Exp]']
p10 = compute.mlm_stats(df[df['Pop Size']==10],'R2').pvalues['Group[T.Exp]']
p15 = compute.mlm_stats(df[df['Pop Size']==15],'R2').pvalues['Group[T.Exp]']
p20 = compute.mlm_stats(df[df['Pop Size']==20],'R2').pvalues['Group[T.Exp]']

#%% Plot (if Pop Size exists)

# Create the seaborn plot
g = sns.displot(df, x='R2', hue='Group', col='Pop Size', log_scale=True, kde=True)

# Annotate each subplot with the corresponding p-value
pop_sizes = [5, 10, 15, 20]
p_values = [p5, p10, p15, p20]

for ax, pop_size, p_value in zip(g.axes.flat, pop_sizes, p_values):
    ax.annotate(f'p-value: {p_value:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=15)

# plt.savefig(plotDrive+'GLM R2 distribution by population size.svg')
plt.show()

#%% Plot (no pop size)

# Create the seaborn plot
g = sns.displot(df, x='R2', hue='Group', log_scale=True, kde=True)

plt.title('GLM R2 distribution p=20 (GLM movies)')
plt.savefig(plotDrive+'GLM R2 distribution p=20 (GLM movies).svg')
plt.show()


#%% Average comparison


# sns.catplot(df,x='Pop Size',y='R2',hue='Group',kind='boxen',log_scale=False)
sns.catplot(df[df['Pop Size']==20],x='Group',y='R2',kind='bar',errorbar='se')
plt.savefig(plotDrive+'GLM R2 at n = 20 (bar).svg')
plt.show()


#%% By session plotting

sns.catplot(df,x='Group',y='R2',hue='Session',kind='point',dodge=True)
plt.savefig(plotDrive+'Average GLM R2 comparison by session at p=20 (GLM movies).svg')
plt.show()

#%% Statistical test between BetaMax values

pBetaMax_greater_than_01 = mannwhitneyu(df[(df['BetaMax']>=0.1)&(df['Group']=='Control')]['BetaMax'],df[(df['BetaMax']>=0.1)&(df['Group']=='Exp')]['BetaMax']).pvalue

pBetaMax_between = mannwhitneyu(df[(df['BetaMax']>=0.05)&(df['BetaMax']<0.1)&(df['Group']=='Control')]['BetaMax'],df[(df['BetaMax']>=0.05)&(df['BetaMax']<0.1)&(df['Group']=='Exp')]['BetaMax']).pvalue

pBetaMax_less_than_005 = mannwhitneyu(df[(df['BetaMax']<0.05)&(df['Group']=='Control')]['BetaMax'],df[(df['BetaMax']<0.05)&(df['Group']=='Exp')]['BetaMax']).pvalue