# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:03:59 2024

@author: jihop
"""

# Import packages needed for running the code
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp, sem, wilcoxon
import palettable 
import pingouin as pg
from functions import extract,compute,plot,glm

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