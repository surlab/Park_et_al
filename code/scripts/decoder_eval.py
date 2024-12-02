# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:03:59 2024

@author: jihop
"""

# Import packages needed for running the code
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import pingouin as pg
from functions import compute

#%% Define paths

# Home directory where the repository is cloned 
# Make sure to change the information accordingly
homepath = os.path.join('C:','Users','jihop','Documents','Park_et_al_2024','')
# Directory containing data files
datapath = os.path.join(homepath,'sample-data','decoder','')
# Directory to save output files
savepath = os.path.join(homepath,'results','sample-output','')
# Directory to save plots 
plotpath = os.path.join(homepath,'results','sample-plots','')

#%% Load the gratings DataFrame file containing AUROC of all sessions

dfGrat = pd.read_csv(datapath+'Decoder_gratings_AUC_results_DF.csv')

#%% T tests at each pop size

popsize= 25

# List of 'Pop Size' values to iterate over
pop_size_values = range(5, popsize+1, 5)

# Dictionary to store t-test results for each 'Pop Size'
t_test_results = {}

for pop_size in pop_size_values:
    
    control_post = dfGrat[(dfGrat['Group'] == 'Control') & (dfGrat['Pop Size'] == pop_size)]
    exp_post = dfGrat[(dfGrat['Group'] == 'Exp') & (dfGrat['Pop Size'] == pop_size)]

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
hold = dfGrat.groupby(['animalID', 'Pop Size', 'Group'])['AUC'].mean().reset_index()

# Perform ANOVA on 'AUC' between 'Group' and 'Pop Size'
anova_result = pg.anova(data=hold[hold['Pop Size']<=25], dv='AUC', between=['Group', 'Pop Size'])

# Print ANOVA result
print(anova_result)

#%% Visualize decoder results 

pGratAUC = compute.mlm_stats(dfGrat, 'decoder').pvalues['Group[T.Exp]']
aovGrat = pg.anova(dfGrat,dv='AUC',between=['Group','Pop Size'])

plt.figure(figsize=(10,5))
popsize = 25

nSessionControl = dfGrat[dfGrat.Group=='Control']['Session'].nunique()
nSessionExp = dfGrat[dfGrat.Group=='Exp']['Session'].nunique()

sns.catplot(data=dfGrat[dfGrat['Pop Size'] <=popsize], x='Pop Size', y='AUC', kind='box', hue = 'Group',
            errorbar=None)

plt.suptitle(f'Gratings mean AUC score box plot (n<={popsize})', fontsize=13, y=1.1)
plt.ylim(0.3, 1.0)  # Set y-axis limits
plt.text(0.5, 0.25, f'nSessionControl: {nSessionControl}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.text(0.5, 0.20, f'nSessionExp: {nSessionExp}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
# plt.savefig(plotpath+'Gratings mean AUC score box plot.svg')
plt.show()

sns.catplot(data=dfGrat[dfGrat['Pop Size'] <=popsize], x='Pop Size', y='AUC', 
            kind='point', hue = 'Group',
            errorbar=None)

plt.suptitle(f'Gratings mean AUC score point plot (n<={popsize})', fontsize=13, y=1.1)
# plt.savefig(plotpath+'Gratings mean AUC score point plot.svg')
plt.show()

#%% Load the movies DataFrame file containing AUROC of all sessions

dfMov = pd.read_csv(datapath+'Decoder_movies_AUC_results_DF.csv')

#%% T tests at each pop size

popsize= 25

# List of 'Pop Size' values to iterate over
pop_size_values = range(5, popsize+1, 5)

# Dictionary to store t-test results for each 'Pop Size'
t_test_results = {}

for pop_size in pop_size_values:
    
    control_post = dfMov[(dfMov['Group'] == 'Control') & (dfMov['Pop Size'] == pop_size)]
    exp_post = dfMov[(dfMov['Group'] == 'Exp') & (dfMov['Pop Size'] == pop_size)]

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
hold = dfMov.groupby(['animalID', 'Pop Size', 'Group'])['AUC'].mean().reset_index()

# Perform ANOVA on 'AUC' between 'Group' and 'Pop Size'
anova_result = pg.anova(data=hold[hold['Pop Size']<=25], dv='AUC', between=['Group', 'Pop Size'])

# Print ANOVA result
print(anova_result)

#%% Visualize decoder results 

pMovAUC = compute.mlm_stats(dfMov, 'decoder').pvalues['Group[T.Exp]']

plt.figure(figsize=(10,5))
popsize = 25

nSessionControl = dfMov[dfMov.Group=='Control']['Session'].nunique()
nSessionExp = dfMov[dfMov.Group=='Exp']['Session'].nunique()

sns.catplot(data=dfMov[dfMov['Pop Size'] <=popsize], x='Pop Size', y='AUC', kind='box', hue = 'Group',
            errorbar=None)

plt.suptitle(f'Movies mean AUC score box plot (n<={popsize})', fontsize=13, y=1.1)
plt.ylim(0.3, 1.0)  # Set y-axis limits
plt.text(0.5, 0.25, f'nSessionControl: {nSessionControl}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.text(0.5, 0.20, f'nSessionExp: {nSessionExp}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.savefig(plotpath+'Movies mean AUC score box plot.svg')
plt.show()

sns.catplot(data=dfMov[dfMov['Pop Size'] <=popsize], x='Pop Size', y='AUC', 
            kind='point', hue = 'Group',
            errorbar=None)

plt.suptitle(f'Movies mean AUC score point plot (n<={popsize})', fontsize=13, y=1.1)
plt.savefig(plotpath+'Movies mean AUC score point plot.svg')
plt.show()

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