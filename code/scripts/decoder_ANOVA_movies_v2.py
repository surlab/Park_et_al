#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 22:41:55 2023

@author: jihopark
"""

import numpy as np

import os

import pingouin as pg
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import re

from palettable.wesanderson import Mendl_4_r
from palettable.wesanderson import Moonrise1_5
from palettable.cartocolors.diverging import ArmyRose_2
from palettable.cartocolors.sequential import Purp_7
from palettable.cartocolors.sequential import RedOr_3
from palettable.cartocolors.sequential import Teal_3
from palettable.cartocolors.sequential import Teal_6
from palettable.cartocolors.sequential import PurpOr_6
from palettable.cartocolors.sequential import BluGrn_6
from palettable.cartocolors.sequential import Mint_7
from palettable.colorbrewer.sequential import YlOrRd_6
from palettable.colorbrewer.sequential import Reds_7
from palettable.wesanderson import Aquatic2_5
from palettable.wesanderson import Darjeeling3_5
from palettable.wesanderson import Margot3_4
from palettable.tableau import TableauMedium_10_r

colors2 = ArmyRose_2.hex_colors
colors2_2 = Aquatic2_5.hex_colors
colors3 = RedOr_3.hex_colors
colors3_2 = Teal_3.hex_colors
colors4 = Mendl_4_r.hex_colors
colors4_2 = Margot3_4.hex_colors 
colors5 = Moonrise1_5.hex_colors
colors5_2 = Darjeeling3_5.hex_colors
colors6 = Teal_6.hex_colors
colors6_2 = PurpOr_6.hex_colors
colors6_3 = BluGrn_6.hex_colors
colors6_4 = YlOrRd_6.hex_colors
colors7 = Purp_7.hex_colors
colors7_2 = Mint_7.hex_colors
colors7_3 = Reds_7.hex_colors
colors10 = TableauMedium_10_r.hex_colors 

colorsAll = ['#5f7479','#69b6c8']
colorsAll4 = ['#8f9ea1','#5f7479','#96ccd9','#69b6c8']
colorsControl = ['#8f9ea1','#5f7479'] # for constitutive 
colorsExp = ['#96ccd9','#69b6c8'] # for constitutive

# Change the font for plotting
plt.rcParams['font.family'] = 'Arial'  # Replace 'Arial' with your desired font name


#%% Define paths

mainFolder = os.path.join('/Users','jihopark','Desktop','data','mrcuts','analysis','')
savepath = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','decoder','')

conditionalDrive = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','plots','conditional','')
constDrive = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','plots','constitutive','')
aucDrive = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','analysis','decoder','auc_movies','')

aucFiles = []

#%% Create a list of all files 

for subdir, dirs, files in os.walk(aucDrive):
    for file in files:
        if file.startswith('2308'):
            file_path = os.path.join(subdir, file) 
            aucFiles.append(file_path)

#%% Load the dataframe in a list 

aucList = []

# loop through the file paths and load each csv file into a dataframe
for file in aucFiles:
    df = pd.read_csv(file)
    aucList.append(df)
    
#%% Separate the dataset to PRE and POST DFs 

dfCombined = pd.concat(aucList,ignore_index=True)

dfMaster = dfCombined.drop('Unnamed: 0',axis=1)

dfMaster['Date'] = dfMaster['Date'].astype(int)
dfMaster['Date'] = dfMaster['Date'].astype(str)

print(dfMaster.Date.unique())

#%% Combine control and experimental DFs into one and save (Conditional)

def get_condition(date_value):
    if date_value == '230116' or date_value == '230117' or date_value == '230118' or date_value == '230120' or date_value == '221005' or date_value == '221006' or date_value == '221123' or date_value == '221126':
        return 'PRE'
    elif date_value == '230218' or date_value == '230219' or date_value == '221222' or date_value == '221025' or date_value == '221026':
        return 'POST'
    else:
        return 'None'

def get_group(anID):
    if anID == 'an316' or anID == 'an318' or anID == 'mrcuts07' or anID == 'mrcut316' or anID == 'mrcut318':
        return 'Control'
    elif anID == 'an317' or anID == 'mrcut317' or anID == 'mrcuts13' or anID == 'mrcuts14' or anID == 'mrcuts15' or anID == 'mrcuts16' or anID == 'mrcuts17':
        return 'Exp'
    else:
        return None

dfMaster['Condition'] = dfMaster['Date'].apply(get_condition)
dfMaster['Group'] = dfMaster['Animal'].apply(get_group)

dfMaster.to_csv(savepath+'Decoder_movies_AUC_master_data.csv')

#%% Label the data by animal and date (CONSTITUTIVE)

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
    if anID == 'mrcuts28' or anID == 'mrcuts29' or anID == 'mrcuts30':
        return 'Control'
    elif anID == 'mrcuts24' or anID == 'mrcuts25' or anID == 'mrcuts26' or anID == 'mrcuts27':
        return 'Exp'
    else:
        return None

dfMaster['Group'] = dfMaster['animalID'].apply(get_group)
# dfMaster = dfMaster[dfMaster['animalID'] != 'mrcuts24']


dfMaster.to_csv(savepath+'Decoder_movies_AUC_master_data_constitutive.csv')

#%% Load master data

os.chdir(savepath)
# dfMaster = pd.read_csv('Decoder_movies_AUC_master_data.csv')
dfMaster = pd.read_csv('Decoder_movies_AUC_master_data_constitutive.csv')

#%% Find the mean AUC scores by population size it by session 

sessions = np.unique(dfMaster.Session)

# dfMaster = dfMaster.drop('Unnamed: 0.1',axis=1)
resultsDF = pd.DataFrame()

for sesh in range(len(sessions)):
    subset = dfMaster[dfMaster.Session==sessions[sesh]]
    AUC = subset.groupby('Pop Size')['AUC'].mean()
    
    hold = pd.DataFrame()
    hold['AUC'] = AUC
    hold['Pop Size']= np.unique(subset['Pop Size'])
    hold['Group'] = subset['Group'].iloc[0]
    hold['Animal'] = subset['Animal'].iloc[0]
    hold['Date'] = subset['Date'].iloc[0]
    hold['Session'] = sessions[sesh]
    resultsDF = pd.concat([resultsDF, hold], ignore_index=True)

resultsDF.to_csv(savepath+'Decoder_movies_AUC_results_DF_constitutive.csv')

aov = pg.anova(data=resultsDF, between=['Group','Pop Size'], dv='AUC')
p_val = aov['p-unc'][2] # extract the p-value from ANOVA results

#%% Load resultsDF 

os.chdir(savepath)
resultsDF = pd.read_csv('Decoder_movies_AUC_results_DF_constitutive.csv')
resultsDF = resultsDF.drop('Unnamed: 0',axis=1)
resultsDF['Pop Size'] = resultsDF['Pop Size'].astype(int)

#%% Visualize the resultsDF

sns.relplot(data=resultsDF,x='Animal',y='AUC',palette=colors4_2,height=6, aspect=2)

table = pd.pivot_table(resultsDF,values='AUC',index='Condition',columns=['Group','Animal'])

# table.plot(kind='bar')

#%% Extract only certain time points and pop sizes (constitutive)

orderGroup = ['Control', 'Exp']

dfMaster = resultsDF

sns.catplot(data=dfMaster, x='Pop Size', y='AUC', 
            col='Group', kind='point', 
            errorbar=None)


sns.catplot(data=dfMaster[(dfMaster['Pop Size'] <= 35)], x='Pop Size', y='AUC', 
            col='Animal', kind='point',
            errorbar=None)

plt.suptitle('movies comparison by each animal', y = 1.1)

sns.catplot(data=dfMaster[(dfMaster['Pop Size'] <= 35)], x='Pop Size', y='AUC', 
            kind='point', hue ='Group', hue_order=orderGroup, palette=colorsAll,
            errorbar=None)

plt.suptitle('movies comparison by Condition', y = 1.1)
#%% Plot boxplots 

fig, axs = plt.subplots(1,2,sharey=True,figsize=[15,5])

sns.boxplot(data=resultsDF[resultsDF.Group=='Control'][resultsDF['Pop Size']<=30], 
            x='Pop Size', y='AUC', palette='ch:r=.2', ax=axs[0])

sns.despine()

sns.boxplot(data=resultsDF[resultsDF.Group=='Exp'][resultsDF['Pop Size']<=30], 
            x='Pop Size', y='AUC', palette='ch:r=.2', ax=axs[1])
sns.despine()

plt.suptitle('movies mean AUC scores by population size', fontsize=13)

plt.savefig(constDrive+'movies mean AUC score comparison by population size.svg')
plt.show()

#%% Extract only certain time points and pop sizes (Conditional)

orderCondition = ['PRE', 'POST']
orderGroup = ['Control', 'Exp']

dfMaster = resultsDF[resultsDF['Condition'].isin(['PRE', 'POST'])]

# dfMaster = resultsDF[~resultsDF['Condition'].isin(['POST (~1wk)'])]

dfMaster[dfMaster.Group=='Control']['Session'].nunique()
sns.catplot(data=dfMaster[(dfMaster['Pop Size'] <= 50)], x='Pop Size', y='AUC', 
            col='Group', kind='point', hue ='Condition', hue_order=orderCondition,
            errorbar=None,palette=colors6_3)

sns.catplot(data=dfMaster[(dfMaster['Pop Size'] <= 50)], x='Pop Size', y='AUC', 
            col='Animal', kind='point', hue ='Condition',
            errorbar=None, palette=colors3_2)

sns.catplot(data=dfMaster[(dfMaster['Pop Size'] <= 50)], x='Pop Size', y='AUC', 
            col='Condition', kind='point', hue ='Group',
            errorbar=None)

plt.suptitle('Movies comparison by condition', y = 1.1)

#%% Plot Pop Size by AUC for each group or specific combinations 

popsize = 30
df2 = resultsDF[resultsDF['Pop Size'] <=50]

nSessionControl = df2[df2.Group=='Control']['Session'].nunique()
nSessionExp = df2[df2.Group=='Exp']['Session'].nunique()

sns.catplot(data=df2[df2['Pop Size'] <=popsize], x='Pop Size', y='AUC', kind='box', hue = 'Group',
            errorbar=None)

plt.savefig(constDrive+f'movies mean AUC score box plot (n<={popsize}).svg')
plt.suptitle(f'movies mean AUC score box plot (n<={popsize})', fontsize=13, y=1.1)
plt.show()

sns.catplot(data=df2[df2['Pop Size'] <=popsize], x='Pop Size', y='AUC', 
            kind='point', hue = 'Group',
            errorbar=None)

plt.savefig(constDrive+f'movies mean AUC score POST (n<={popsize}).svg')
plt.suptitle(f'movies mean AUC score POST (n<={popsize})', fontsize=13, y=1.1)
plt.show()

#%% T tests at each pop size

df2 = dfMaster
popsize= 35 

# dfPost = resultsDF[(resultsDF['Pop Size'] <=popsize) & (resultsDF['Condition'].isin(['POST']))]
dfPost = df2[df2['Pop Size']<=popsize]
dfPost['Session'].nunique()

# List of 'Pop Size' values to iterate over
pop_size_values = range(5, 36, 5)

# Dictionary to store t-test results for each 'Pop Size'
t_test_results = {}

for pop_size in pop_size_values:
    
    control_post = dfPost[(dfPost['Group'] == 'Control') & (dfPost['Pop Size'] == pop_size)]
    exp_post = dfPost[(dfPost['Group'] == 'Exp') & (dfPost['Pop Size'] == pop_size)]

    tPost = mannwhitneyu(control_post['AUC'], exp_post['AUC'])

    # Store t-test results in the dictionary
    t_test_results[pop_size] = {'POST': tPost[1]}
    
# Print t-test results for each 'Pop Size'
for pop_size, results in t_test_results.items():
    print(f"Pop Size: {pop_size}")
    print("POST p-value:", results['POST'])
    print("------------------------")
    
#%% Plot them
# Create the catplot
g = sns.catplot(data=dfPost, x='Pop Size', y='AUC', 
                kind='box', hue = 'Group', hue_order=orderGroup, errorbar=None, 
                palette=colorsAll, height=6, aspect=1)

# plt.ylim([0.45,0.95])
plt.ylim([0.3, 1.0])

# Annotate each point with the corresponding p-value
for ax in g.axes.flat:
    for i, pop_size in enumerate(pop_size_values):
        post_p_val = t_test_results[pop_size]['POST']
        x_pos = pop_size/10 - 0.5+ i *0.5 # Adjust the x position
        y_pos = ax.get_ylim()[1]
        text = f"p<={post_p_val:.2f}"
        text_color = 'red' if post_p_val <= 0.05 else 'black'
        ax.text(x_pos, y_pos, text, fontsize=10, color=text_color)


# Save the figure as an SVG file
g.savefig(constDrive + 'post_ttest_popsize v2 (movies).svg', format='svg')

# Show the figure
plt.show()
#%% Compare the POST only

dfPost = resultsDF[resultsDF['Condition'].isin(['POST'])]

animalsToRemove = ['mrcuts13', 'mrcuts14', 'mrcuts17']
# animalsToRemove = ['mrcuts15', 'mrcuts14', 'mrcuts317']

dfNew = resultsDF[~resultsDF['Animal'].isin(animalsToRemove)]


fig, axs = plt.subplots(1,2,sharey=True,figsize=[10,5])

sns.boxplot(data=dfNew[dfNew.Group=='Control'][dfNew['Pop Size']<=30], 
            x='Pop Size', y='AUC', hue='Condition', hue_order=orderCondition, palette='ch:r=.2', ax=axs[0])

sns.despine()
# os.chdir(savepath)
# plt.savefig('Control mean AUC score comparison by population size.svg')
# plt.title('Control mean AUC scores by population size')
# plt.show()

sns.boxplot(data=dfNew[dfNew.Group=='Exp'][dfNew['Pop Size']<=30], 
            x='Pop Size', y='AUC', hue='Condition', hue_order=orderCondition, palette='ch:r=.2', ax=axs[1])
sns.despine()

plt.savefig(constDrive+'Movies mean AUC score comparison by population size.svg')
plt.suptitle('Movies mean AUC scores by population size', fontsize=13)
plt.show()

#%% Plot Pop Size by AUC for each group

sns.catplot(data=dfNew[(dfNew['Pop Size'] <=35)&(dfNew['Condition']=='POST')], x='Pop Size', y='AUC', 
            kind='point', hue = 'Group',
            errorbar=None,palette=colors2)

# Filter rows with maximum 'Pop Size' greater than 35
# dfFiltered = resultsDF[resultsDF['Pop Size'] > 35]
# dfFiltered.groupby(['Animal','Condition']).mean()
# dfFiltered['Session'].nunique()

# sns.catplot(data=dfFiltered, x='Pop Size', y='AUC', 
#             col='Animal', kind='point', hue = 'Condition',
#             errorbar=None,palette=colors2)

pg.anova(data=dfNew, between=['Group','Pop Size','Condition'], dv='AUC')

#%% Make df containing only two Conditions and pop size 

dfFinal = dfNew[(dfNew['Pop Size'] <=30) & (dfNew['Condition'].isin(['POST']))]
# df2.groupby(['Animal','Session','Pop Size']).mean()
dfFinal['Session'].nunique()

for session, count in dfFinal['Session'].value_counts().items():
    print(session)

sns.catplot(data=dfFinal, x='Pop Size', y='AUC', 
            hue='Group', kind='point',
            errorbar=None, palette=colors10)

#%% Plot Pop Size by AUC for each group or specific combinations 

nSessionControl = dfFinal[dfFinal.Group=='Control']['Session'].nunique()
nSessionExp = dfFinal[dfFinal.Group=='Exp']['Session'].nunique()

sns.catplot(data=dfFinal, x='Pop Size', y='AUC', 
            hue ='Group', kind='box', 
            errorbar=None,palette=colors10)

plt.savefig(constDrive+'Movies mean AUC score box plot (Pop Size <=50).svg')
plt.suptitle('Movies mean AUC score box plot (Pop Size <=50)', fontsize=13, y=1.1)
plt.show()

#%%

# sns.catplot(x='Pop Size', y='AUC',hue='Condition',col='Group',
            # data=resultsDF[resultsDF['Pop Size']<=45],kind='box')

sns.catplot(x='Pop Size', y='AUC',hue='Group',
            data=dfFinal,kind='point',errorbar=None,
            palette='BuPu_r')

# sns.catplot(data=resultsDF[resultsDF['Pop Size']<=25],x='Condition',y='AUC',hue='Pop Size',col='Group',kind='box')

anova = pg.anova(data=dfFinal,dv='AUC',between=['Group','Pop Size'])


plt.suptitle(f"Significant p-value: {anova['p-unc'][0]:0.3f}", fontsize=12, fontweight='bold') # add title with p-value

plt.savefig(constDrive+'Movies mean AUC score comparison by population size <= 50.svg')
plt.show()

#%% T tests at each pop size

popsize = 30

# List of 'Pop Size' values to iterate over
pop_size_values = range(5, popsize+1, 5)

# Dictionary to store t-test results for each 'Pop Size'
t_test_results = {}

for pop_size in pop_size_values:
    
    control_post = dfFinal[(dfFinal['Group'] == 'Control') & (dfFinal['Pop Size'] == pop_size)]
    exp_post = dfFinal[(dfFinal['Group'] == 'Exp') & (dfFinal['Pop Size'] == pop_size)]

    tPost = mannwhitneyu(control_post['AUC'], exp_post['AUC'])

    # Store t-test results in the dictionary
    t_test_results[pop_size] = {'POST': tPost['p-val'].values[0]}
    
# Print t-test results for each 'Pop Size'
for pop_size, results in t_test_results.items():
    print(f"Pop Size: {pop_size}")
    print("POST p-value:", results['POST'])
    print("------------------------")
    
#%% Plot them
# Create the catplot
g = sns.catplot(data=dfFinal, x='Pop Size', y='AUC', 
            kind='point', hue = 'Group', errorbar=None,
            palette=colors4, height=6, aspect=1)

# Annotate each point with the corresponding p-value
for ax in g.axes.flat:
    for i, pop_size in enumerate(pop_size_values):
        post_p_val = t_test_results[pop_size]['POST']
        x_pos = pop_size/10 - 0.5+ i *0.5 # Adjust the x position
        y_pos = ax.get_ylim()[1] - 0.02
        text = f"p={post_p_val:.2f}"
        text_color = 'red' if post_p_val <= 0.05 else 'black'
        ax.text(x_pos, y_pos, text, fontsize=10, color=text_color)


# Save the figure as an SVG file
g.savefig(constDrive + 'post_ttest_popsize (movies).svg', format='svg')

# Show the figure
plt.show()

#%% Look at the distribution of the AUC scores 


# Together
# sns.displot(resultsDF[resultsDF['Pop Size'] <= popsize], x='AUC', hue='Condition', 
#             col='Group', kind='kde')

lw = 3 

popsize = 30

# List of 'Pop Size' values to iterate over
pop_size_values = range(5, popsize+1, 5)

# Create a figure with subplots
fig, axs = plt.subplots(1, 6, figsize=(20, 3))

# Iterate over each 'Pop Size'
for i, pop_size in enumerate(pop_size_values):

    sns.kdeplot(data=dfFinal[dfFinal['Pop Size'] == pop_size],
                x='AUC', hue='Group', palette=colors2, 
                linewidth=lw, ax=axs[i])

    # Add title to the top subplot of each column
    axs[i].set_title(f"Pop Size= {pop_size}")
    
# Adjust layout
plt.suptitle(f'AUC Score Distribution by population size (n<={pop_size})', fontsize=16)
sns.despine()
plt.tight_layout()

# Show the combined figure
fig.savefig(constDrive + f' Movies kde plot by population size (n<={popsize}) (filtered).svg', format='svg')
plt.show()

