# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:35:00 2024

@author: jihop

# Plot different basic properties by visual stimuli

"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import mannwhitneyu

### GRATINGS ###

# Plot OSI >= 0.3
def plot_OSI_greater_than_x(df, x, savepath):
    # Define the OSI value ranges for the pie chart based on the threshold x
    osi_ranges = ['< ' + str(x), '>= ' + str(x)]

    # Calculate the counts of cells in each OSI value range for 'Control' group
    nOSIControl = [len(df[(df['Group'] == 'Control') & (df['OSI'] < x)]),
                   len(df[(df['Group'] == 'Control') & (df['OSI'] >= x)])]

    # Calculate the counts of cells in each OSI value range for 'Exp' group
    nOSIExp = [len(df[(df['Group'] == 'Exp') & (df['OSI'] < x)]),
               len(df[(df['Group'] == 'Exp') & (df['OSI'] >= x)])]

    plt.figure(figsize=(10, 5))

    # Create a pie chart for 'Control' group
    plt.subplot(1, 2, 1)
    plt.pie(nOSIControl, labels=osi_ranges, autopct='%1.1f%%')
    plt.title('OSI Distribution for Control Group')

    # Create a pie chart for 'Exp' group
    plt.subplot(1, 2, 2)
    plt.pie(nOSIExp, labels=osi_ranges, autopct='%1.1f%%')
    plt.title('OSI Distribution for Experimental Group')

    plt.tight_layout()
    plt.savefig(savepath+'OSI greater than 0.3.svg')
    plt.show()
    
    
# Plot average trial traces of each neuron 
def plot_single_neuron_trial_average(df,sessDur,nRep,tOff,tOn,freqNeuro):
    time = np.arange(0, sessDur, 1)
    
    for session in range(len(df)):
        sessionName = df['Session'][session]
        print(f'Processing session: {sessionName}')
        gratStruct = df['GratStruct'][session]
        nUnits = gratStruct['roi'].shape[1]
        dff = gratStruct['neuro_trial']
        idxVis = gratStruct['idx'][0][0]['vis'].astype(bool)
        
        for n in range(nUnits):
            grat_sig = np.concatenate(gratStruct['roi'][0][n]['grat_dir']['grat_sig'][0]).astype(bool)
            visResp = idxVis[n]
            roiDff = dff[n]
            avgDff = np.mean(roiDff,axis=1)
            
            fig, ax = plt.subplots(figsize=(15, 4))
            
            for i in range(nRep):
            
                if visResp == False:
                    ax.plot(time, roiDff[:,i], color='gray', lw=1, alpha=0.5)
                
                if visResp == True:
                    ax.plot(time, roiDff[:,i], color='darksalmon', lw=1, alpha=0.7)
                    
            ax.plot(time,avgDff,lw=2, color='crimson')
                    
            # Define the stimulus "on" period
            stimulusOnset = int(tOff * freqNeuro)
            stimulusDuration = int(tOn * freqNeuro)
            
            # Calculate the positions for label placement
            label_positions = [(angle, angle + 45) for angle in range(0, 360, 45)]
            
            for start_angle, is_significant in zip(label_positions, grat_sig):
                label_text = f'{start_angle[0]}°'
                ax.fill_between(   
                    [stimulusOnset, stimulusOnset + stimulusDuration],
                    -1, 4.0,  # Adjust the y-limits as needed
                    color='khaki', alpha=0.5, edgecolor='none'
                )
                
                # Calculate the position for placing the label
                label_position_x = (stimulusOnset + stimulusDuration / 2)
                label_position_y = 4.1  # Adjust the y-position of the label
                
                if is_significant:
                    ax.text(
                        label_position_x, label_position_y, label_text,
                        ha='center', fontweight='bold', fontsize=15, color='blue'
                    )
                else:
                    ax.text(
                        label_position_x, label_position_y, label_text,
                        ha='center', fontsize=15, color='black'
                    )
                
                stimulusOnset += int((tOff + tOn) * freqNeuro)  # Move to the next stimulus "on" period

            ax.set_ylim([-1, 4.0])
            ax.set_xlim([0, sessDur])
            sns.despine()
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("dff")
            
            # Set x ticks every 10 seconds
            tick_locations = np.arange(0, sessDur, 10 * freqNeuro)
            ax.set_xticks(tick_locations)
            ax.set_xticklabels([str(int(t / freqNeuro)) for t in tick_locations])  # Convert to seconds
            plt.title(f'Neuron = {n+1}', y=1.1, fontsize=20)
            plt.show()
    
    
# Plot Tuning curves 
# def plot_tuning_curve():
    
#     listTC = []
#     listVmfx = []
#     # listVmfit = []
#     listVisresp = []
    
    
#     for n in range(len(zippedList)):
#         hold = zippedList[n]
#         sessionName = hold[0]
#         roiStruct = hold[1]
        
#         nUnits = roiStruct['roi'].shape[1]
        
#         angle_vector = roiStruct['roi'][0]['tc'][0]['info'][0][0]['angle_vec'][0][0][0]
        
#         # # Create an array of angles in degrees using linspace
#         # vmAngles = np.linspace(angle_vector[0], angle_vector[-1], 20000)
    
#         # # Convert angles to radians
#         # vmAngRad = vmAngles * (np.pi / 180)
        
#         # Determine the color based on sessionName using regex
#         if re.search(r'mrcuts24|mrcuts25|mrcuts26|mrcuts27', sessionName):
#             session_color = 'blue'
#         elif re.search(r'mrcuts28|mrcuts29|mrcuts30', sessionName):
#             session_color = 'black'
#         else:
#             session_color = 'red'  # Default color
        
#         listVmfit = []
        
#         for i in range(nUnits):
        
#             tc = roiStruct['roi'][0]['tc'][i]['mean_r'][0][0]['shift'][0][0][0]
#             vmfx = roiStruct['roi'][0]['tc'][i]['mean_r'][0][0]['vm_fx'][0][0][0]
#             vmfit = roiStruct['roi'][0]['tc'][i]['mean_r'][0][0]['vm_fit'][0][0][0]
#             sterr = roiStruct['roi'][0]['resp'][i]['mean_r'][0][0]['stat'][0][0]['sterr'][0][0][0]
#             # visresp = roiStruct['idx']['vis'][0][i].astype(bool)
            
#             listVmfit.append(vmfit)
            
#             angle_vm = np.linspace(angle_vector[0], angle_vector[-1], 20000)
            
#             # Errorbar for tc_mean and tc_sterr with specified colors
#             plt.errorbar(angle_vector, tc, yerr=sterr, fmt='.', markersize=10, linewidth=1, color=session_color, label='TC')
        
#         avgVmfit = np.mean(listVmfit)
        
        
#         # Set the x-axis limit
#         plt.xlim(angle_vector[0], angle_vector[-1])
#         plt.ylim([-0.5, 3])
        
#         # Place the average vmfit
#         plt.text(200, 2.5, f'Average GOF: {avgVmfit:0.3f}')
        
#         # Set labels, title, and legend for the session
#         plt.xlabel('Orientation (°)')
#         plt.ylabel('DF/F')
#         plt.title(sessionName, fontsize=15)
#         sns.despine()
        
#         # plt.savefig(plotDrive+f'TC sterr {sessionName}.svg', format='svg')
#         plt.show()  # Display the plot for the session
    
### Any stimType ###
def plot_with_pval(df,dv,plottype,orderGroup):
    
    pval = mannwhitneyu(x=df[df['Group']=='Control'][dv],y=df[df['Group']=='Exp'][dv]).pvalue
    
    # Create the plot
    g = sns.catplot(data=df, x='Group', y=dv, kind=plottype, order=orderGroup)
    
    # Add text with p-value slightly below the title
    ylim = g.ax.get_ylim()
    text_y = ylim[1] *0.9  # Adjust the fraction as needed
    plt.text(0.5, text_y, f'p = {pval:.4e}', ha='center', va='bottom')
    plt.title(f'{dv} plotted with pvalue',fontsize=15)
    # Show the plot
    plt.show()