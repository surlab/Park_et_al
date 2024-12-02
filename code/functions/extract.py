#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:00:32 2024

@author: jihopark

# Extract data from master.mat files after MATLAB preprocessing
# Extracts DFF, gratings, and movies analyzed data 
# Save as master DFs to access for analyses in python
# Should work for different kinds of master files 
 
"""

import os
import re
import scipy.io
import pandas as pd
import numpy as np
from scipy.stats import zscore

# Read master files from data directory
def load_data(datapath):
    
    listFiles = []
    listMat = []
    listNames = []
    listGratStruct = []
    listNeuroStruct = []
    listMovStruct = []
    listPupil = []
    listWheel = []
    listPupilEvents = []
    listWheelEvents = []
    listStates = []
    
    for subdir, dirs, files in os.walk(datapath):
        for file in files:
            # All files that end with .mat
            if file.endswith('master.mat'):
                file_path = os.path.join(subdir, file) 
                listFiles.append(file_path)

    print(f'Number of master files = {len(listFiles)}')

    for file in listFiles:
        # Extract the file name and save in listNames
        fileName = os.path.basename(file)
        print(f'Loading {fileName}...')
        
        # Load the MATLAB file
        mat = scipy.io.loadmat(file)
        # Extract the data structure and save in listDataStruct
        master = mat['master']
        if 'data' in master.dtype.fields:
            
            pattern = r'\_master\.mat'
            sessionName = re.sub(pattern, '', fileName)
            listNames.append(sessionName)
            listMat.append(mat)
            data = master['data'][0][0]
            neuro = data['neuro'][0][0]
            listNeuroStruct.append(neuro)
            
            if 'analysis' in master.dtype.fields and 'neuro' in master['analysis'][0][0].dtype.fields:
                analysis = master['analysis'][0][0]['neuro'][0][0]
                
                # Check if 'grat' field exists in the analysis structure
                if 'grat' in analysis.dtype.fields:
                    grat = analysis['grat'][0][0][0][0]
                else:
                    grat = None
                    print("This file does NOT contain grat structure!")
                
                
                # Check if 'mov' field exists in the dataset
                if 'mov' in analysis.dtype.fields:
                    mov = analysis['mov'][0][0][0][0]
                else:
                    mov = None
                    print("This file does NOT contain mov structure!")
            
            else:
                analysis = None
                grat = None
                mov = None
                print("This file is a spo data!")
            
            listGratStruct.append(grat)
            listMovStruct.append(mov)
            
            if 'analysis' in master.dtype.fields and 'pupil' in master['analysis'][0][0].dtype.fields and 'wheel' in master['analysis'][0][0].dtype.fields:
                eventsPupil = master['analysis'][0][0]['pupil'][0][0]['events']
                eventsWheel = master['analysis'][0][0]['wheel'][0][0]['events']
            else:
                eventsPupil = None
                eventsWheel = None
                print("This file does NOT have behavior analysis data!")
            
            if 'analysis' in master.dtype.fields and 'state' in master['analysis'][0][0].dtype.fields:
                states = master['analysis'][0][0]['state'][0][0]
            else:
                states = None
                print("This file does NOT have state information!")
                
            listPupilEvents.append(eventsPupil)
            listWheelEvents.append(eventsWheel)
            listStates.append(states)
            
            # Check if 'pupil' and 'wheel' fields exist in the dataset
            if 'pupil' in data.dtype.fields and 'wheel' in data.dtype.fields:
                pupil = mat['master']['data'][0][0]['pupil'][0][0][0][0]['diam']['filt_zsc'][0][0][0]  # Get the filtered pupil data
                wheel = mat['master']['data'][0][0]['wheel'][0][0][0][0]['norm'][0]  # Get the normalized wheel data
            else:
                pupil = None
                wheel = None
                
            listPupil.append(pupil)
            listWheel.append(wheel)
        
        else:
            pass
        
    print(f'Total number of files = {len(listNames)}')
    print(f'Total number of neuro structures loaded = {len(listNeuroStruct)}')
    print(f'Total number of grat structures loaded = {len(listGratStruct)}')
    print(f'Total number of mov structures loaded = {len(listMovStruct)}')
    print(f'Total number of pupil data loaded = {len(listPupil)}')
    print(f'Total number of wheel data loaded = {len(listWheel)}')
    
    # Combine the lists into a DataFrame
    dfMaster = pd.DataFrame({
        'Session': listNames,
        'Mat': listMat,
        'NeuroStruct': listNeuroStruct,
        'GratStruct': listGratStruct,
        'MovStruct': listMovStruct,
        'Pupil': listPupil,
        'Wheel': listWheel,
        'PupilStruct': listPupilEvents,
        'WheelStruct': listWheelEvents,
        'States': listStates,
    })
    
    
    # Extract the date and animalID and save in the df
    for i in range(len(dfMaster)):
        session = dfMaster.iloc[i]['Session']
        
        pattern = r'(\d{6}).*'
        match = re.search(pattern, session)
        
        if match:
            date = match.group(1)
            dfMaster.at[i, 'Date'] = date
        
        # Identify animalID
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
    
    return dfMaster

def get_animalID(dfMaster):
    for i in range(len(dfMaster)):
        session = dfMaster.iloc[i]['Session']
        
        pattern = r'(\d{6}).*'
        match = re.search(pattern, session)
        
        if match:
            date = match.group(1)
            dfMaster.at[i, 'Date'] = date
        
        # Identify animalID
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
    
    return dfMaster

# Assign group based on animalID (update for different animals)
def get_group(anID):
    if anID == 'an316' or anID == 'an318' or anID == 'mrcuts07' or anID == 'mrcuts28' or anID == 'mrcuts29' or anID == 'mrcuts30':
        return 'Control'
    elif anID == 'mrcuts24' or anID == 'mrcuts25' or anID == 'mrcuts26' or anID == 'mrcuts27':
        return 'Exp'
    else:
        return None
    
# Assign condition based on date (update for different animals)
def get_condition(date):
    if date == '230117' or date == '230120' or date == '230116' or date == '221005' or date == '221006' or date == '230118':
        return 'Pre'
    elif date == '230218' or date == '230219' or date == '230211' or date == '221025' or date == '221026':
        return 'Post'
    else:
        return None

# Extract just the spontaneous files 
def extract_spo(dfMaster):
    idxSpo = dfMaster[dfMaster['GratStruct'].isna() & dfMaster['MovStruct'].isna()].index
    df = dfMaster.loc[idxSpo].reset_index(drop=True)
    df.drop(columns=['Mat','GratStruct','MovStruct'], inplace=True)
    return df 

# Extract just the gratings file 
def extract_grat(dfMaster):
    idxGrat = dfMaster[dfMaster['GratStruct'].notna()].index
    df = dfMaster.loc[idxGrat].reset_index(drop=True)
    df.drop(columns=['Mat','MovStruct'], inplace=True)
    return df

# Extract just the gratings file 
def extract_mov(dfMaster):
    idxMov = dfMaster[dfMaster['MovStruct'].notna()].index
    df = dfMaster.loc[idxMov].reset_index(drop=True)
    df.drop(columns=['Mat','GratStruct'], inplace=True)
    return df

# Extract DFF
def get_dff(df):
    listDff = []
    for i in range(len(df)):
        dff = df['NeuroStruct'].iloc[i]['dff'][0][0]
        listDff.append(dff)
    df['DFF'] = listDff
    df.drop(columns=['NeuroStruct'], inplace=True)
    return df 

# Extract visual response properties from grating files 
def get_osi(df):
    listOSI = []
    for i in range(len(df)):
        osi = df['GratStruct'].iloc[i]['osi']
        listOSI.append(osi)
    df['OSI'] = listOSI
    return df
    
def get_visresp(df):
    listVisresp = []
    for i in range(len(df)):
        visresp = df['GratStruct'].iloc[i]['idx']['vis'][0][0].astype(bool)
        listVisresp.append(visresp)
    df['VisResp'] = listVisresp
    return df

def get_tc(df):
    listTC = []
    for i in range(len(df)):
        dfSession = df['GratStruct'].iloc[i]
        nUnits = dfSession['roi'].shape[1]
        listBySession = []
        for n in range(nUnits):
            tc = dfSession['roi'][0][n]['tc'][0]['mean_r'][0][0]['shift'][0][0]
            listBySession.append(tc)
        listTC.append(listBySession)
    df['TC'] = listTC
    return df

def get_pref_angle(df):
    listAngles = []
    for i in range(len(df)):
        dfSession = df['GratStruct'].iloc[i]
        nUnits = dfSession['roi'].shape[1]
        listBySession = []
        for n in range(nUnits):
            tc = dfSession['roi'][0][n]['tc'][0]['mean_r'][0][0]['pref_grat'][0][0][0]
            listBySession.append(tc)
        listAngles.append(listBySession)
    df['Pref_angle'] = listAngles
    return df
    

def get_vmfit(df):
    listVM = []
    for i in range(len(df)):
        dfSession = df['GratStruct'].iloc[i]
        nUnits = dfSession['roi'].shape[1]
        listBySession = []
        for n in range(nUnits):
            vm = dfSession['roi'][0][n]['tc'][0]['mean_r'][0][0]['vm_fx'][0][0]
            listBySession.append(vm)
        listVM.append(listBySession)
    df['VM'] = listVM
    return df

# Extract reliability indices from movies files 
def get_rel(df):
    listRel = []
    for i in range(len(df)):
        rel = np.concatenate(df['MovStruct'].iloc[i]['rel'])
        listRel.append(rel)
    df['Rel'] = listRel
    return df


# Get only one type of stimulus type data
def get_session(df,keyword):
    # Initialize an empty list to store rows matching the condition
    filtered_rows = []
    
    # Iterate over each row in the DataFrame
    for i in range(len(df)):
        name = df['Session'][i]
        
        # Check if the stimType is contained within the session name
        if keyword in name:
            # Append the row to the list of filtered rows
            filtered_rows.append(df.iloc[i])
    
    # Concatenate the filtered rows into a DataFrame
    filtered_df = pd.concat(filtered_rows, axis=1).T.reset_index(drop=True)
    
    return filtered_df

#%% Extract spiking activity data

# Read spks.npy and iscell.npy files from suite2p output and save as df
def load_spks_data(datapath):
    
    listNames = []
    listSpks = []
    listIscell = []
    
    for subdir, dirs, files in os.walk(datapath):
        for file in files:
            if file.startswith('spks'):
                
                parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(subdir)))
                listNames.append(os.path.basename(parent_dir))
                
                file_path = os.path.join(subdir, file) 
                listSpks.append(file_path)
                
                
            if file.startswith('iscell'):
                file_path = os.path.join(subdir, file) 
                listIscell.append(file_path)

    print(f'Number of spks files = {len(listSpks)}')
    
    dataSpks = [np.load(file) for file in listSpks]
    dataIscell = [np.load(file) for file in listIscell]
    
    listSpksNew = []
    listSpksNewZ = []
    
    for (name, spks, iscell) in zip(listNames, dataSpks, dataIscell):
        
        print(f'Processing {name}...')
        spks = spks
        nUnits = spks.shape[0]
        idxCell = iscell[:, 0].astype(bool)
        
        # Check if the nRoi of spks and iscell match
        if nUnits == iscell.shape[0]:
            spksNew = spks[idxCell]
            spksNewZ = zscore(spksNew, axis=1)
        else:
            spksNew = None
            spksNewZ = None 
            print("This file does not have matching spks.npy and iscell.npy sizes!")
        
        listSpksNew.append(spksNew)
        listSpksNewZ.append(spksNewZ)
        
    # Combine the lists into a DataFrame
    dfMaster = pd.DataFrame({
        'Session': listNames,
        'Spks': listSpksNew,
        'zSpks': listSpksNewZ
    })
    
    for i in range(len(dfMaster)):
        session = dfMaster.iloc[i]['Session']

        pattern = r'(\d{6}).*'
        match = re.search(pattern, session)
        
        if match:
            date = match.group(1)
            dfMaster.at[i, 'Date'] = date
        
        # Identify animalID
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

    return dfMaster

#%% Test the code

# datapath = os.path.join('/Users','jihopark','Google Drive','My Drive','mrcuts','data','master','')
# datapath =  os.path.join('G:','My Drive','mrcuts','data','master','')
        
# dfMaster = load_data(datapath)

#%% Test the code 2

# df = extract_grat(dfMaster)

#%% Test the code 3 (spks)

# datapath =  os.path.join('G:','My Drive','mrcuts','data','preprocessed','')

# dfMaster = load_spks_data(datapath)


    