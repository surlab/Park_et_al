# Sample-data 
We provide sample data, one FOV from each animal consisting of 3 sessions (spontaneous activity, gratings, movies), to run analysis scripts used in the manuscript.

## Overview
This folder contains 4 folders with differentially formatted sample data for each analysis script.
Make sure to specify the datapath in each script in "code/scripts/" to each of the folders. 

1. "master/"
   - Contains the master MATLAB files that end with "_master.mat"
   - Generated from custom-made preprocessing MATLAB scripts that preprocesses and combines 2P fluorescence data, pupil dynamics, and movement. 
   - Each master file contains a nested MATLAB structure
     | session_info   | data    | analysis       |
     | ---    | ---   | ---     |
     | Contains basic information including 2P imaging parameters| Contains the raw and preprocessed neuronal, pupil, and wheel data | Contains the basic properties analyzed based on the visual stimulus type (ex. visual responsiveness, OSI, reliability index) |
   - Each analysis script extracts relevant information from the nested structure for further analyses 

2. "spks/"
   - Contains npy files (spks.npy, iscell.npy) in each session folder
   - 
3. "glm/"
   - Contains data MATLAB files that end with "_data.mat" which serve as input files for run_script.m found in "scripts/glm_gratings/"
   - Each file has been formatted from the master files to extract only necessary variables and reshape the data for run_script.m
   - Each file contains 2 structures and 1 matrix
      - **D** is a 1 x 1 structure with [nUnits + 2] fields containing individual neuronal calcium data, pupil dynamics, and speed (movement). Each field (variable) contains a 1 x 128 cell where the entire trace is reshaped by number of trials.
      - **TE** is a 1 x 1 structure with 3 fields containing direction (1 x 128 matrix containing labels), nTrials, and dt (time difference between two data points).
      - **t** is a 1 x 49 matrix containing time points for time window [-1 2].
4. "decoder/"
   - Contains csv files that are summarized outputs of batch_decoder_gratings.py or batch_decoder_movies.py
   - Files serve as input files for decoder_eval.py found in "scripts/"
   - Each file ends with "_AUC_results_DF.csv" which has a dataframe containing
      - | AUC   | Pop Size | Group | Animal | Date | Session
        | --- | ---   | ---  | --- | --- | --- | 
        | AUROC scores| number of units used for training | group information (i.e. control, exp) | animal ID | date | session name|

## User instructions
Each script in "code/scripts/" will call for specific input files listed above. 
