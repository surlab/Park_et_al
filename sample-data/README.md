# Sample-data 
We provide sample data (one session from each animal) to run analysis scripts used in the manuscript. 

## Overview
This folder contains 4 folders with differentially formatted sample data for each analysis script.
Make sure to specify the datapath in each script in "code/scripts/" to each of the folders. 

1. master
   - Contains the master MATLAB files that end with "_master.mat"
2. raw
   - Contains npy files in each session folder
3. glm
   - Contains data MATLAB files that end with "_data.mat" which serve as input files for run_script.m found in "scripts/glm_gratings/"
4. decoder
   - Contaings csv files that are outputs of batch_decoder_gratings.py or batch_decoder_movies.py
   - Files serve as input files for decoder_eval.py found in "scripts/"
