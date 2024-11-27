# Analysis scripts for 2P imaging data
This folder contains the scripts and functions used for analyzing time-series fluorescence data collected during 2P imaging together with pupil dynamics and movement data.

## Installation instructions
All codes run on Python 3.12.2. or MATLAB 2024a.
For Python, we recommend using a Python IDE (ex. Spyder) to run the main script. 
The following packages need to be installed in the environment prior to running the code
- os
- re
- pandas
- numpy
- matplotlib
- scipy
- sklearn
- seaborn
- statsmodels
- palettable

## Usage instructions 

### Overview 
#### "functions/" contains helper functions used in the main scripts in "scripts/".
#### "scripts/" contains 8 different files.
1. main_analysis.py
   - Contains code for basic visualization of 2P calcium activity 
   - Computes and plots neuronal properties including maximum response magnitude, orientation selectivity index, and reliability index 
2. spks_analysis.py
   - Contains code for extracting deconvolved spikes from Suite2P processing and computing spiking activity-based analyses including firing rates, pairwise correlations, signal and noise correlation
3. batch_decoder_gratings.py
   - Contains code for running SVM-based decoding analysis for **drifting gratings** and creating CSV outputs for each session
4. batch_decoder_movies.py
   - Contains code for running SVM-based decoding analysis for **natural movies** and creating CSV outputs for each session
5. batch_glm_movies.py
   - Contains code for running GLM analysis of single neuron encoding of population activity for **natural movies**
6. decoder_eval.py
   - Contains code for evaluation and visualization of population decoding performance
7. glm_movies_eval.py
   - Contains code for evalulation and visualization of single neuron GLM performance for **natural movies**
8. glm_gratings_eval.py
   - Contains code for evalulation and visualization of single neuron encoding property of **drifting gratings, pupil dynamics, and movement**

#### "glm_gratings/" contains 1 MATLAB script and 1 folder containing helper MATLAB functions.
1. run_script.m
   - Runs the code for single neuron GLM encoding model of drifting gratings, pupil dynamics, and lomotion.
2. "+glm_/"
   - Contains the helper functions used in run_script.m

### Input files
Each script expects a different kind of input files.
1. main_analysis.py and batch_glm_movies requires master mat files that contain MATLAB structures.
   - The sample master files are in "sample-data/master"
   - Each file ends with "_master.mat"
2. spks.analysis.py requires a set of npy files that contain deconvolved spike data, cell index.
   - The sample npy files are in "sample-data/raw"
   - Each session contains a set of npy files including "spks.npy" and "iscell.npy" 
3. batch_decoder_gratings.py and batch_decoder_movies.py require csv files that contain nUnits x dFF data for each session
   - The sample csv files are in "sample-data/csv"
   - Each file ends with "_neuro.csv"
5. decoder_eval.py requires csv files that contain the AUC scores of decoding performance for each session
   - The sample csv files are in "sample-data/decoder"
   - Each file ends with "_auc_scores.csv"
6. glm_movies_eval.py requires csv files that contain GLM results.
   - The sample result files are in "sample-data/glm"
   - Each file ends with "_glm_results_with_weights.csv"
7. glm_gratings_eval.py requires mat files that contain MATLAB structures. 
   - The sample result files are in "sample-data/glm"
   - Each file ends with "_results.mat"

### Running the script
For each script, open a Python IDE (or MATLAB for "glm_gratings/"), nagivate to "scripts/" directory and open each script.  
Each script is commented in each section for instructions. 

