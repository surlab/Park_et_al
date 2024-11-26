# Analysis scripts for 2P imaging data
This folder contains the scripts and functions used for analyzing time-series fluorescence data collected during 2P imaging together with pupil dynamics and movement data.

## Installation instructions
All codes run on Python 3.12.2.
The following packages need to be installed prior to running the code
- os
- re
- pandas
- numpy
- matplotlib
- scipy
- sklearn
- seaborn
- statsmodels
 

## Usage instructions 

### Overview 
All code can be run through "scripts/main_analysis.py" that contains sections of code. 

### Input files
The code expects to find 2 kinds of data files
1. Master files that contain MATLAB structures.
   - The sample master files are in "sample-data/sample-preprocessed"
2. npy files that contain post-Suite2P data for deconvolved spike data.
   - The sample npy files are in "sample-data/sample-raw"
  
 
