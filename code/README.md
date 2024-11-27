# Analysis scripts for 2P imaging data
This folder contains the scripts and functions used for analyzing time-series fluorescence data collected during 2P imaging together with pupil dynamics and movement data.

## Installation instructions
All codes run on Python 3.12.2.
We recommend using a Python IDE to run the main script. 
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
All analysis code can be run through "scripts/main_analysis.py" except for SVM-decoding analysis and GLM analysis. 
In the Python IDE, navigate to scripts/ directory and open main_analysis.py.
For SVM-decoding analysis, navigate to scripts/ directory and open batch_decoder_gratings.py or batch_decoder_movies.py.
For GLM analysis, navigate to scripts/ directory and open ...

### Input files
The code expects to find 2 kinds of data files
1. Master files that contain MATLAB structures.
   - The sample master files are in "sample-data/sample-preprocessed"
2. npy files that contain post-Suite2P data for deconvolved spike data.
   - The sample npy files are in "sample-data/sample-raw"

### Running the script
Execute the script section by section to analyze the sample data. 
Please read the comments in each section for instructions. 
