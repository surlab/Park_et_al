%% Main script for running glm codes
% Created by jihopark 240608

clc
clear all
close all


% Add path to helper functions
addpath(genpath('/Users/jihopark/Documents/GitHub/Park_et_al_2024/code/scripts/glm_gratings'));

%% Define paths 

datapath = '/Users/jihopark/Documents/GitHub/Park_et_al_2024/sample-data/glm/';
savepath = '/Users/jihopark/Documents/GitHub/Park_et_al_2024/results/sample-output/';

list = dir(datapath);

% Identify all the input files formatted for glm_gratings 
f_list = dir(fullfile(datapath,'**','*mrcuts*data.mat'));

%% Batch run glm analysis on each session and save output as a MATLAB file 

for i=1:length(f_list)
    fn = strcat(f_list(i).folder,'/',f_list(i).name);
    disp(['Loading ', f_list(i).name])
    load(fn);

    [rez,adjp,opts] = glm_.runglm_eachSess(D,TE,t);

    filename = strrep(f_list(i).name, 'data', 'results');

    save([savepath,filename],'rez','adjp','opts');
    disp(['Saved ', filename])

end
