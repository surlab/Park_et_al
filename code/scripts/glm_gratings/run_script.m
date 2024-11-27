%% Main script for running glm codes
% Created by jihopark 240608

clc
clear all
close all

addpath(genpath('/Users/jihopark/Documents/GitHub/preprocessing'));
addpath(genpath('/Users/jihopark/Documents/GitHub/neuron-analysis/GLM/yuma_glm'));

%% Define paths 

drive = '/Users/jihopark/Google Drive/My Drive/mrcuts/data/glm_data/';
savepath = '/Users/jihopark/Google Drive/My Drive/mrcuts/analysis/yuma_glm/';

list = dir(drive);

f_list = dir(fullfile(drive,'**','*mrcuts*data.mat'));

%% Test on one session

n = 5;

load(strcat(f_list(n).folder,'/',f_list(n).name))

[rez,adjp,opts] = glm_.runglm_eachSess(D,TE,t);

%% Batch 

for i=1:length(f_list)
    fn = strcat(f_list(i).folder,'/',f_list(i).name);
    disp(['Loading ', f_list(i).name])
    load(fn);

    [rez,adjp,opts] = glm_.runglm_eachSess(D,TE,t);

    filename = strrep(f_list(i).name, 'data', 'results');

    save([savepath,filename],'rez','adjp','opts');
    disp(['Saved ', filename])

end


%% Load the results 

savepath = '/Users/jihopark/Google Drive/My Drive/mrcuts/analysis/yuma_glm/';

f_list = dir(fullfile(savepath,'**','*results.mat'));

% Initialize a cell array to store the combined data
combinedData = cell(length(f_list), 2); % Assuming 2 columns, one for name and one for data

% Loop through each file in f_list
for i = 1:length(f_list)
    % Load the three items from the current file
    filepath = fullfile(f_list(i).folder, f_list(i).name);
    fileData = load(filepath, 'rez', 'adjp', 'opts');
    
    % Regular expression pattern to match everything up to "grat"
    pattern = '(.*grat)';

    % Apply the regular expression to extract the desired part
    tokens = regexp(f_list(i).name, pattern, 'tokens');
    
    filename = tokens{1}{1};

    % Make sure the filename is a valid field name
%     filename = matlab.lang.makeValidName(filename);

    % Combine the data into a cell array
    combinedData{i, 1} = filename;
    combinedData{i, 2} = fileData;
end

% Display the structure to verify the data
disp(combinedData);


%% Load results

f_list = dir(fullfile(savepath,'**','*mrcuts*results.mat'));

n = 5;

load(strcat(f_list(n).folder,'/',f_list(n).name));

%% Let's look at results 

R2_s = [];R2 = [];
p = [];adjp =[];
for i = 1:length(rez)
    R2_s(i) = mean(rez(i).R2_s.all);
    R2(i) = mean(rez(i).R2.all);
    [~,p(i,:)] = ttest(rez(i).R2_s.diff);
    [~,~,~,adjp(i,:)] = glm_.fdr_bh(p(i,:));
end

figure();hold on;
histogram((R2_s),'BinWidth',0.01)
figure;
imagesc(adjp<0.05)

sum(adjp(:,11)<0.05)/length(rez)


%% Plot 

for i=1:length(combinedData)
    session = combinedData{i,1};
    data = combinedData{i,2};

    R2_s = [];
    for i = 1:length(data.rez)
        R2_s(i) = mean(data.rez(i).R2_s.all);
    end
    
    % Create the figure

figure();

% Subplot 1: Histogram
subplot(1, 2, 1); % 1 row, 2 columns, 1st subplot
hold on;
histogram(R2_s, 'BinWidth', 0.01);
title(['Histogram of R2_s for ' session]);
xlabel('R2_s');
ylabel('Frequency');
hold off;

% Subplot 2: Boxplot
subplot(1, 2, 2); % 1 row, 2 columns, 2nd subplot
boxplot(R2_s);
title(['Boxplot of R2_s for ' session]);
xlabel('R2_s');
ylabel('Values');

% Adjust layout
sgtitle(['Analysis of R2_s for ' session]); % Super title for the figure% Create the figure

end


