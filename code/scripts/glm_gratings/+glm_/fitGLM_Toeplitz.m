function [rez,par,dataNumber] = fitGLM_Toeplitz(data,time,opts)
%% set params 

par = [];
par = glm_ACA.setParams(par,'stim-cont1-dir0',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont2-dir0',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont3-dir0',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont1-dir45',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont2-dir45',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont3-dir45',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont1-dir90',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont2-dir90',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont3-dir90',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont1-dir135',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont2-dir135',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont3-dir135',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont1-dir180',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont2-dir180',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont3-dir180',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont1-dir225',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont2-dir225',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont3-dir225',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont1-dir270',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont2-dir270',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont3-dir270',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont1-dir315',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont2-dir315',[-1 3],'direction','delayed');
par = glm_ACA.setParams(par,'stim-cont3-dir315',[-1 3],'direction','delayed');
% par = glm_ACA.setParams(par,'StimulusOffset',[0 1],'direction','delayed');
par = glm_ACA.setParams(par,'Puff',[],'puff','whole');
par = glm_ACA.setParams(par,'Speed',[],'speed_cm/s','continuous');
par = glm_ACA.setParams(par,'PupilSize',[],'pupil_zscore','continuous');
par = glm_ACA.setParams(par,'FaceVelocity',[],'face','continuous');
% par = glm_ACA.setParams(par,'Time',[],'time(s)','continuous');
par = glm_ACA.setParams(par,'Pupil_binary',[],'pupil_binary','continuous');
par = glm_ACA.setParams(par,'Speed_binary',[],'speed_binary','continuous');

%% trial split 
d.trialWindow = [-1 3];
d.alignment = 'contrast';
d.speed_threshold = 0.5;
d.pupil_threshold = 20;

[TE, D, t] = glm_ACA.getTrialStruct(data,time,d);
[TE, D] = glm_ACA.deleteTrials(TE,D);

%% make design matrix

[DM, y, par, I, dataNumber] = glm_ACA.makeDesignMatrix(TE,D,par,t,opts.nFold);

%% stimulus window 

opts.stimIdx = find(time>=opts.stimWindow(1)-0.01 & time<=opts.stimWindow(2)+0.001);

%% fit the model

for i = 1:size(y.trn,2)
    tic;
    Da.y.trn = y.trn(:,i);
    Da.y.test = y.test(:,i);
    Da.dm = DM;
    [rez(i)] = glm_ACA.modelFit(Da,I,par,opts);

    disp([num2str(i),' out of ',num2str(size(y.trn,2)),' neurons completed']);
    disp(['Took ',num2str(toc),' seconds']);

end






end