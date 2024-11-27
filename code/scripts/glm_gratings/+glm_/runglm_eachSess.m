function [rez,adjp,opts] = runglm_eachSess(D,TE,t)
%% GLM params 
opts = struct;
opts.nFold = 8;
opts.useGPU = false;
opts.method = 'lasso'; %lasso, normal
opts.searchLambda = logspace(-6,-0.5,11);
opts.dist = 'normal';
opts.stimWindow = [0 2];
opts.lambda = nan;
opts.link = 'identity';

% opts.link.Link = @(mu) log(mu);          
% opts.link.Inverse  = @(eta) exp(eta);     
% opts.link.Derivative  = @(mu) 1 ./ mu;        

%% task params assignment

par = [];
par = glm_.setParams(par,'stim-dir0',[0 2],[],'delayed');
par = glm_.setParams(par,'stim-dir45',[0 2],[],'delayed');
par = glm_.setParams(par,'stim-dir90',[0 2],[],'delayed');
par = glm_.setParams(par,'stim-dir135',[0 2],[],'delayed');
par = glm_.setParams(par,'stim-dir180',[0 2],[],'delayed');
par = glm_.setParams(par,'stim-dir225',[0 2],[],'delayed');
par = glm_.setParams(par,'stim-dir270',[0 2],[],'delayed');
par = glm_.setParams(par,'stim-dir315',[0 2],[],'delayed');

par = glm_.setParams(par,'Wheel',[],'speed','continuous');
par = glm_.setParams(par,'PupilSize',[],'pupil_zscore','continuous');

%% create design matrix

[DM, y, par, I, dataNumber] = glm_.makeDesignMatrix(TE,D,par,t,opts.nFold);

%% add stimulus window 

opts.stimIdx = find(t>=opts.stimWindow(1)-0.01 & t<=opts.stimWindow(2)+0.001);

%% fit the model

for i = 1:size(y.trn,2)
    tic;
    Da.y.trn = y.trn(:,i);
    Da.y.test = y.test(:,i);
    Da.dm = DM;
    [rez(i)] = glm_.modelFit(Da,I,par,opts);

    disp([num2str(i),' out of ',num2str(size(y.trn,2)),' neurons completed']);
    disp(['Took ',num2str(toc),' seconds']);

end

%% 

R2_s = [];R2 = [];
p = [];adjp =[];
for i = 1:length(rez)
    R2_s(i) = mean(rez(i).R2_s.all);
    R2(i) = mean(rez(i).R2.all);
    [~,p(i,:)] = ttest(rez(i).R2_s.diff);
    [~,~,~,adjp(i,:)] = glm_.fdr_bh(p(i,:));
end

% figure();hold on;
% histogram((R2_s),'BinWidth',0.01)
% histogram((R2),'BinWidth',0.01)
% 
%  figure;
%  imagesc(adjp<0.05)
% 
% sum(adjp(:,11)<0.05)/length(rez)

end