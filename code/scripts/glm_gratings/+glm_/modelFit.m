function [rez] = modelFit(D,TD,par,opts)
% fit the model to empirical unit activity data with k-fold cross
% validation using linear link function (identity function).
% input
%     D     :    design matrix [struct]
%     TD    :    Trial dataset [struct]
%     par   :    parameter information [struct]
% output
%     rez   :    rezults [struct]

% whether to draw the results to figure (if not, set fig = 0)
fig = 0;

nFold = length(TD.TrainSize);

pos = [];
for param_i = 1:length(par)
    pos = [pos par(param_i).L];
end
dmPos = [0 cumsum(pos)];

%% parameters setting

targetDir = {'dir0','dir45','dir90','dir135','dir180','dir225','dir270','dir315'}';
pname = {par.name}';
idx = find(~contains(pname,'stim'));
params = [targetDir ; pname(idx) ; 'Stim'];

%% core calculation

R2.all = zeros(nFold,1);
R2.partial = zeros(nFold,length(params)-1);
R2.diff = zeros(nFold,length(params)-1);
R2_s.diff = zeros(nFold,length(params)-1);
R2_s.partial = zeros(nFold,length(params)-1);
beta = [];
rez = [];

for fold_i = 1:nFold
    TL = TD.TestSize(fold_i);

    switch opts.method
        case 'normal'
            b = glmfit(D.dm.trn{fold_i}',D.y.trn{fold_i}',opts.dist,'link',opts.link);
            yhat = glmval(b,D.dm.test{fold_i}',opts.link);
            beta(:,fold_i) = b;
        case 'lasso'
            if isnan(opts.lambda)
                [bestL] = glm_.getBestLambda(D.dm.trn{fold_i}',D.y.trn{fold_i}',opts);
            else
                bestL = opts.lambda;
            end
            [B,FitInfo] =  lassoglm(D.dm.trn{fold_i}',D.y.trn{fold_i}',opts.dist,'Lambda',bestL);
            B0 = FitInfo.Intercept;
            coef = [B0; B];
            yhat = glmval(coef,D.dm.test{fold_i}',opts.link);
            beta(:,fold_i) = coef;
    end

    
    y_pred = reshape(yhat',[length(yhat)/TL,TL]);
    y_emp = reshape(D.y.test{fold_i},[length(yhat)/TL,TL]);
    
    rez.y_pred{fold_i} = y_pred';
    rez.y_emp{fold_i} = y_emp';
    
%     R2.all(fold_i) = corr(mean(y_pred,2),mean(y_emp,2))^2;
    R2.all(fold_i) = corr(y_pred(:),y_emp(:))^2;

    y_pred_stim = y_pred(opts.stimIdx,:);
    y_emp_stim = y_emp(opts.stimIdx,:);

    R2_s.all(fold_i) = corr(y_pred_stim(:),y_emp_stim(:))^2;
    
    for param_i = 1:length(params)
        dm_partial = D.dm.test{fold_i}';
        if ~strcmp(params{param_i},'Stim')
            idx = find(contains(pname,params{param_i}));
            pos = [];
            for i = 1:length(idx)
                tmp = dmPos(idx(i))+1 : dmPos(idx(i)+1);
                pos = [pos tmp];
            end
            dm_partial(:,pos) = 0;
        else
            idx = find(contains(pname,'stim'));
            pos = [];
            for i = 1:length(idx)
                tmp = dmPos(idx(i))+1 : dmPos(idx(i)+1);
                pos = [pos tmp];
            end
            dm_partial(:,pos) = 0;
        end
        
        switch opts.method
            case 'normal'
                yhat = glmval(b,dm_partial,'identity');
            case 'lasso'
                yhat = glmval(coef,dm_partial,'identity');
        end
        
        y_pred = reshape(yhat',[length(yhat)/TL,TL]);
        y_emp = reshape(D.y.test{fold_i},[length(yhat)/TL,TL]);
        
        R2.partial(fold_i,param_i) = corr(y_pred(:),y_emp(:))^2;
        y_pred_stim = y_pred(opts.stimIdx,:);
        y_emp_stim = y_emp(opts.stimIdx,:);
        R2_s.partial(fold_i,param_i) = corr(y_pred_stim(:),y_emp_stim(:))^2;
        
        R2.diff(fold_i,param_i) = R2.all(fold_i) - R2.partial(fold_i,param_i);
        R2_s.diff(fold_i,param_i) = R2_s.all(fold_i) - R2_s.partial(fold_i,param_i);
    end
end

%% draw the results in figure

paramName =[];
paramName = ['All' ; params];
% for param_i = 1:length(par)
%     switch par(param_i).name
%         case 'StimulusOnset'
%             a = 0:360/length(par(param_i).L):315;
%             for i = 1:length(par(param_i).L)
%                 paramName{end+1} = ['dir',num2str(a(i))];
%             end
%         otherwise
%             paramName{end+1} = par(param_i).name;
%     end
% end

% y = [R2.all R2.partial];
% beta_mean = mean(beta,2);
% beta_mean = beta(:,fold_i);
% yhat = glmval(b,D.dm.all',F);
% 
% if fig
%     figure();
%     subFunc.drawPrediction(D.y.all,yhat);
%     R2.all
%     
%     figure();
%     tiledlayout(1,nFold);
%     for fold_i = 1:nFold
%         nexttile;
% %         subFunc.drawPrediction(mean(rez.y_emp{fold_i}),mean(rez.y_pred{fold_i}));
%         subFunc.drawPrediction(rez.y_emp{fold_i}(:),rez.y_pred{fold_i}(:));
%     end
% end

rez = struct;
rez.R2 = R2;
rez.R2_s = R2_s;
rez.param = paramName;
rez.beta = beta;
rez.predictY = yhat;
rez.enpiricalY = y_emp';
rez.paramPos = dmPos;

end