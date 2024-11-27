function [DM, y, par, c, data] = makeDesignMatrix(TE,D,par,t,nFold)
f = fieldnames(D);
I = find(contains(f,'neuron'));
data = f(I);

%%
P = [];
L =[];
for trial_i = 1:TE.nTrials
    for p_i = 1:length(par)
        switch par(p_i).type
            case 'delayed'
                [P{trial_i,p_i},L(p_i)] = glm_.makeToeplitz_param_delayed(par(p_i),TE,D,t,trial_i);
            case 'continuous'
                [P{trial_i,p_i},L(p_i)] = glm_.makeToeplitz_param_continuous(par(p_i),TE,D,t,trial_i);
            case 'whole'
                [P{trial_i,p_i},L(p_i)] = glm_.makeToeplitz_param_whole(par(p_i),TE,D,t,trial_i);
        end
    end
end

%% concate
uDir = unique(TE.direction);

dm_trial = [];
for trial_i = 1:TE.nTrials
    dm_trial{trial_i} = [];
    for p_i = 1:length(par)
        dm_trial{trial_i} = [dm_trial{trial_i} ; P{trial_i,p_i}];
        par(p_i).L = L(p_i);
%         switch par(p_i).name
%             case 'StimulusOnset'
%                 for i = 1:length(uDir)
%                     if TE.direction(trial_i) == uDir(i)
%                         dm_trial{trial_i} = [dm_trial{trial_i} ; P{trial_i,p_i}];
%                     else
%                         dm_trial{trial_i} = [dm_trial{trial_i} ; P{trial_i,p_i}*0];
%                     end
%                 end
% 
%                 par(p_i).L = repmat(L(p_i),[1,length(uDir)]);
% 
%             case 'Puff'
%                 dm_trial{trial_i} = [dm_trial{trial_i} ; P{trial_i,p_i}];
%                 par(p_i).L = L(p_i);
% 
%             otherwise
%                 dm_trial{trial_i} = [dm_trial{trial_i} ; P{trial_i,p_i}];
%                 par(p_i).L = 1;
%         end
    end
end

%% train test split

tmp = (TE.direction+1);
c = cvpartition(tmp,'KFold',nFold,'Stratify',true);

DM = [];
y = [];
for f_i = 1:nFold
    idx = find(c.training(f_i));
    tmp = [];
    for i = 1:c.TrainSize(f_i)
        tmp = [tmp dm_trial{idx(i)}];
    end
    DM.trn{f_i} = tmp;

    for a_i = 1:length(data)
        tmp = [];
        for i = 1:c.TrainSize(f_i)
            tmp = [tmp D.(data{a_i}){idx(i)}'];
        end
        y.trn{f_i,a_i} = tmp;
    end

    
    idx = find(c.test(f_i));
    tmp = [];
    for i = 1:c.TestSize(f_i)
        tmp = [tmp dm_trial{idx(i)}];
    end
    DM.test{f_i} = tmp;
    
    for a_i = 1:length(data)
        tmp = [];
        for i = 1:c.TestSize(f_i)
            tmp = [tmp D.(data{a_i}){idx(i)}'];
        end
        y.test{f_i,a_i} = tmp;
    end
end



end

