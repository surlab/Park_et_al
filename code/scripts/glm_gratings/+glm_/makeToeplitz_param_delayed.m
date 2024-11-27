function [DM, L] = makeToeplitz_param_delayed(par,TE,D,t,trial_i)

DM = [];

uDir = unique(TE.direction);

switch par.name
    case 'stim-dir0'
        [DM,L] = getDM_stim(t,par,TE);
        if ~(TE.direction(trial_i) == 0)
            DM = DM*0;
        end

    case 'stim-dir45'
        [DM,L] = getDM_stim(t,par,TE);
        if ~(TE.direction(trial_i) == 45)
            DM = DM*0;
        end

    case 'stim-dir90'
        [DM,L] = getDM_stim(t,par,TE);
        if ~(TE.direction(trial_i) == 90)
            DM = DM*0;
        end

    case 'stim-dir135'
        [DM,L] = getDM_stim(t,par,TE);
        if ~(TE.direction(trial_i) == 135)
            DM = DM*0;
        end

    case 'stim-dir180'
        [DM,L] = getDM_stim(t,par,TE);
        if ~(TE.direction(trial_i) == 180)
            DM = DM*0;
        end

    case 'stim-dir225'
        [DM,L] = getDM_stim(t,par,TE);
        if ~(TE.direction(trial_i) == 225)
            DM = DM*0;
        end

    case 'stim-dir270'
        [DM,L] = getDM_stim(t,par,TE);
        if ~(TE.direction(trial_i) == 270)
            DM = DM*0;
        end

    case 'stim-dir315'
        [DM,L] = getDM_stim(t,par,TE);
        if ~(TE.direction(trial_i) == 315)
            DM = DM*0;
        end

end

% switch par.name
%     case 'StimulusOnset'
%         [~, idx] = min(abs(t - 0));
%         frame = round(par.window/TE.dt + idx);
%         frames = frame(1):frame(end);
%
%         DM = zeros(length(frames),length(t));
%
%         for i = 1:length(frames)
%             DM(i,frames(i)) = 1;
%         end
%
%         if size(DM,2) > length(t)
%             DM(:,length(t)+1 : end) = [];
%         end
%
%         L = length(frames);
%         DM = DM * TE.contrast(trial_i);
%
%     case 'StimulusOffset'
%         [~, idx] = min(abs(t - 1));
%         frame = round(par.window/TE.dt + idx);
%         frames = frame(1):frame(end);
%
%         DM = zeros(length(frames),length(t));
%
%         for i = 1:length(frames)
%             DM(i,frames(i)) = 1;
%         end
%
%         if size(DM,2) > length(t)
%             DM(:,length(t)+1 : end) = [];
%         end
%
%         L = length(frames);
%         DM = DM * TE.contrast(trial_i);
% 
%     case 'Puff'
%         [~, idx] = min(abs(t - (- 2)));
%         frame = round(par.window/TE.dt + idx);
%         frames = frame(1):frame(end);
% 
%         DM = zeros(length(frames),length(t));
% 
%         if TE.puff(trial_i) == 1
%             for i = 1:length(frames)
%                 DM(i,frames(i)) = 1;
%             end
%         end
% 
%         if size(DM,2) > length(t)
%             DM(:,length(t)+1 : end) = [];
%         end
% 
%         L = length(frames);
% end





end


function [DM,L] = getDM_stim(t,par,TE)
[~, idx] = min(abs(t - 0));
frame = round(par.window/TE.dt + idx);
frames = frame(1):frame(end);

DM = zeros(length(frames),length(t));

for i = 1:length(frames)
    DM(i,frames(i)) = 1;
end

if size(DM,2) > length(t)
    DM(:,length(t)+1 : end) = [];
end

L = length(frames);
end