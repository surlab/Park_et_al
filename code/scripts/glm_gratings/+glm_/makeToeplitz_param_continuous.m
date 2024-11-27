function [DM, L] = makeToeplitz_param_continuous(par,TE,D,t,trial_i)

% N = name2params(par.align);
DM = D.(par.align){trial_i}(1:length(t))';
L = 1;

end

% function [N] = name2params(name)
% switch name
%     case 'speed_cm/s'
%         N = 'speed';
%     case 'pupil_zscore'
%         N = 'pupil_zscore';
%     case 'face'
%         N = 'face';
%     case 'time(s)'
%         N = 'time';
%     case 'pupil_binary'
%         N = 'pupil_binary';
%     case 'speed_binary'
%         N = 'speed_binary';
% end
% end
%     