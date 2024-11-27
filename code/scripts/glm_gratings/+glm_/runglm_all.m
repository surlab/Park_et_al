function runglm_all(path,path_save,name)
%%

opts = struct;
opts.nFold = 10;
opts.useGPU = false;
opts.method = 'lasso'; %lasso, normal
opts.searchLambda = logspace(-6,-0.5,11);
opts.dist = 'normal';
opts.stimWindow = [0 2];
opts.lambda = logspace(-6,-0.5,11);


%%

% you can change here to fit your data structure 
d = dir(path);
dataName = {d.name};
dataName(strcmp(dataName,'.')) = [];
dataName(strcmp(dataName,'..')) = [];
dataName(contains(dataName,'._')) = [];
%


rez = [];
dataNumber = [];
for sess_i = 1:length(dataName)
    selpath = fullfile(path,dataName{sess_i});
    [data, time, info] = prep.getData(selpath);
    [data info] = prep.normalizeSignal(data,info);


    [tmp,par,dataNumber{sess_i}] = glm_.fitGLM_Toeplitz(data,time,opts);

    if sess_i == 1
        rez = tmp;
    else
        l = length(rez);
        rez(l+1:length(tmp)+l) = tmp;
    end

    disp([num2str(sess_i),' out of ',num2str(length(dataName)),' sessions completed']);
    save(fullfile(path_save,name),'rez','sess_i','dataName','dataNumber','-v7.3');
end




end