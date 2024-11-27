function [bestL] = getBestLambda(X,Y,opts)

cv = cvpartition(length(Y),"KFold",3);
R2 =[];
for i = 1:length(opts.searchLambda)
    for cvi = 1:length(cv.TrainSize)
        [B,FitInfo] = lassoglm(X(cv.training(cvi),:),Y(cv.training(cvi)),opts.dist,'Lambda',opts.searchLambda(i),...
            'Alpha',1);
        B0 = FitInfo.Intercept;
        coef = [B0; B];
        yhat = glmval(coef,X(cv.test(cvi),:),'identity');
        R2(cvi,i) = corr(Y(cv.test(cvi)),yhat)^2;
    end
end

[~,idx] = max(mean(R2,1));
bestL = opts.searchLambda(idx(1));

end