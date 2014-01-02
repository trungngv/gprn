function e = mysmse(ytrue, ypred)
% function r = mse(ytrue,ypred)
% computes the SMSE: The mean square error normalised by the variance
% of the test targets
% INPUT:
%   - ytrue: real
%   - ypred: prediction from my model
%
% Edwin V. Bonilla

res = ytrue - ypred; % residuals
e   = mean(res.^2,1);
vari = var(ytrue, 1, 1); % Normalizes over N
e = e./vari;

return;



 