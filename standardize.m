function [X, Xmean, Xstd] = standardize(X,Xmean, Xstd)
%STANDARDIZE X = standardize(X,X_mean,X_std)
%
% Standardize a matrix to be normally distributed. X is the feature matrix
% of size NxD where N is the number of observations and D is the input
% dimension.
% 
% This function can handle the missing data case: all missing data should
% be given the value NaN which will be ignored when performing normalization.
%
% INPUT
%   - X : input feature matrix
%   - Xmean : mean of the training data (when normalize test data)
%   - Xstd : std of the training data (when normalize test data)
%
% OUTPUT
%   - X : the standardized matrix
%
% Trung V. Nguyen (trung.ngvan@gmail.com)
% Last updated: 19/11/12
%
if isempty(Xmean) || isempty(Xstd)
  Xmean = zeros(1,size(X,2));
  Xstd = zeros(1,size(X,2));
  for i=1:size(X,2)
    x = X(:,i);
    Xmean(i) = mean(x(~isnan(x)));
    Xstd(i) = std(x(~isnan(x)));
  end
end

X = X - repmat(Xmean, size(X,1), 1);
X = X ./ repmat(Xstd, size(X,1), 1);

