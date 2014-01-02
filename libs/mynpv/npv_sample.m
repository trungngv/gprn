function samples = npv_sample(mu,s2,M)
    
    % Sample from the NPV approximation.
    %
    % USAGE: samples = npv_sample(mu,s2,M)
    %
    % INPUTS:
    %   mu - [N x D] component means
    %   s2 - [N x 1] component bandwidths
    %   M - number of samples
    %
    % OUTPUTS:
    %   samples - [M x D] random draws
    %
    % Sam Gershman, June 2012
    
    [N D] = size(mu);
    b = randsample(N,M,true);
    samples = normrnd(mu(b,:),repmat(s2(b),1,D));