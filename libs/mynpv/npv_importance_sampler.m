function [samples W] = npv_importance_sampler(mu,s2,M,nlogpdf)
    
    % Importance sampling using the NPV appoximation as a proposal distribution.
    %
    % USAGE: [samples W] = npv_importance_sampler(mu,s2,M,nlogpdf)
    %
    % INPUTS:
    %   mu - [N x D] component means
    %   s2 - [N x 1] component bandwidths
    %   M - number of samples
    %   nlogpdf - function handle for negative log joint pdf
    %
    % OUTPUTS:
    %   samples - [M x D] random draws from the NPV approximation
    %   W - [M x 1] importance weights
    %
    % Sam Gershman, June 2012
    
    samples = npv_sample(mu,s2,M);
    q = npv_pdf(samples,mu,s2);
    logp = zeros(M,1);
    for i=1:M; logp(i,1) = nlogpdf(samples(i,:)); end
    W = exp(-logp-log(q));
    W = W./sum(W);