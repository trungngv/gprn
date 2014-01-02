function p = npv_pdf(samples,mu,s2)
    
    % Evaluate the probability of samples under the NPV approximation.
    %
    % USAGE: p = npv_pdf(samples,mu,s2)
    %
    % INPUTS:
    %   samples - [M x D] datapoints
    %   mu - [N x D] component means
    %   s2 - [N x 1] component bandwidths
    %
    % OUTPUTS:
    %   m - [M x 1] probabilities
    %
    % Sam Gershman, June 2012
    
    [M D] = size(samples);
    N = size(mu,1);
    p = zeros(M,N);
    for n = 1:N
        p(:,n) = prod(normpdf(samples,repmat(mu(n,:),M,1),repmat(s2(n),M,D)),2);
    end
    p = mean(p,2);