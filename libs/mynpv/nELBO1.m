function [nF d] = nELBO1(theta_n,nlogpdf,K,D,s2min,theta,n,params)
    
    % Negative evidence lower bound, first order approximation.
    % Note that we exclude constant terms.
    %
    % USAGE: [nF d] = nELBO1(theta_n,nlogpdf,N,D,s2min,theta,n)
    %
    % INPUTS:
    %   theta_n - [1 x D+1] parameter vector, where theta(D+1)=log(sigma_n^2)
    %   nlogpdf - function handle for negative log joint pdf
    %   K - number of components
    %   s2min - minimum bandwidth
    %   theta - [N x D+1] complete set of NVB components
    %   n - which component currently optimizing
    %
    % OUTPUTS:
    %   nF - negative approximate ELBO
    %   d - gradient
    %
    % Sam Gershman, Feb 2012
    
    % entropy lower bound
    theta(n,1:end-1) = theta_n;
    if nargout > 1
        [nF dmu] = lower_bound_MoG(theta,K,D,s2min,n,params);
    else
        nF = lower_bound_MoG(theta,K,D,s2min,n,params);
    end
    
    % calculate negative log joint and its derivatives
    if nargout > 1  % get derivatives
        [nf df1] = nlogpdf(theta_n);
        dmu = dmu + df1'/K;
    else
        nf = nlogpdf(theta_n);
    end
    % because nf is negative log joint, we can use lower-bound hence become
    % minimizing the bound
    nF = nF + nf/K;
    
    if nargout > 1
        d = dmu(:);
    end
    
    