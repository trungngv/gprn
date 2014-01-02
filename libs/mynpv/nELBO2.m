function [nF nd] = nELBO2(s2,K,D,mu,s2min,h,params)
    
    % Negative evidence lower bound, second-order approximation.
    % Note that we exclude constant terms (terms that do not contain sigma_n)
    % as we use this bound to optimise sigma_n.
    %
    % USAGE: [nF d] = nELBO2(s2,N,D,mu,s2min,h)
    %
    % INPUTS:
    %   s2 - [N x 1] log variance parameters
    %   nlogpdf - function handle for negative log joint PDF
    %   K - number of components
    %   D - number of latent variables
    %   mu - component means
    %   s2min - minimum bandwidth
    %   h - [1 x N] Hessian trace for each component
    %
    % OUTPUTS:
    %   nF - negative approximate ELBO
    %   d - gradient
    %
    % Sam Gershman, Feb 2012

    % entropy lower bound
    if nargout > 1
        [nF , ~, nds2] = lower_bound_MoG([mu s2],K,D,s2min,[],params);
    else
        nF = lower_bound_MoG([mu s2],K,D,s2min,[],params);
    end
    
    s2 = exp(s2) + s2min;
    
    % calculate negative log joint and its derivatives
    for k = 1:K
        if nargout > 1  % get derivatives
            % this is because we are taking derivatives wrt log s2(k)
            nds2(k) = nds2(k) + 0.5*(s2(k)-s2min)*h(k)/K;
        end
        nF = nF + 0.5*s2(k)*h(k)/K; % because h(n) is already negative by the negative logjoint
    end
    
    if nargout > 1; nd = nds2'; end
    %fprintf('nF = %.10f\n', nF);
    
