function [F mu s2] = npv_run(nlogpdf,theta,nIter,tol)
    
    % Run nonparametric variational Bayesian inference.
    % This function uses the "incremental" training protocol, wherein components are optimized
    % one at a time.
    %
    % USAGE: [F mu s2] = npv_run(nlogpdf,theta,[nIter])
    %
    % INPUTS:
    %   nlogpdf - function handle for negative log joint pdf. Should return
    %             the gradient as a second-output
    %   theta - [N x D+1] initial parameter settings, where N is the number
    %           of components, D is the number of latent variables in the model,
    %           and the last column contains the log bandwidths (variances)
    %   nIter (optional) - maximum number of iterations (default: 10)
    %   tol (optional) - change in the evidence lower bound (ELBO) for
    %   convergence (default: 0.0001)
    %
    % OUTPUTS:
    %   F - [nIter x 1] approximate ELBO value at each iteration
    %   mu - [N x D] component means
    %   s2 - [N x 1] component bandwidths
    %
    % Sam Gershman, Feb 2012
    
    if nargin < 3 || isempty(nIter); nIter = 10; end
    if nargin < 4 || isempty(tol); tol = 0.0001; end
    s2min = 0.0000001;      % enforce minimium bandwidth to avoid numerical problems
    [N D] = size(theta); D = D - 1;
    opts = struct('Display','off','Method','lbfgs','MaxIter',5000,'MaxFunEvals',5000,'DerivativeCheck','on');  % optimization options
    
    for iter = 1:nIter
        disp(['iteration: ',num2str(iter)]);
        
        % first-order approximation (L1): optimize mu, one component at a time
        for n = 1:N
            func = @(x) nELBO1(x,@(x) nlogpdf(x),N,D,s2min,theta,n,[]);
            theta(n,1:end-1) = minFunc(func,theta(n,1:D)',opts);
        end
        
        % second-order approximation (L2): optimize s2
        mu = theta(:,1:D);
        h = zeros(N,1);
        for n = 1:N; h(n) = hesstrace(nlogpdf,mu(n,:)); end % compute Hessian trace using finite differencing
        func = @(theta) nELBO2(theta,N,D,mu,s2min,h,[]);
        theta = minFunc(func, theta(:,D+1),opts);
        theta = [mu reshape(theta,N,1)];
        
        F(iter,1) = ELBO(theta,nlogpdf,h,s2min,[]);              % calculate the approximate ELBO (L2)
        if iter > 1 && abs(F(iter)-F(iter-1))<tol; break; end % check for convergence
    end
    
    mu = theta(:,1:D);
    s2 = exp(theta(:,end)) + s2min;