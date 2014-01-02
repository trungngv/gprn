function [f df df2] = logreg(theta,x,c,a0,b0,jac)
    
    % Negative complete log-likelihood for L2-regularized logistic regression.
    %
    % USAGE: [f df df2] = logreg(theta,x,c,[a0],[b0],[jac])
    %
    % INPUTS:
    %   theta - [1 x D+1] vector, where D is the number of covariates. The
    %           first D components correspond to the regression
    %           coefficients (w) and the last component corresponds to
    %           the prior precision (alpha)
    %   x - [N x D] input data, where N is the number of observations
    %   c - [N x 1] output data (class labels), in {-1,1}
    %   a0, b0 (optional) - logistic regression hyperparameters (default: a0=b0=1)
    %   jac (optional) - include Jacobian terms? (default: 1)
    %
    % OUTPUTS:
    %   f - negative log-likelihood
    %   df - first derivatives
    %   df2 - second derivatives (Hessian diagonal)
    %
    % Sam Gershman, June 2012
    
    if nargin < 4; a0 = 1; end
    if nargin < 5; b0 = 1; end
    if nargin < 6; jac = 1; end
    
    if size(theta,2)==1; theta = theta'; end
    w = theta(1:end-1);
    alpha = exp(theta(end));
    D = length(w);
    
    m = x*w';
    p = 1./(1+exp(-c.*m));
    av = (alpha/2)*(w*w');
    loglik = sum(log(p));
    logprior_w = (D/2)*log(alpha/(2*pi)) - av;
    logprior_alpha = (a0-1).*log(alpha) - b0.*alpha;
    f = -loglik - logprior_w - logprior_alpha;
    if jac; f = f - theta(end); end %Jacobian term
    
    % first derivatives
    if nargout > 1
        c_hat = 1./(1+exp(-m));
        dw = ((c+1)/2 - c_hat)'*x - alpha*w;
        dalpha = D/2 - av + (a0-1) - b0.*alpha;
        if jac; dalpha = dalpha + 1; end
        df = -[dw dalpha]';
    end

    % second derivatives
    if nargout > 2
        a = c_hat.*(1-c_hat);
        dw2 = -a'*(x.^2) - alpha;
        dalpha2 = -0.5*sum(alpha.*(w.^2)) - alpha;
        df2 = -[dw2 dalpha2];
    end