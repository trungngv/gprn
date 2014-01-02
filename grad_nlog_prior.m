function ng = grad_nlog_prior(hyp, covfunc, x, y, L)
%GRAD_NLOG_PRIOR g = grad_nlog_prior(hyp, covfunc, x, y, L)
% 
% Compute the derivatives of a the negative log prior wrt the log
% hyperparameters: -dlog p(y|hyp)/dhyp
% where hyp is the log parameters and 
% p(y|hyp) = N(y; 0, K_y) with K_y = cov(x,x|hyp).
%
% INPUT
%   - hyp : log hyper-parameters (of the covariance function)
%   - covfunc : covariance function
%   - x : training inputs
%   - y : outputs
%   - L : chol(K)' where K is covariance matrix (can be optional or empty)
%
% OUTPUT
%   - g : vector containing dlogp(y|hyp)/dhyp
%
% Trung V. Nguyen
% 05/10/12
[n, D] = size(x);
if nargin < 5 || isempty(L)
  K = feval(covfunc, hyp, x);
  [L ~] = jit_chol(K);
  L = L';
end
% alpha = K^{-1}y -> K*alpha = y -> alpha = solve(chol(K),y)
%TODO: implement a version with noise for f
alpha = solve_chol(L',y);
Kinv = L'\(L\eye(n));
Q = Kinv - alpha*alpha'; % Q = K^{-1} - \alpha\alpha'
ng = zeros(size(hyp));
for i = 1:numel(hyp)-1
  ng(i) = sum(sum(Q.*feval(covfunc, hyp, x, [], i)))/2; % using the new gpml-matlab
end
% numerically much more stable
ng(numel(hyp)) = n - y'*alpha;
end

