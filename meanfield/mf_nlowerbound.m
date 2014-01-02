function [nf ndltheta ndlsy mu var model] = mf_nlowerbound(mu,var,ltheta,lsy,Lf,Lw,model,...
  func_arg,optim)
%MF_NLOWERBOUND [nf ndltheta ndlsy mu var model] = mf_nlowerbound(mu,var,ltheta,lsy,Lf,Lw,model,...
%  func_arg,optim)
%   
% Returns the negative expected log joint -E_q[ln p(D,f,w)] and its derivatives
% for mean-field approximation.
% 
% Usage:
% nf = mf_nlowerbound()
% [nf, ndltheta] = mf_nlowerbound()
% [nf, ~, ndlsy] = mf_nlowerbound()
% [nf, ~, ~, mu, var] = mf_nlowerbound()
%
% INPUT
%   - mu : variational mean parameters
%   - var : variational variance parameters 
%   - ltheta : log hyperparameters
%   - lsy : log sigma_y
%   - Lf : the cholesky decomposition (chol(Kf)') of the cov matrix of
%   latent functions
%   - Lw : the cholesky decomposition (chol(Kw)') of the cov matrix of
%   weight functions
%   - model : model configuration
%   - func_arg : specify the argument to the lower bound function
%      -1 : as a function of all variables
%       0 : update mu and var 
%       1 : as a function of hyperparameters
%       2 : as a function of the log \sigma_y
%   - optim : optimisation configurations
%
% OUTPUT
%   - the negative lowerbound and derivatives corresponding to argument
%   passed in by func_arg
%
% Trung V. Nguyen
% 18/10/12
%

%TODO: convert value of Var_F to exp(Var_F) as Var_F must be positive.
P = model.num_outputs; Q = model.num_latents;
X = model.x; Y = model.y;
[N D] = size(X);
nf = 0; ndlsy = []; ndltheta = [];

[mu_f, mu_w] = u_to_fhat_w(mu, model);
[var_f, var_w] = u_to_fhat_w(var, model);
mu_w(model.w_isnan) = 0; var_w(model.w_isnan) = 0;
Mu_f = reshape(mu_f,N,Q); Mu_w = reshape(mu_w, N*P,Q);
Var_f = reshape(var_f,N,Q); Var_w = reshape(var_w, N*P,Q);
sy2 = exp(2*lsy);

if func_arg == 1
  nf = -expected_logprior(Mu_f,Mu_w,Lf,Lw,model);
elseif func_arg == 2
  nf = -expected_log_likelihood(Mu_f,Mu_w,Var_f,Var_w,sy2,model);
elseif func_arg < 0
  nf = -(expected_log_likelihood(Mu_f,Mu_w,Var_f,Var_w,sy2,model)...
         + expected_logprior(Mu_f,Mu_w,Lf,Lw,model) + entropy(model));
end
  
if func_arg == 0 % optimise mu and var
  Mu_wcell = mat2cell(Mu_w, N*ones(1,P), ones(1,Q));
  Var_wcell = mat2cell(Var_w, N*ones(1,P), ones(1,Q));
  
  % optimise Var(fj) and Mu(fj), keeping everything else fixed
  Kf = Lf*Lf'; Kw = Lw*Lw';
  for j=1:Q
    Diagfj = zeros(N,1); tmp = zeros(N,1);
    for i=1:P
      Diagfj = Diagfj + Mu_wcell{i,j}.^2 + Var_wcell{i,j};
      Sum_nj = zeros(N,1); % \sum_{k # j} E[w_ik] o E[f_k] (sum not j)
      for k=1:Q
        if k ~=j
          Sum_nj = Sum_nj + Mu_wcell{i,k}.*Mu_f(:,k);
        end
      end
      tmp = tmp + (Y(:,i) - Sum_nj).*Mu_wcell{i,j};
      tmp(model.Y_isnan(:,i)) = 0;
    end
    model.CovF{j} = Kf - Kf*((diag(sy2./Diagfj) + Kf)\Kf);
    Var_f(:,j) = diag(model.CovF{j});
    Mu_f(:,j) = model.CovF{j}*tmp/sy2;
  end
  
  % optimise Var(wij) and Mu(wij), keeping every else fixed
  for i=1:P
    valid_ind = ~model.Y_isnan(:,i);
    Kwi = Kw(valid_ind,valid_ind);
    for j=1:Q
      Mu_fj = Mu_f(:,j); Var_fj = Var_f(:,j);
      Diagij = Mu_fj(valid_ind).^2 + Var_fj(valid_ind);
      model.CovW{i,j} = Kwi - Kwi*((diag(sy2./Diagij) + Kwi)\Kwi);
      Var_wcell{i,j}(valid_ind) = diag(model.CovW{i,j});
      Sum_nj = zeros(N,1); % \sum_{k#j} E[Mu_fk] o E[Mu_fj] o E[Mu_wik]
      for k=1:Q
        if k~=j
          Sum_nj = Sum_nj + Mu_f(:,k).*Mu_wcell{i,k};
        end
      end
      tmp = (Y(:,i) - Sum_nj).*Mu_f(:,j);
      Mu_wcell{i,j}(valid_ind) = model.CovW{i,j}*tmp(valid_ind)/sy2;
    end
  end
  
  % re-group into mu and var
  Mu_w = cell2mat(Mu_wcell); Var_w = cell2mat(Var_wcell);
  mu = [Mu_f(:); Mu_w(:)]; var = [Var_f(:); Var_w(:)];
end      

ind_Wxn = N*(0:P-1)';

if nargout == 2             % -deE/d log theta
ltheta_f = ltheta(1:D+1);
ltheta_w = ltheta(D+2:end);
Kfinv = Lf'\(Lf\eye(N));
Kwinv_full = Lw'\(Lw\eye(N));
Kw = Lw*Lw';

dtheta_f = zeros(size(ltheta_f)); dtheta_w = zeros(size(ltheta_w));
Mu_f = reshape(mu_f,N,Q); Mu_w=reshape(mu_w,N*P,Q);
W = mat2cell(Mu_w,N*ones(1,P),ones(1,Q));
for j=1:Q
  dtheta_f = dtheta_f + grad_nlog_prior(ltheta_f,model.covfunc_f,X,Mu_f(:,j));
  for i=1:P
    dtheta_w = dtheta_w + grad_nlog_prior(ltheta_w,model.covfunc_w,...
          X(~model.Y_isnan(:,i),:),W{i,j}(~model.Y_isnan(:,i)));
  end
end

dtheta2_f = zeros(size(ltheta_f));
Sum_CovF = zeros(N,N);
for j=1:Q
  Sum_CovF = Sum_CovF + model.CovF{j};
end
for idx=1:D
  dK = feval(model.covfunc_f, ltheta_f, X, [], idx);
  dtheta2_f(idx) = trace(Kfinv*dK*Kfinv*Sum_CovF);
end
if optim.optimise_sf
  dtheta2_f(D+1) = 2*trace(Kfinv*Sum_CovF);
end

dtheta2_w = zeros(size(ltheta_w));
for i=1:P
  valid_ind = ~model.Y_isnan(:,i);
  if sum(valid_ind) ~= N
    tmp = jit_chol(Kw(valid_ind,valid_ind));
    Kwinv = tmp\(tmp'\eye(sum(valid_ind)));
  else
    Kwinv = Kwinv_full;
  end
  Sum_CovW = zeros(size(Kwinv));
  for j=1:Q
    Sum_CovW = Sum_CovW + model.CovW{i,j};
  end
  for idx=1:D
    dK = feval(model.covfunc_w, ltheta_w, X(valid_ind,:), [], idx);
    dtheta2_w(idx) = dtheta2_w(idx) + trace(Kwinv*dK*Kwinv*Sum_CovW);
  end
  if optim.optimise_sw
    dtheta2_w(D+1) = dtheta2_w(D+1) + 2*trace(Kwinv*Sum_CovW);
  end
end

ndltheta = [dtheta_f; dtheta_w] - 0.5*[dtheta2_f; dtheta2_w];
if ~optim.optimise_sf, ndltheta(D+1) = 0; end
if ~optim.optimise_sw,  ndltheta(end) = 0; end
end

if nargout == 3
  % -dE/d log sigma_y
  % 0.5\sum_n P dlsy2/dlsy + likelihood part
  ndlsy = (N*P-sum(sum(model.Y_isnan))) + ndkdy(Mu_f,Mu_w,sy2,model)...
    - expected_llh_variational(Mu_f,Mu_w,Var_f,Var_w,model)/sy2;
end

end

% compute the expected log likelihood E[p(y|W, Fhat)]
function logl = expected_log_likelihood(Mu_f,Mu_w,Var_f,Var_w,sy2,model)
Y = model.y; N = size(Y, 1);
P = model.num_outputs;

%vectorised version
Wxn_ind = N*(0:P-1)';
W_ind = bsxfun(@plus,Wxn_ind,1:N);
Wblk = Mu_w(W_ind(:),:);
tt = repmat(1:N,P,1);
Fblk = Mu_f(tt(:),:);
Ymean = reshape(sum(Wblk.*Fblk,2),P,N)';
Ydiff = (Y-Ymean).^2;
Ydiff(model.Y_isnan) = 0; % missing data part
logl = -0.5*sum(sum(Ydiff))/sy2;
logl = logl - 0.5*(N*P-sum(sum(model.Y_isnan)))*log(sy2); % due to missing data
logl = logl - 0.5*expected_llh_variational(Mu_f,Mu_w,Var_f,Var_w,model)/sy2;
end

% compute the term in expected likelihood that only contains variational
% parameters
function value = expected_llh_variational(Mu_f,Mu_w,Var_f,Var_w,model)
P = model.num_outputs;
Q = model.num_latents;
N = size(model.x,1);
Mu_wcell = mat2cell(Mu_w, N*ones(1,P), ones(1,Q));
Var_wcell = mat2cell(Var_w, N*ones(1,P), ones(1,Q));
value = 0;
for i=1:P
  for j=1:Q
    value = value + sum(Var_f(:,j).*Mu_wcell{i,j}.*Mu_wcell{i,j})...
      + sum(Var_wcell{i,j}.*Mu_f(:,j).*Mu_f(:,j)) + sum(Var_f(:,j).*Var_wcell{i,j});
  end
end
end

% compute the expected log prior E[p(f,w)]
function logprior = expected_logprior(Mu_f,Mu_w,Lf,Lw,model)
%TODO: the term tr(Kf^{-1}\Sigma_fj) and tr(Kw^{-1}\Sigma_wij) can be
%computed (perhaps more numerically stable) by using the stationary
%equation for \Sigma_fj and \Sigma_wij. But this takes one more matrix
%inversion operation.
N = size(model.x,1); P = model.num_outputs; Q = model.num_latents;
Mu_wcell = mat2cell(Mu_w, N*ones(1,P), ones(1,Q));
Kfinv = Lf'\(Lf\eye(N));
logprior = -Q*sum(log(diag(Lf)));
Kw = Lw*Lw';
Sum_CovF = zeros(size(model.CovF{1}));
for j=1:Q
  %log_p = log_p - 0.5*F(:,j)'*Kf_inv*F(:,j);
  alpha = Lf\Mu_f(:,j);
  logprior = logprior - 0.5*(alpha')*alpha;
  Sum_CovF = Sum_CovF + model.CovF{j};
end
logprior = logprior - 0.5*trace(Kfinv*Sum_CovF);

for i=1:P
  valid_ind = ~model.Y_isnan(:,i);
  if sum(valid_ind) ~= N,     Lwi = jit_chol(Kw(valid_ind, valid_ind))';
  else     Lwi = Lw;  end
  Kwi_inv = Lwi'\(Lwi\eye(sum(valid_ind)));
  for j=1:Q
    %log_p = log_p - 0.5*W{i,j}'*Kw_inv*W{i,j};
    alpha = Lwi\Mu_wcell{i,j}(valid_ind);
    logprior = logprior - 0.5*(alpha')*alpha - 0.5*trace(Kwi_inv*model.CovW{i,j});
  end
  logprior = logprior - Q*sum(log(diag(Lwi)));
end
end

% - d log p(y|mu_k) / d log sigma_y
function dy = ndkdy(Mu_f,Mu_w,sy2,model)
Y = model.y;
[N,P] = size(Y);
Wxn_ind = N*(0:P-1)';
W_ind = bsxfun(@plus,Wxn_ind,1:N);
Wblk = Mu_w(W_ind(:),:);
tt = repmat(1:N,P,1);
Fblk = Mu_f(tt(:),:);
Ymean = reshape(sum(Wblk.*Fblk,2),P,N)';
Ydiff = (Y-Ymean).^2;
Ydiff(model.Y_isnan) = 0; % missing data part
dy = -sum(sum(Ydiff))/sy2;
end

