function [nf ndu ndlsk ndltheta ndlsy] = exact_nlogjoint(u,lsk,ltheta,lsy,Lf,Lw,...
  model,func_arg,optim)
%EXACT_NLOGJOINT [nf ndu ndlsk ndltheta ndlsy] = exact_nlogjoint(u,lsk,ltheta,lsy,Lf,Lw,model,func_arg,optim)
%   
% Returns the negative expected log joint -E_q[ln p(D,f,w)] and its derivatives.
% 
% Usage:
% nf = exact_nlogjoint()
% [nf ndmu] = exact_nlogjoint()
% [nf, ~, ndlsk] = exact_nlogjoint()
% [nf, ~, ~, ndltheta] = exact_nlogjoint()
% [nf, ~, ~, ~, ~, ndlsy] = exact_nlogjoint()
%
% INPUT
%   u : mean of one component
%   lsk : log sigma_ k for a particular component
%   ltheta : log hyperparameters
%   lsy : log sigma_y
%   Lf : the cholesky decomposition (chol(Kf)') of the cov matrix of
%     latent functions
%   Lw : the cholesky decomposition (chol(Kw)') of the cov matrix of
%     weight functiosn
%   func_arg : specify the argument to this function
%     1 for u,
%     2 for lsk
%     3 for ltheta
%     4 for lsy
%
% OUTPUT
%   - 
%
% Trung V. Nguyen
% 10/10/12
P = model.num_outputs; Q = model.num_latents;
X = model.x; Y = model.y;
[N D ] = size(X);
ndu = []; ndlsk = []; ndlsy = []; ndltheta = [];

[fhat,w] = u_to_fhat_w(u, model);
w(model.w_isnan) = 0;
Fhat = reshape(fhat,N,Q); Wmat = reshape(w, N*P,Q);
Wcell = mat2cell(Wmat,N*ones(1,P),ones(1,Q));
sy2 = exp(2*lsy); sk2 = exp(2*lsk);

if func_arg == 3
  nf = -expected_logprior(fhat,w,Lf,Lw,model,sk2);
elseif func_arg == 4
  nf = -expected_log_likelihood(Wmat,Fhat,sy2,sk2,model);
else
  nf = -(expected_log_likelihood(Wmat,Fhat,sy2,sk2,model)...
          + expected_logprior(fhat,w,Lf,Lw,model,sk2));
end      

ind_Wxn = N*(0:P-1)';
Kw = Lw*Lw';

if nargout == 2 % -dE/dMu
  % likelihood part
  dFf = zeros(N,Q); dFw = zeros(N*P,Q);
  for n=1:N
    Wxn = Wmat(ind_Wxn+n,:);
    yd = Y(n,:)' - Wxn*Fhat(n,:)'; % y(xn) - W(xn)fhat(xn)
    yd(isnan(yd)) = 0; % for missing data
    dFf(n,:) = yd'*Wxn; % dp(y|u)/d fhat(xn)
    dFw(ind_Wxn+n,:) = yd*Fhat(n,:); % dp(y|u)/dW(xn)
  end
  ndu = (-[dFf(:); dFw(:)] + sk2*[P*fhat; w])/sy2; % dp(y|u)/du
  
  % prior part
  dFu2 = zeros(size(ndu)); last_ind = 1;
  for j=1:Q
    alpha = solve_chol(Lf', Fhat(:,j)); % Kf^{-1} * Fhat(:,j)
    dFu2(last_ind : last_ind + numel(alpha) - 1) = alpha;
    last_ind = last_ind + numel(alpha);
  end

  for j=1:Q
    for i=1:P
      if sum(model.Wcell_isnan{i,j}) > 0 % missing data
        valid_ind = ~model.Wcell_isnan{i,j};
        alpha = zeros(N,1);
        alpha(valid_ind) = solve_chol(jit_chol(Kw(valid_ind,valid_ind)),Wcell{i,j}(valid_ind));
      else
        alpha = solve_chol(Lw', Wcell{i,j});
      end  
      %alpha = Kwinv*Wcell{i,j};
      dFu2(last_ind : last_ind + numel(alpha) - 1) = alpha;
      last_ind = last_ind + numel(alpha);
    end
  end
  ndu = ndu + dFu2;
  ndu(N*Q+find(model.w_isnan)) = 0; % because du = [df; dw]
  return;
end

if nargout == 3             % -dE/d log sigma_k
  % likelihood part
  ndlsk = (P*sum(fhat.^2) + sum(w.^2) + 2*Q*(N*P-sum(sum(model.Y_isnan)))*sk2)/sy2;
  % prior part
  Kfinv = Lf'\(Lf\eye(N));
  Kwinv = Lw'\(Lw\eye(N));
  ndlsk = ndlsk + Q*sum(diag(Kfinv));
  for i=1:P
    if sum(model.Y_isnan(:,i)) > 0 % missing data
      valid_ind = ~model.Y_isnan(:,i);
      Lwij = jit_chol(Kw(valid_ind,valid_ind))';
      ndlsk = ndlsk + Q*sum(diag(Lwij'\(Lwij\eye(size(Lwij,1)))));
    else
      ndlsk = ndlsk + Q*sum(diag(Kwinv));
    end  
  end
  ndlsk = sk2*ndlsk;
  return;
end

if nargout == 4             % -deE/d log theta
ltheta_f = ltheta(1:D+1);
ltheta_w = ltheta(D+2:end);
Lf = Lf'; 
Lw_full = Lw';
Kfinv = Lf\(Lf'\eye(N));
Kwinv_full = Lw_full\(Lw_full'\eye(N));

dtheta_f = zeros(size(ltheta_f)); dtheta_w = zeros(size(ltheta_w));
Fhat = reshape(fhat,N,Q); Wmat=reshape(w,N*P,Q);
W = mat2cell(Wmat,N*ones(1,P),ones(1,Q));
for j=1:Q
  dtheta_f = dtheta_f + grad_nlog_prior(ltheta_f,model.covfunc_f,X,Fhat(:,j));
  for i=1:P
    dtheta_w = dtheta_w + grad_nlog_prior(ltheta_w,model.covfunc_w,...
          X(~model.Y_isnan(:,i),:),W{i,j}(~model.Y_isnan(:,i)));
  end
end

dtheta2_f = zeros(size(ltheta_f));
for idx=1:D
  dK = feval(model.covfunc_f, ltheta_f, X, [], idx);
  dtheta2_f(idx) = Q*trace(Kfinv*dK*Kfinv);
end
if optim.optimise_sf
  dtheta2_f(D+1) = 2*Q*trace(Kfinv);
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
  for idx=1:D
    dK = feval(model.covfunc_w, ltheta_w, X(valid_ind,:), [], idx);
    dtheta2_w(idx) = dtheta2_w(idx) + Q*trace(Kwinv*dK*Kwinv);
  end
  if optim.optimise_sw
    dtheta2_w(D+1) = dtheta2_w(D+1) + 2*Q*trace(Kwinv);
  end
end

ndltheta = [dtheta_f; dtheta_w] - 0.5*sk2*[dtheta2_f; dtheta2_w];
if ~optim.optimise_sf, ndltheta(D+1) = 0; end
if ~optim.optimise_sw,  ndltheta(end) = 0; end
end

if nargout == 5
  % -dE/d log sigma_y
  ndlsy = N*P-sum(sum(model.Y_isnan)); % 0.5\sum_n P dlsy2/dlsy
  Fhat = reshape(fhat,N,Q); Wmat=reshape(w,N*P,Q);
  ndlsy = ndlsy + ndkdy(model.y,Fhat,Wmat,sy2,model)...  
      - (P*sum(fhat.^2) + sum(w.^2) + Q*(N*P-sum(sum(model.Y_isnan)))*sk2)*sk2/sy2;
end

end

% compute the expected log likelihood E[p(y|W, Fhat)]
function logl = expected_log_likelihood(W,Fhat,sy2,sk2,model)
Y = model.y; N = size(Y, 1);
P = model.num_outputs;
Q = model.num_latents;

%vectorised version
Wxn_ind = N*(0:P-1)';
W_ind = bsxfun(@plus,Wxn_ind,1:N);
Wblk = W(W_ind(:),:);
tt = repmat(1:N,P,1);
Fblk = Fhat(tt(:),:);
Ymean = reshape(sum(Wblk.*Fblk,2),P,N)';
Ydiff = (Y-Ymean).^2;
Ydiff(model.Y_isnan) = 0; % missing data part
logl = -0.5*sum(sum(Ydiff))/sy2;
logl = logl - 0.5*(N*P-sum(sum(model.Y_isnan)))*log(2*pi*sy2); % due to missing data
logl = logl - 0.5*P*sk2*sum(sum(Fhat.^2))/sy2 - 0.5*sk2*sum(sum(W.^2))/sy2...
  - 0.5*(N*P-sum(sum(model.Y_isnan)))*Q*(sk2^2)/sy2;
end

% compute the expected log prior E[p(f,w)]
function logprior = expected_logprior(f,w,Lf,Lw,model,sk2)
N = size(model.x,1); P = model.num_outputs; Q = model.num_latents;
F = reshape(f,N,Q); W = mat2cell(reshape(w,N*P,Q), N*ones(1,P), ones(1,Q));
Kfinv = Lf'\(Lf\eye(N));
logprior = -Q*sum(log(diag(Lf))) - 0.5*Q*sk2*trace(Kfinv);
Kw = Lw*Lw';
for j=1:Q
  %log_p = log_p - 0.5*F(:,j)'*Kf_inv*F(:,j);
  alpha = Lf\F(:,j);
  logprior = logprior - 0.5*(alpha')*alpha;
end  
for i=1:P
  valid_ind = ~model.Y_isnan(:,i);
  if sum(valid_ind) ~= N % missing data
    Lwi = jit_chol(Kw(valid_ind, valid_ind))';
  else
    Lwi = Lw;
  end
  for j=1:Q
    %log_p = log_p - 0.5*W{i,j}'*Kw_inv*W{i,j};
    alpha = Lwi\W{i,j}(valid_ind);
    logprior = logprior - 0.5*(alpha')*alpha;
  end
  Kwi_inv = Lwi'\(Lwi\eye(sum(valid_ind)));
  logprior = logprior - Q*sum(log(diag(Lwi))) - 0.5*Q*sk2*trace(Kwi_inv);
end
end

% - d log p(y|mu_k) / d log sigma_y
function dy = ndkdy(Y,Fhat,W,sy2,model)
[N,P] = size(Y);
Wxn_ind = N*(0:P-1)';
W_ind = bsxfun(@plus,Wxn_ind,1:N);
Wblk = W(W_ind(:),:);
tt = repmat(1:N,P,1);
Fblk = Fhat(tt(:),:);
Ymean = reshape(sum(Wblk.*Fblk,2),P,N)';
Ydiff = (Y-Ymean).^2;
Ydiff(model.Y_isnan) = 0; % missing data part
dy = -sum(sum(Ydiff))/sy2;
end

