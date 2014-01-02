function [mae,smse,ystar] = mf_predict(xtest,ytest,mu,hyp,model)
%MF_PREDICTION [mae smse ystar] = mf_predict(xtest,ytest,mu,theta,model)
% 
% Prediction using mean-field approximation.
%
% INPUT
%   - xtest : test input features (Ntest x dim)
%   - ytest : test output features (Ntest x P) (for computing error measurements)
%   - mu : all learned variational parameters
%   - theta : all learned hyper-parameters
%   - model : the trained model
%
% See also MF_TRAIN
%
% OUTPUT
%   - mae : mean absolute error
%   - smse : standardised mean squared error (see mysmse.m)
%   - ystar : predicted values (Ntest x P)
%
% Trung V. Nguyen
% 20/10/13
P = model.num_outputs;
Q = model.num_latents;
[Ntest D] = size(xtest);
X = model.x;
Ntrain = size(X,1);
ltheta_f = hyp(1:D+1);
ltheta_w = hyp(D+2:end);

Kf = feval(model.covfunc_f, ltheta_f, model.x);% + (sf^2) * eye(N);
Kw = feval(model.covfunc_w, ltheta_w, model.x);
Lf = jit_chol(Kf)'; Lw = jit_chol(Kw)';

[ft wt] = u_to_fhat_w(mu, model);
% Step 1: Compute expected value of w for the missing data
if sum(model.w_isnan) > 0
wt(model.w_isnan) = 0;
W = mat2cell(reshape(wt(:), Ntrain*P,Q), Ntrain*ones(1,P), ones(1,Q));
for p=1:P
  for q=1:Q
    if (sum(model.Wcell_isnan{p,q})) > 0
      obs_ind = ~model.Wcell_isnan{p,q};
      Kw = feval(model.covfunc_w, ltheta_w, X(obs_ind,:), X(obs_ind,:));
      Lw_obs = jit_chol(Kw)';
      kw_s = feval(model.covfunc_w, ltheta_w, X(obs_ind,:), X(model.Wcell_isnan{p,q},:));
      W{p,q}(model.Wcell_isnan{p,q}) = (Lw_obs\kw_s)' * (Lw_obs\W{p,q}(obs_ind));
    end  
  end
end
wt = cell2mat(W);
mu = [ft(:); wt(:)];
end

% Step 2: Prediction
ystar = zeros(P,Ntest);
[ft wt] = u_to_fhat_w(mu, model);
ft = ft(:); wt = wt(:);
wt(model.w_isnan) = 0;
%wt = wt(~params.w_isnan); % discard missing data
for i=1:Ntest
  kf_s = feval(model.covfunc_f, ltheta_f, X, xtest(i,:));
  kw_s = feval(model.covfunc_w, ltheta_w, X, xtest(i,:));
  alphaLf = (Lf\kf_s)'; alphaLw = (Lw\kw_s)';
  idx_f = 1; idx_w = 1;
  Wstar = zeros(P,Q); fstar = zeros(Q,1);
  for q=1:Q
    fstar(q) = alphaLf * (Lf\ft(idx_f:idx_f+Ntrain-1));
    idx_f = idx_f + Ntrain;
    for p=1:P
      Wstar(p,q) = alphaLw * (Lw\wt(idx_w:idx_w+Ntrain-1));
      idx_w = idx_w + Ntrain;
    end
  end
  ystar(:,i) = ystar(:,i) + Wstar * fstar;
end

if model.std_mean
  Ymean = repmat(model.ymean,Ntest,1);
  Ystd = repmat(model.ystd,Ntest,1);
  ystar = ystar'.*Ystd + Ymean;
else
  ystar = ystar';
end

mae = mean(abs(ystar-ytest));
smse = mysmse(ystar, ytest);
  
fprintf('mae = %.4f\n', mae);
fprintf('smse = %.4f\n', smse);

