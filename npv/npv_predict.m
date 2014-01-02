function [mae smse ystar] = npv_predict(xtest,ytest,mu,theta,model)
%NPV_PREDICT [mae smse ystar] = npv_predict(xtest,ytest,mu,theta,model)
% 
% Prediction using NPV approximation.
%
% INPUT
%   - xtest : test input features (Ntest x dim)
%   - ytest : test output features (Ntest x P) (for computing error measurements)
%   - mu : all learned variational parameters
%   - theta : all learned hyper-parameters
%   - model : the trained model
%
% See also NPV_TRAIN
%
% OUTPUT
%   - mae : mean absolute error
%   - smse : standardised mean squared error (see mysmse.m)
%   - ystar : predicted values (Ntest x P)
%
% Trung V. Nguyen
% 02/04/13
P = model.num_outputs;
Q = model.num_latents;
K = model.num_modes;
[Ntest D] = size(xtest);
X = model.x;
Ntrain = size(X,1);
ltheta_f = theta(1:D+1);
ltheta_w = theta(D+2:end);

Kf = feval(model.covfunc_f, ltheta_f, model.x);% + (sf^2) * eye(N);
Kw = feval(model.covfunc_w, ltheta_w, model.x);
Lf = jit_chol(Kf)'; Lw = jit_chol(Kw)';

% Step 1: Compute the mean of f and w for the missing data
for k=1:K
  [ft wt] = u_to_fhat_w(mu(k,:), model);
  ft = ft(:);
  if sum(model.w_isnan) == 0
    break;
  end
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
  mu(k,:) = [ft; wt(:)];
end

% Step 2: Prediction
ystar = zeros(P,Ntest);
for k=1:K
  [ft wt] = u_to_fhat_w(mu(k,:), model);
  ft = ft(:); wt = wt(:);
  fstar = zeros(Q,1);   Wstar = zeros(P,Q);
  for i=1:Ntest
    kf_s = feval(model.covfunc_f, ltheta_f, X, xtest(i,:));
    kw_s = feval(model.covfunc_w, ltheta_w, X, xtest(i,:));
    alphaLf = (Lf\kf_s)'; alphaLw = (Lw\kw_s)';
    idx_f = 1; idx_w = 1;
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
end

if model.std_mean
  Ymean = repmat(model.ymean,Ntest,1);
  Ystd = repmat(model.ystd,Ntest,1);
  ystar = (ystar'/K).*Ystd + Ymean;
else
  ystar = ystar'/K;
end

mae = mean(abs(ystar-ytest));
smse = mysmse(ystar,ytest);

fprintf('mae = %.4f\n', mae);
fprintf('smse = %.4f\n', smse);


