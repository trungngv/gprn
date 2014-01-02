function [mu hyp model] = mf_train(x,y,std_mean,num_latents,hyp0,max_iters)
%MF_TRAIN [mae, smse] = MF_train(x,y,std_mean,num_latents,hyp0,max_iters)
%
% Trains a model with mean-field approximation.
%
% Usage:
% [mu hyp model] = MF_TRAIN(x,y,std_mean,num_latents,hyp0,max_iters)
% where all of the arguments except x and y can be empty in which case they 
% take on default values.
%
% INPUT
%   - x : train input features matrix (Ntrain x dim)
%   - y : train output matrix (Ntrain x P)
%   - std_mean : true to normalise the outputs to have zero mean (default =
%   true)
%   - num_latenst : number of latent functions (default = 1)
%   - hyp0 : initial hyper-parameters in the form of
%     [log \theta_f; \log \theta_w; log \sigma_noise] (default = random)
%   - max_iters : max number of optimisation iterations
%
% OUTPUT
%   - mu : all learned variational parameters
%   - hyp : all learned hyper-parameters 
%   - model : the trained model
%
% Trung V. Nguyen
% 19/10/12

% model configuration
if isempty(num_latents)
  num_latents = 1;
end
if isempty(std_mean)
  std_mean = true;
end
num_outputs = size(y,2);
model = prepare_model(1,x,y,[],num_latents,std_mean);

% init hyper-parameters and variational parameters
if isempty(hyp0)
  hyp0 = hyp_init(size(x,2));
end  
D = size(x,1)*num_latents*(num_outputs+1);
mu = randn(D,1);
var = rand(D,1);

% optimisation configuration
optim.optimise_sf = false;
optim.optimise_sw = true;
optim.optimise_hyp = true;
optim.max_iters = max_iters;

% run
[F mu var hyp lsy] = mf_optimise(mu,var,hyp0(1:end-1),hyp0(end),model,optim);
end

function [F mu var hyp lsy] = mf_optimise(mu,var,hyp,lsy,model,optim)
tol = 0.00001; F = zeros(optim.max_iters,1);
opts = struct('Display','off','Method','lbfgs','MaxIter',200,...
  'MaxFunEvals',200,'DerivativeCheck','off'); 

% pre-compute Lf, Lw 
X = model.x; Dx = size(X,2); Q = model.num_latents;
tbegin = tic;
for iter = 1:optim.max_iters
  if iter == 1 || optim.optimise_hyp
    if ~optim.optimise_sf,  hyp(Dx+1) = log(1);  end
    if ~optim.optimise_sw,  hyp(end) = log(1);   end
    disp(['hyp = ', num2str(exp([hyp' lsy]))])
    Kf = feval(model.covfunc_f, hyp(1:Dx+1), X);
    Kw = feval(model.covfunc_w, hyp(Dx+2:end), X);
    Lf = jit_chol(Kf)'; Lw = jit_chol(Kw)';
  end  
  if iter > 1,
    F(iter) = -mf_nlowerbound(mu,var,[],lsy,Lf,Lw,model,-1);
  end
  fprintf('iter %d \t  %.5f\n', iter, F(iter));
  if iter > 1 && abs(F(iter)-F(iter-1)) < tol; break; end
  
  % optimise mu and var
  [~,~,~,mu,var,model] = mf_nlowerbound(mu,var,[],lsy,Lf,Lw,model,0);
        
  % optimise noise
  func = @(x) nbound_lsy(x,mu,var,[],Lf,Lw,model);
  lsy = minFunc(func,lsy,opts);
  
  % optimise hyp of covariance function
  if optim.optimise_hyp
    func = @(x) nbound_ltheta(x,mu,var,lsy,model,optim);
    hyp = minFunc(func, hyp, opts);
  end  
end
fprintf('\n\nmean-field training completed in %.2f(s)\n', toc(tbegin));
end

function [nf ng] = nbound_ltheta(ltheta,mu,var,lsy,model,optim)
X = model.x; D = size(X,2);
% Must 'synchronise' Var(Wij) and CovW{i,j} when Kw changes!
if ~optim.optimise_sf,   ltheta(D+1) = log(1); end
if ~optim.optimise_sw,   ltheta(end) = log(1); end
Kf = feval(model.covfunc_f, ltheta(1:D+1), X);
Kw = feval(model.covfunc_w, ltheta(D+2:end), X);
[Lf, ~] = jit_chol(Kf); [Lw, ~] = jit_chol(Kw);
Lf = Lf'; Lw = Lw';
if nargout > 1
  [nf, ng] = mf_nlowerbound(mu,var,ltheta,lsy,Lf,Lw,model,1,optim);
else
  nf = mf_nlowerbound(mu,var,ltheta,lsy,Lf,Lw,model,1);
end
end

function [nf ng] = nbound_lsy(lsy,mu,var,ltheta,Lf,Lw,model)
if nargout > 1
  [nf, ~, ng] = mf_nlowerbound(mu,var,ltheta,lsy,Lf,Lw,model,2);
else
  nf = mf_nlowerbound(mu,var,ltheta,lsy,Lf,Lw,model,2);
end
end

