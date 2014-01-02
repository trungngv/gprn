function [mu hyp model] = npv_train(x,y,std_mean,num_modes,num_latents,hyp0,max_iters)
%NPV_TRAIN [mae, smse] = npv_train(x,y,std_mean,num_modes,num_latents,hyp0,max_iters)
%
% Trains a model with NPV approximation.
%
% Usage:
% [mu hyp model] = NPV_TRAIN(x,y,std_mean,num_modes,num_latents,hyp0,max_iters)
% where all of the arguments except x and y can be empty in which case they 
% take on default values.
%
% INPUT
%   - x : train input features matrix (Ntrain x dim)
%   - y : train output matrix (Ntrain x P)
%   - std_mean : true to normalise the outputs to have zero mean (default =
%   true)
%   - num_modes : number of mixture components (default = 1)
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
if isempty(num_modes)
  num_modes = 1;
end
if isempty(num_latents)
  num_latents = 1;
end
if isempty(std_mean)
  std_mean = true;  
end
num_outputs = size(y,2);
model = prepare_model(2,x,y,num_modes,num_latents,std_mean);

% init hyper-parameters and variational parameters
if isempty(hyp0)
  hyp0 = hyp_init(size(x,2));
end  
D = size(model.x,1)*num_latents*(num_outputs+1);
theta0 = randn(num_modes, D+1);
theta0(:,D+1) = log(1);

% optimisation configuration
optim.optimise_sf = false;
optim.optimise_sw = true;
optim.optimise_hyp = true;
optim.max_iters = max_iters;

[F mu s2 hyp lsy] = npv_optimise(theta0,hyp0(1:end-1),hyp0(end),model,optim);
end

function [F mu s2 hyp lsy] = npv_optimise(theta,hyp,lsy,model,optim)
tol = 0.00001; F = zeros(optim.max_iters,1);
[K D] = size(theta); D = D - 1;
opts = struct('Display','off','Method','lbfgs','MaxIter',200,...
  'MaxFunEvals',200,'DerivativeCheck','off'); 

tbegin = tic;
X = model.x; Dx = size(X,2); Q = model.num_latents;
% init Lf and Lw
if ~optim.optimise_sf,  hyp(Dx+1) = log(1);  end
if ~optim.optimise_sw,  hyp(end) = log(1);   end
disp(['hyp = ', num2str(exp([hyp' lsy]))])
Kf = feval(model.covfunc_f, hyp(1:Dx+1), X);
Kw = feval(model.covfunc_w, hyp(Dx+2:end), X);
Lf = jit_chol(Kf)'; Lw = jit_chol(Kw)';
for iter = 1:optim.max_iters
  F(iter) = -npv_nELBO(theta(:,1:D),theta(:,D+1),lsy,Lf,Lw,model,0);
  fprintf('iter %d\t %.5f\n', iter, F(iter));
  if iter > 1 && abs(F(iter)-F(iter-1)) < tol; break; end
  
  % optimise component mean
  for k = 1:K
    func = @(x) npv_nELBO(theta(:,1:D),theta(:,D+1),lsy,Lf,Lw,model,1,k,x);
    theta(k,1:D) = minFunc(func,theta(k,1:D)',opts);
  end
        
  % optimise hyper-parameters
  if optim.optimise_hyp
    func = @(x) nbound_ltheta(x,theta(:,1:D),theta(:,D+1),lsy,model,optim);
    hyp = minFunc(func, hyp, opts);
    %update Lf and Lw
    if ~optim.optimise_sf,  hyp(Dx+1) = log(1);  end
    if ~optim.optimise_sw,  hyp(end) = log(1);   end
    Kf = feval(model.covfunc_f, hyp(1:Dx+1), X);
    Kw = feval(model.covfunc_w, hyp(Dx+2:end), X);
    Lf = jit_chol(Kf)'; Lw = jit_chol(Kw)';
  end  
  
  % optimise noise
  func = @(x) nbound_lsy(x,theta(:,1:D),theta(:,D+1),[],Lf,Lw,model);
  lsy = minFunc(func,lsy,opts);
  
  % optimise component variance log \sigma_k^2
  func = @(x) npv_nELBO(theta(:,1:D),x,lsy,Lf,Lw,model,2);
  theta(:,D+1) = minFunc(func,theta(:,D+1),opts);
end
fprintf('\n\nnpv learned completed in %.2f(s)\n', toc(tbegin));
    
mu = theta(:,1:D);
s2 = exp(theta(:,end));
end

function [nf ng] = nbound_ltheta(ltheta,Mu,lsk,lsy,model,optim)
nf = 0; ng = zeros(size(ltheta));
X = model.x; D = size(X,2);
if ~optim.optimise_sf,   ltheta(D+1) = log(1); end
if ~optim.optimise_sw,   ltheta(end) = log(1); end
Kf = feval(model.covfunc_f, ltheta(1:D+1), X);
Kw = feval(model.covfunc_w, ltheta(D+2:end), X);
[Lf, ~] = jit_chol(Kf); [Lw, ~] = jit_chol(Kw);
Lf = Lf'; Lw = Lw';
for n=1:model.num_modes
  if nargout > 1
    [f, ~, ~, g] = npv_nlogjoint(Mu(n,:),lsk(n),ltheta,lsy,Lf,Lw,model,3,...
      optim);
    ng = ng + g;
  else
    f = npv_nlogjoint(Mu(n,:),lsk(n),ltheta,lsy,Lf,Lw,model,3,[]);
  end
  nf = nf + f;
end  
end

function [nf ng] = nbound_lsy(lsy,Mu,lsk,ltheta,Lf,Lw,model)
nf = 0; ng = 0; K = model.num_modes;
for n=1:K
  if nargout > 1
    [f, ~, ~, ~, g] = npv_nlogjoint(Mu(n,:),lsk(n),ltheta,lsy,Lf,Lw,model,4);
    ng = ng + g;
  else
    f = npv_nlogjoint(Mu(n,:),lsk(n),ltheta,lsy,Lf,Lw,model,4);
  end
  nf = nf + f;
end
end

