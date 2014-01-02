% synthetic data
Ntrain = 200; Ntest = 50; D = 3;
num_latents = 1; num_outputs = 2;
sigma_y = 0.05;
theta_f = [0.3; 0.1; 1];
theta_w = [2; 2; 1];
fx = @(x) cos(x(:,1)).^2 + sin(x(:,2)) + x(:,1).*x(:,2).*cos(x(:,3));
x = rand(Ntrain,D);
xtest = rand(Ntrain,D);
y = [fx(x),-fx(x)]; % output1 = -output2
ytest = [fx(xtest),-fx(xtest)];

% npv train 
num_modes = 1;
[mu,hyp,model] = npv_train(x,y,false,num_modes,num_latents,[],100);
% npv prediction
[mae,smse,ystar] = npv_predict(xtest,ytest,mu,hyp,model);

% mean-field train
[mu,hyp,model] = mf_train(x,y,false,num_latents,[],100);

% mf prediction
[mae,smse,ystar]= mf_predict(xtest,ytest,mu,hyp,model);

