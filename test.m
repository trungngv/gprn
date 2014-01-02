addpath(genpath('./'));

alpha = 10;
% synthetic data
Ntrain = 100; Ntest = 1000; D = 1;
num_latents = 1; num_outputs = 1;
sigma_y = 0.000005;
% theta_f = [0.1; ];
% theta_w = [0.3;];
x = linspace(-1,1,Ntrain)';
xtest = linspace(-1,1,Ntest)';
y = atan(alpha*x);
ytest = zeros(Ntest,1);

h = figure;
hold on;
plot(x,y,'.', 'markersize',14);
disp('press any key');


% mean-field train
[mu,hyp,model] = mf_train(x,y,false,num_latents,[],200);

% mf prediction
[mae,smse,ystar]= mf_predict(xtest,ytest,mu,hyp,model);

plot(xtest,ystar);

% disp('press any key');
% pause

% close(h);
% clear all;
