% Demonstration of hierarchical logistic regression with nonparametric
% variational inference

clear
rng(1110, 'twister');
% parameters
K = 10;         % number of covariates
D = K + 1;      % number of latent variables
T = 200;        % number of observations
N = 3;          % number of NPV components
a = 2; b = 1;   % regression hyperparameters

% generate simulated data
x = randn(T,K);                           % inputs (covariates)
alpha = gamrnd(a,b);                      % weight precision parameter
w = normrnd(0,1./alpha,1,K);              % regression weights
c = binornd(1,1./(1+exp(-x*w')));
c(c==0) = -1;                             % outputs (class labels)
nlogpdf = @(theta) logreg(theta,x,c,a,b); % negative log PDF

theta0 = randn(N,D+1);                    % initial parameters
[F mu s2] = npv_run(nlogpdf,theta0);      % run NPV

% plot results
figure;
scatter(w,mean(mu(:,1:end-1)))
lsline
xlabel('True weights');
ylabel('Predicted weights');

figure;
plot(F,'-o','LineWidth',2);
xlabel('Iteration');
ylabel('ELBO');

