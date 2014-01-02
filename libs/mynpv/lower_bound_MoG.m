function [H dmu ds2] = lower_bound_MoG(theta,K,D,s2min,n,params)
%LOWER_BOUND_MOG [H dmu ds2] = lower_bound_MoG(theta,K,D,s2min,n,params)
%
% Lower bound on the entropy of a mixture of Gaussians.
%
% Derivatives are w.r.t to parameters of one component.
%
% INPUT:
%   - theta : K x (D + 1) where the first D columns are mean components and
%   the last column is variance (parametrised as log sigma_k^2)
%   - K : number of components
%
% OUTPUT:
%   - H
%   - dmu : 
%   - ds2 : derivative wrt log sigma_k2
%
% Trung Nguyen

% theta is a Kx(D+1) matrix where the KxD is mu and the last column is
% log(sigma^2)
mu = theta(:,1:D);
s2 = exp(theta(:,end))' + s2min;
beta = ones(K,1)./K;

if ~isempty(params)
  mu(:,params.w_isnan) = 0; % zero contribution from missing data for weight functions
end  
S = my_sq_dist(mu'); % S_nj = (mu_n-mu_j)'(mu_n-mu_j)
s = bsxfun(@plus,s2,s2'); % s_nj = sigma_n^2 + sigma_j^2
logP = -0.5.*S./s - 0.5*D*log(s); % P = N(mu_n; mu_j, s_nj)
a = zeros(1,K);
for i=1:K
  a(i) = -log(K) + logsum(logP(i,:)); % a_i = logq_i = log(1/K) + logsum N(mu_i; mu_j, si2 + sj2))
end
H = a*beta;
    
%---------- derivatives -----------------%
dmu = zeros(1,D);
if nargout == 2
  dmu = zeros(1,D);
  for j=1:K
    pp = 1/sum(exp(logP(n,:)-logP(n,j))) + 1/sum(exp(logP(j,:)-logP(n,j)));
    dmu = dmu + (mu(j,:)-mu(n,:))/s(n,j)*pp;
  end
  dmu=dmu/K;
  if ~isempty(params)
    dmu(params.w_isnan) = 0; % always set gradients of missing data to 0
  end  
end
    
if nargout == 3
ds2 = zeros(size(s2));
for i=1:K
  for j=1:K
    pp = 1/sum(exp(logP(i,:)-logP(i,j))) + 1/sum(exp(logP(j,:)-logP(i,j)));
    ds2(i) = ds2(i) + (S(i,j)/s(i,j)^2 - D/s(i,j))*pp;
  end
end
ds2 = ds2.*(s2-s2min)*0.5/K;

end

