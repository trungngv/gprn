function [H dmu ds2] = deprecated_lower_bound_MoG(theta,N,D,s2min,n)
% Lower bound on the entropy of a mixture of Gaussians.

eps = 0;

% theta is a Nx(D+1) matrix where the NxD is mu and the last column is
% log(sigma^2)
mu = theta(:,1:D);
s2 = exp(theta(:,end))' + s2min;
beta = ones(N,1)./N;

S = sq_dist(mu'); % S_nj = (mu_n-mu_j)'(mu_n-mu_j)
s = bsxfun(@plus,s2,s2'); % s_nj = sigma_n^2 + sigma_j^2
% Old code: blow up for high dimensional case    
% P = exp(-0.5.*S./s)./((2*pi*s).^(D/2));
% a = beta'*P; % [q_1 ... q_n]
% H = log(a)*beta;
% q = a;
%logP = log(P);

% this somehow is an approximation and not the exact value
% hence gradients are also approximation
logP = -0.5.*S./s - 0.5*D*log(2*pi*s); % P = N(mu_n; mu_j, s_nj)
a = zeros(1,N);
for i=1:N
  a(i) = -log(N) + logsum(logP(i,:)); % a_i = logq_i = log(1/N) + logsum N(mu_i; mu_j, si2 + sj2))
end
H = a*beta;
%P = exp(logP) + rand*eps; % to ensure that dmu and ds2 not = 0 due to P too small
%q = exp(a) + rand*eps; % to avoid numeric problem
    
%---------- derivatives -----------------%
dmu = zeros(1,D);
if nargout > 1 && nargout < 3
  % old code
%   dm = bsxfun(@plus,mu(n,:),-mu);
%   for d = 1:D
%     dS = zeros(N);
%     dS(n,:) = dm(:,d);
%     dS(:,n) = dm(:,d);
%     dP = -P.*dS./s;
%     dmu(d) = ((beta'*dP)./q)*beta;
%   end

  % my code: for efficiency and stability
  dmu = zeros(1,D);
  for j=1:N
    pp = 1/sum(exp(logP(n,:)-logP(n,j))) + 1/sum(exp(logP(j,:)-logP(n,j)));
    dmu = dmu + (mu(j,:)-mu(n,:))/s(n,j)*pp;
  end
  dmu=dmu/N;
end
    
if nargout > 2
% old code
%   ds2 = s2;
%   for i = 1:N
%     ds = zeros(N);
%     ds(i,:) = 1; ds(:,i) = 1; ds(i,i) = 2;
%     dP = P.*ds.*0.5.*(S./(s.^2) - D./s);
%     % this is because we are using log(s2) as the argument/variable
%     % for the function (and takes derivative wrt log(s2)
%     ds2(i) = (s2(i)-s2min).*((beta'*dP)./q)*beta;
%   end
%   ds2 = ds2./(s2-s2min);
  
ds2 = zeros(size(s2));
for n=1:N
  for j=1:N
    pp = 1/sum(exp(logP(n,:)-logP(n,j))) + 1/sum(exp(logP(j,:)-logP(n,j)));
    ds2(n) = ds2(n) + (S(n,j)/s(n,j)^2 - D/s(n,j))*pp;
  end
end
ds2 = ds2.*(s2-s2min)*0.5/N;

end

