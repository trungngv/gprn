function [nF nd] = npv_nELBO(Mu,lsk,lsy,Lf,Lw,params,func_arg,n,mu_n)
%NPV_NELBO [nF nd] = npv_nELBO(Mu,lsk,lsy,Lf,Lw,params,func_arg,n,mu_n)
% 
% Exact negative evidence lower bound.
%
% INPUTS:
%   Mu - [K x D] component means
%   lsk - [K x 1] log variance parameters
%   s2min - minimum bandwidth
%   func_arg: specify the argument to the lower bound function
%     0          for all variables
%     1          for the mean of component n
%     2          for the variance of component n
%   n : the component as described in func_arg == 1
%   mu_n : the mean of the component (if func_arg == 1 and n ~= [])
%
% OUTPUTS:
%   nF - negative approximate ELBO
%   d - gradient
%
% Trung V. Nguyen
% 11/10/2012
[K D] = size(Mu);
s2min = 1e-7;

% lower bound and/or derivatives w.r.t to component variance log sigma_k^2
if func_arg ~= 1
  nf1 = zeros(K,1); ndlsk1 = zeros(K,1);
  if nargout > 1
    for k=1:K % because we optimise all k together
      [nf1(k), ~, ndlsk1(k)] = npv_nlogjoint(Mu(k,:),lsk(k),[],lsy,Lf,Lw,params,2);
    end
    [nf2, ~, ndlsk2] = lower_bound_MoG([Mu 2*lsk],K,D,s2min,[],params);
    nd = ndlsk1/K + 2*ndlsk2'; % taking derivatives of lower_bound_MOG w.r.t log \sigma_k
  else
    for k=1:K
      nf1(k) = npv_nlogjoint(Mu(k,:),lsk(k),[],lsy,Lf,Lw,params,2);
    end
    nf2 = lower_bound_MoG([Mu 2*lsk],K,D,s2min,[],params);
  end
  
  nF = sum(nf1)/K + nf2;
  return;
end

% lower bound and/or derivatives w.r.t to component mean
Mu(n,:) = mu_n;
if nargout > 1
  [nf1, ndmu1] = npv_nlogjoint(Mu(n,:),lsk(n),[],lsy,Lf,Lw,params,1);
  [nf2, ndmu2] = lower_bound_MoG([Mu 2*lsk],K,D,s2min,n,params);
  nd = ndmu1/K + ndmu2';
else
  nf1 = npv_nlogjoint(Mu(n,:),lsk(n),[],lsy,Lf,Lw,params,1);
  nf2 = lower_bound_MoG([Mu 2*lsk],K,D,s2min,n,params);
end
nF = nf1/K + nf2;

end
