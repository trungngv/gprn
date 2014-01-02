function F = ELBO(theta,nlogpdf,h,s2min,params)
    
    % Calculate evidence lower bound (2nd-order approximation, L2).
    % For checking covergence
    
    [K D] = size(theta); D = D - 1;
    nF = lower_bound_MoG(theta,K,D,s2min,[],params);
    s2 = exp(theta(:,end)) + s2min;
    F = -nF;
    for k = 1:K
      nf = nlogpdf(theta(k,1:end-1));
      F = F - nf/K - 0.5*s2(k)*h(k)/K;
    end
    
    