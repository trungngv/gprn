function h = hesstrace(func,theta)
    
    % Calculate Hessian trace using finite differencing.
    %
    % USAGE: h = hesstrace(func,theta)
    
    ep = 2*sqrt(1e-12)*(1+norm(theta))/norm(length(theta));
    h = 0;
    
    for d = 1:length(theta)
        f = func(theta);
        theta(d) = theta(d) + ep;
        a = func(theta);
        theta(d) = theta(d) - 2*ep;
        b = func(theta);
        h = h + (a + b - 2*f)./(ep.^2);
    end