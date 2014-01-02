function C = sq_dist(a)
    
    % Calculate squared distance.
    
    a = bsxfun(@minus,a,mean(a,2));
    C = bsxfun(@plus,sum(a.*a,1)',bsxfun(@minus,sum(a.*a,1),2*(a'*a)));
    C = max(C,0);          % numerical noise can cause C to negative 