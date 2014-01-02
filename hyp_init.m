function hyp = hyp_init(D)
%HYP_INIT hyp = hyp_init(D)
% 
thetaf = rand(D+1,1);
thetaw = [rand(D,1); 0];
hyp = [thetaf; thetaw; log(rand)];
end
