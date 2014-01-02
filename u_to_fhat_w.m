function [fhat,w] = u_to_fhat_w(u, model)
%U_TO_FHAT_W [fhat,w] = u_to_fhat_w(u, model)
%
%
u = u(:);
N = size(model.x, 1);
fhat = u(1:(N*model.num_latents));
w = u(length(fhat)+1:end);
end

