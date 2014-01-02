function f = entropy(model)
%ENTROPY f = entropy(model)
% 
% Computes entropy of the posterior approximation.
%
f = 0;
model.num_latents;
for j=1:model.num_latents
  L = jit_chol(model.CovF{j});
  f = f + sum(log(diag(L)));
  for i=1:model.num_outputs
    L = jit_chol(model.CovW{i,j});
    f = f + sum(log(diag(L)));
  end
end
end

