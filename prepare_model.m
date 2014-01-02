function model = prepare_model(alg,x,y,num_modes,num_latents,std_mean)
%PREPARE_MODEL model = prepare_model(alg,x,y,num_modes,num_latents,std_mean)
%
% Prepare a model (gprn-mf or gprn-npv) for the dataset. 
% 
% INPUT:
%   - alg: 1 for gprn-mf and 2 for gprn-npv
%   - x : input feature matrix (NxD)
%   - y : output feature matrix (NxP)
%   - num_latents : number of latent functions to use
%   - num_modes : number of modes for NPV (pass [] for meanfield)
%   - std_mean : true to standarise the outputs
%
% OUTPUT:
%   - model : the initialized model with all necessary parameters
%
% Trung V. Nguyen
% 19/11/12
%

% two cases:
% 1) unseen inputs -> corresponding output sets to nan
% 2) missing inputs 

GPRNMF = 1;
GPRNNPV = 2;
num_outputs = size(y,2);
model = struct('num_latents', num_latents, 'num_outputs', num_outputs,...
  'covfunc_f', 'covSEard', 'covfunc_w', 'covSEard', 'std_mean', std_mean);
if alg == GPRNNPV
  model.num_modes = num_modes;
end
model.x = x;
if std_mean
  [model.y,model.ymean,model.ystd] = standardize(y,[],[]);  
else
  model.y = y;
end

% for missing data
model.Y_isnan = isnan(model.y);
model.Wcell_isnan = cell(num_outputs, num_latents);
for i=1:num_outputs
  for j=1:num_latents
    model.Wcell_isnan{i,j} = false(size(model.y,1),1);
    model.Wcell_isnan{i,j}(model.Y_isnan(:,i)) = true;
  end
end
model.W_isnan = cell2mat(model.Wcell_isnan);
model.w_isnan = model.W_isnan(:);

