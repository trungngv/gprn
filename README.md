Efficient Variational Inference for Gaussian Process Regression Networks package. 

This package implements the two variational inference methods for GPRNs of Nguyen and Bonilla [1].

Author: 
-----
Trung V. Nguyen (trung.ngvan@gmail.com)

Last update: 03/04/2013


Main Content
-----------
1. meanfield/mf_train.m : train a GPRN model with mean-field inference method
2. meanfield/mf_predict.m : predict using a trained mean-field model obtained from mf_train.m
3. npv/npv_train.m : train a GPRN model with npv inference method
4. npv/npv_predict.m : predict using a trained npv model obtained from npv_predict.m
5. demo.m : A script demonstrating the use of mean-field and npv on a toy dataset.

Dependencies
------------------
This code uses two external libraries: the gpml package [2] and the code in [3]
with modifications by Trung V. Nguyen to handle high-dimensional datasets.

References
-----------
[1] Trung V. Nguyen and Edwin V. Bonilla.
"Efficient Variational Inference for Gaussian Process Regression Networks"
In Proceedings of the 16th International Conference on Artificial Intelligence and Statistics (AISTATS)
2013, Scottsdale, AZ, USA.

[2] Rasmussen, Carl Edward, and Hannes Nickisch.
"Gaussian processes for machine learning (GPML) toolbox."
The Journal of Machine Learning Research 11 (2010): 3011-3015.

[3] Gershman, Samuel, Matt Hoffman, and David Blei.
"Nonparametric variational inference."
arXiv preprint arXiv:1206.4665 (2012)

