README for the nonparametric variational Bayes (NPV) algorithm, as described in:
Gershman, Hoffman & Blei (2012). Nonparametric variational inference. International Conference on Machine Learning.

The main function is npv_run.m. See npv_demo_logreg for a demonstration on a hierarchical logistic regression model (this model is described in section 5.1 of the paper).

Note that if you nonlinearly transforming your variables to be real-valued, you need to add the appropriate Jacobian factor to the objective function.

The algorithm uses the optimization routines in Mark Schmidt's minFunc package, included as a sub-directory. Specifically, it uses the L-BFGS algorithm for optimization.

This code comes with no guarantees whatsoever. Please direct questions to Sam Gershman (sam.gershman@gmail.com)