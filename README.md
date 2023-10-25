# Discriminative Calibration

Yuling Yao & Justin Domke. (2023). Discriminative calibration. Neurips.

[link to paper](https://arxiv.org/abs/2305.14593). 


"Discriminative calibration" is a flexible diagnostic tool to check if approximate Bayesian computation is accurate. We can use it to diagnose both the simulation-based/likelihood-free inference, and traditional Markov chain Monte Carlo or variational inference. 
It returns an interpretable divergence measure of miscalibration, computed from classification accuracy. This measure typically has a higher statistical power than the traditional rank-based SBC test. 

We generalize the binary two-sample test and include various feature-mapping. Here we implement two types of classifier calibration, binary and multiclass, and both are based on MLP. The multiclass classifier has been designed to incorport autocorrelated sampler such as  Markov chain Monte Carlo.  The required input is an amortized simulation table containing parameters from priors, data from the forward model, and approximate inference. We recommend adding additional statistically-meaningful features, such as log prior, log likelihood, log approximate posteror density, and ranks. 


