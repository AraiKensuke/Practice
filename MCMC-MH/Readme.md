Reversible Jump MCMC for Binomial-Negative Binomial model.

Implementation of a RJMCMC that allows fitting of count data to either the BN or NB model.  The BN model accommodates variances that are less than the mean, while the NB model accommodates variances greater than the mean.  By allowing MCMC to explore both models, we can fit count models whose variance may be less than or greater than the mean.

