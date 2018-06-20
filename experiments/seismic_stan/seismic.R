setwd('experiments/seismic_stan')

library("rstan")
library("ggplot2")

set.seed(123)

data = read.csv("../data/seismic/data.csv", header = F)
y = t(data[, 1:4])
x = data[, 5]

N <- length(x)

P <- nrow(y)

mu_prior = c(200, 500, 1600, 2200, 1950, 2300, 2750, 3650)
var_prior = c(900, 5625, 57600, 108900, 38025, 52900, 75625, 133225)
sigma2y = c(0.0006, 0.0025, 0.0056, 0.0100)

gp_mult <- stan(file = 'seismic.stan', data = list(x = as.numeric(x), P = P, N = N, N_latent=8, y=y, 
                                                   mu_prior=mu_prior,
                                                   var_prior = var_prior,
                                                   sigma2y = sigma2y
                                                   ), chains = 1, iter = 200, seed = 123)
gp_mult_ss <- rstan::extract(gp_mult)
f <- gp_mult_ss$f

f_mean = apply(f, c(2,3), mean)
f_mean[1,1:113]
dim(f_mean)
typeof(f)
f[200,]
y <- gp_mult_ss$y[200,]
