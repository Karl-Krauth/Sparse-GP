setwd('experiments/seismic_stan')

library("rstan")
library("ggplot2")

set.seed(123)

data = read.csv("../data/seismic/data.csv", header = F)
y = t(data[, 1:4])
x = data[, 5]

N <- length(x)

D <- nrow(y)

mu_prior = c(200, 500, 1600, 2200, 1950, 2300, 2750, 3650) * 0
var_prior = c(900, 5625, 57600, 108900, 38025, 52900, 75625, 133225)

y = list(as.numeric(y[1]), as.numeric(y[2]), as.numeric(y[3]), as.numeric(y[4]))

gp_mult <- stan(file = 'seismic.stan', data = list(x = as.numeric(x), D = D, N = N, N_latent=8, y=y, 
                                                   mu_prior=mu_prior,
                                                   var_prior = var_prior
                                                   ), chains = 3, iter = 200, seed = 123)
gp_mult_ss <- rstan::extract(gp_mult)
y <- gp_mult_ss$y[200,]
