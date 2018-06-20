setwd('experiments/seismic_stan')

library("rstan")
library("ggplot2")

set.seed(123)

x <- (-50:50)/25
N <- length(x)

x1 <- x
x2 <- x
x_mat <- expand.grid(x1, x2)
x <- x_mat[sample(dim(x_mat)[1], N),]
D <- 2

gp_mult <- stan(file = 'seismic.stan', data = list(x = x, D = D, N = N), chains = 3, iter = 200, seed = 123)
gp_mult_ss <- rstan::extract(gp_mult)
y <- gp_mult_ss$y[200,]
