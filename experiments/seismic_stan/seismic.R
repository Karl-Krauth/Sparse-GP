library("rstan")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

library("ggplot2")

data = read.csv("../data/seismic/data.csv", header = F)
y = t(data[, 1:4])
x = data[, 5]

N <- length(x)

P <- nrow(y)

mu_prior = c(200, 500, 1600, 2200, 1950, 2300, 2750, 3650)
var_prior = c(900, 5625, 57600, 108900, 38025, 52900, 75625, 133225)
sigma2y = c(0.025, 0.05, 0.075, 0.1) ** 2

start_time <- Sys.time()

gp_mult <- stan(file = 'seismic.stan', data = list(x = as.numeric(x), P = P, N = N, N_latent=8, y=y, 
                                                   mu_prior=mu_prior,
                                                   var_prior = var_prior,
                                                   sigma2y = sigma2y
                                                   ), chains = 3, iter = 2000, seed = 123, control = list(adapt_delta = 0.95, max_treedepth = 15))
end_time <- Sys.time()

save.image('seismic.RData')