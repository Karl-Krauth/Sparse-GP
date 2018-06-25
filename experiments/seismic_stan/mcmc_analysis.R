load('seismic.RData')
gp_mult_ss <- rstan::extract(gp_mult)
f <- gp_mult_ss$f

f_mean = apply(f, c(2,3), mean)
f_mean[4,1:113]
dim(f_mean)
typeof(f)
f[200,]
y <- gp_mult_ss$y[200,]