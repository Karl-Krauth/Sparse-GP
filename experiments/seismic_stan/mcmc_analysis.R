load('seismic.RData')
gp_mult_ss <- rstan::extract(gp_mult)
f <- gp_mult_ss$f

dim(f[10000:12000,,])
f_mean = apply(f[10000:12000,,], c(2,3), mean)
dim(f_mean)
f_mean[4:113]
dim(f_mean)
typeof(f)
f[200,]
y <- gp_mult_ss$y[200,]