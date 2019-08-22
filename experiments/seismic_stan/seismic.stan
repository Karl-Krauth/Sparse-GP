data {
    int<lower=1> N;
    int<lower=1> P;
    int<lower=1> N_latent;
    real x[N];
    vector[N] y[P];
    vector[N_latent] mu_prior;
    vector[N_latent] var_prior;
    vector[P] sigma2y;
}

transformed data {
}
parameters {
    vector[N] f[N_latent];
}

model {
    int vel_ix = 4;
    matrix[N, N] K[N_latent];
    vector[N] mu[N_latent];
    vector[N] g[P];
    for (k in 1:N_latent){
        K[k] = cov_exp_quad(x, sqrt(var_prior[k]), 1.0) + diag_matrix(rep_vector(1e-5, N));
        mu[k] = rep_vector(mu_prior[k], N);
        f[k] ~ multi_normal(mu[k], K[k]);
    };
    
    g[1] = 2 * f[1] ./ f[1 + vel_ix];
    for (p in 2:P){
        g[p] = 2 * (f[p] - f[p - 1]) ./ f[ p + vel_ix] + g[p-1] ;
    };
    
    for (p in 1:P){
        y[p] ~ normal(g[p], rep_vector(sqrt(sigma2y[p]), N));
    };
}