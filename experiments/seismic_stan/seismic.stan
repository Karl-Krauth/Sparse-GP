data {
    int<lower=1> N;
    int<lower=1> D;
    int<lower=1> N_latent;
    real x[N];
    vector[N] y[D];
    vector[N_latent] mu_prior;
    vector[N_latent] var_prior;
}

transformed data {
}
parameters {
    vector[N] f[N_latent];
}

model {
    matrix[N, N] K[N_latent];
    vector[N] mu[N_latent];
    for (k in 1:N_latent){
        K[k] = cov_exp_quad(x, var_prior[k], 1.0) + diag_matrix(rep_vector(1e-5, N));
        mu[k] = rep_vector(mu_prior[k], N);
        f[k] ~ multi_normal(mu[k], K[k]);
    };
    
    for (k in 1:4){
        y[k] ~ normal(f[k], rep_vector(0.01, N));
    };
}
