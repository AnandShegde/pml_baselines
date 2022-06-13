import stan
import distrax
import jax
import numpy as np

coin_toss = """
 data{
     int<lower=0>N; //no of data points
     int data_points[N]; //data points
 }
 parameters{
     real<lower=0,upper =1>theta;
 }
 model{
     target += beta_lpdf( theta | 60.0,40.0);
     target += bernoulli_lpmf(data_points | theta);
 }
"""

key = jax.random.PRNGKey(100)
samples = distrax.Bernoulli(0.6).sample(seed = key,sample_shape = 100)
samples = list(samples)
data = {
        "N" : 100,
        "data_points" : samples
       }

posterior = stan.build(coin_toss, data=data, random_seed=1)
fit = posterior.sample(num_chains=4, num_samples=1000)
theta = fit["theta"]
print(theta)