# pml_baselines

Compare different baselines for the Baysian Inference problems.

We compare:
* MCMC- Markov Chain Monte Carlo
* ajax - Variational Inference using ajax.
* Laplace Approximation
* PyStan - Variational Inference using Stan


# 1. Coin toss Problem
## Stan

```
coin_toss = """
data {
  int<lower=0> n;
  int<lower=0, upper=1> heads[n];
  real alpha0;
  real beta0;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  heads ~ bernoulli(theta);
  theta ~ beta(alpha0, beta0);
}
"""
```
## Model
![image](https://user-images.githubusercontent.com/79975787/171649924-f9cac98e-8327-4ccb-8f97-0fb96b03f34d.png)

## Results
|||
|-|-|
|Stan|Ajax|
|![image](https://user-images.githubusercontent.com/79975787/171647694-ce1f8fd3-10ca-4e8e-87da-33461ec97711.png)|![VI_ajax_coin_toss](https://user-images.githubusercontent.com/79975787/171648141-f291c986-3180-47d8-9429-2cba003aec91.jpeg)|
|Laplace Approximation| RMH in Blackjax|
|![laplace_coin_toss](https://user-images.githubusercontent.com/79975787/171648134-c6a32162-5fcb-4619-8c64-93bac8f4c6a9.jpeg)|![mcmc_coin_toss](https://user-images.githubusercontent.com/79975787/171648139-3571aab4-54fa-4f4f-9adf-de6418fdf772.jpeg)|

# 2. Linear Regression
# 3. Logistic Regression
