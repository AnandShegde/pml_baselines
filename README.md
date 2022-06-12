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
View the results here-https://anandshegde.github.io/pml_baselines/results/coin_toss/coin_toss_results.html
![image](https://user-images.githubusercontent.com/79975787/173233408-139781fc-60c9-4bc6-9313-cef9235e7376.png)

# 2. Linear Regression
# 3. Logistic Regression
