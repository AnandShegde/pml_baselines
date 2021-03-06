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
<!-- ![image](https://user-images.githubusercontent.com/79975787/173255249-357a40f8-966a-48d3-8446-f90266b981b0.png)
 -->
<!--  ![coin toss results](https://user-images.githubusercontent.com/79975787/175093288-25f1fd87-4cfb-49d8-aa86-1d594de123f4.png)
 -->
 ![coin_toss_fig](https://user-images.githubusercontent.com/54628243/175222959-ce6f6a26-ab63-4e10-9f37-3c8d9003f689.png)

 
# 2. Linear Regression
View the results here-https://anandshegde.github.io/pml_baselines/results/linear_regression/linear_regression_results.html
<!-- ![image](https://user-images.githubusercontent.com/79975787/173254412-4c99a5e4-4006-43a2-91b1-8ac351670543.png)
 -->
![linear_regression_results](https://user-images.githubusercontent.com/54628243/175223487-ba392dab-e4ae-4fcd-acde-5aabf6b86658.png)

# 3. Logistic Regression
View the results here-https://anandshegde.github.io/pml_baselines/results/logistic_regression/logistic_regression_results.html
<!-- ![image](https://user-images.githubusercontent.com/79975787/173255290-4ad7edf1-990b-4263-a799-d68266ccad9d.png)
 -->
 ![result_logistic_regression](https://user-images.githubusercontent.com/54628243/175223100-44588927-e775-4ed2-8ca9-e44f0c63e05b.png)

# 4. Weibull Poisson
View the results here-https://anandshegde.github.io/pml_baselines/results/weibull_poisson/weibull_poisson_results.html 
<!-- ![weibull_poisson](https://user-images.githubusercontent.com/54628243/173325147-914e1672-c73f-4e35-bf77-d8ca1a1f22cd.png)
 -->
 ![weibull_poisson](https://user-images.githubusercontent.com/54628243/175223260-01fbc919-c022-44a8-a423-dd31d7fa817d.png)

