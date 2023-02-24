# AeX

The following currently works:

``` python
import aesara.tensor as at
import aex

srng = at.random.RandomStream(0)

sigma_rv = srng.normal(1.)
mu_rv = srng.normal(0, 1)
Y_rv = srng.normal(mu_rv, sigma_rv)

sampler = aex.prior_sampler(Y_rv, mu_rv)
sampler(rng_key, 1_000_000)
```

## Coming

Sampling from the posterior distribution using Blackjax's NUTS sampler:

``` python
sampler = aex.mcmc({Y_rv: 1.}, aex.NUTS())
samples, info = sampler(rng_key, 1000, 1000)
```

Sampling from the posterior by arbitrarily combining Blackjax step functions:

``` python
sampler = aex.mcmc({Y_rv: 1.}, {[mu_rv, sigma_rv]: aex.NUTS(), Y_rv: aex.RMH()})
samples, info = sampler(rng_key, 1000)
```

Sampling from the posterior predictive distribution:

``` python
sampler = aex.posterior_predictive(trace, Y_rv)
sampler(rng_key, 1000)
```
