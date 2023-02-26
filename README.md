# AeX

## What is Aex?

AeX allows to sample from probabilistic models defined with [Aesara](https://github.com/aesara-devs/aesara) using JAX & [Blackjax](https://github.com/blackjax-devs/blackjax).

## Why Aex?

[Aesara](https://github.com/aesara-devs/aesara) is a very expressive and flexible modeling language. [Blackjax](https://github.com/blackjax-devs/blackjax) is fast and modular, but is too low-level for most. AeX brings them together with a no-nonsense wrapper.

## Work in progress

### Sample from the prior and the posterior

Prior predictive sampling currently works:

``` python
import aesara.tensor as at
import aex
import jax

srng = at.random.RandomStream(0)

mu_tt = at.scalar("mu")
sigma_rv = srng.normal(mu_tt)
mu_rv = srng.normal(0, 1)
Y_rv = srng.normal(mu_rv, sigma_rv)

rng_key = jax.random.PRNGKey(0)
sampler = aex.prior_sampler(Y_rv, mu_rv)
sampler(rng_key, 1_000_000, {mu_tt: 1.})
```

Sampling from the posterior distribution using Blackjax's NUTS sampler and the window adaptation:

``` python
import aesara.tensor as at
import aex
import jax

srng = at.random.RandomStream(0)

mu_rv = srng.halfnormal(0, 1, name="mu")
Y_rv = srng.normal(mu_rv, 1, name="y")

rng_key = jax.random.PRNGKey(0)
sampler = aex.mcmc(Y_rv, num_chains)
samples, info = sampler(rng_key, {Y_rv: 10.0}, num_samples, num_warmup)
```

### Coming soon

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

## Contribute

AeX is a thin wrapper around Aesara/AePPL and Blackjax, and we want to re-direct contributions to these libraries every time we can.

**There's a bug I don't understand**

Open an issue on this repo.

**I need a distribution that's not available**

The best place to contribute is [Aesara](https://github.com/aesara-devs/aesara) or [AePPL](https://github.com/aesara-devs/aeppl)

**I want to be able to sample from X model**

AePPL's aim is to support any model that's mathematically defined. The best place to contribute is [AePPL](https://github.com/aesara-devs/aeppl).

**My favourite sampler is not available**

First check if the sampler exists in [Blackjax](https://github.com/blackjax-devs/blackjax). If it does, open an issue on AeX. Otherwise, start a discussion on Blackjax and open an issue on AeX.
