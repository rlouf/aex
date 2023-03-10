import aesara.tensor as at
import jax
import jax.numpy as jnp

import aex


def test_posterior_sampler():
    srng = at.random.RandomStream(0)
    mu_rv = srng.halfnormal(0, 1, name="mu")
    Y_rv = srng.normal(mu_rv, 1, name="y")

    samples = {mu_rv: jnp.array([1., 0.9, 0.5])}

    rng_key = jax.random.PRNGKey(0)
    posterior_sampler = aex.posterior_predictive_sampler(samples, Y_rv)
    samples = posterior_sampler(rng_key, 100)
