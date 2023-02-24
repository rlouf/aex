import aesara.tensor as at
import jax
import numpy as np

import aex


def test_count_rvs():
    srng = at.random.RandomStream(0)
    x_rv = srng.normal(0, 1)

    assert aex.prior.count_model_rvs(x_rv) == 1

    srng = at.random.RandomStream(0)
    x_rv = srng.normal(0, 1)
    y_rv = srng.normal(x_rv, 1)

    assert aex.prior.count_model_rvs(y_rv) == 2
    assert aex.prior.count_model_rvs(x_rv, y_rv) == 2


def test_split_key_to_list():
    rng_key = jax.random.PRNGKey(0)
    keys = aex.prior.split_key_to_list(rng_key, 10)
    assert len(keys) == 10
    assert keys[0].shape == (2,)


def test_prior_sampler():
    srng = at.random.RandomStream(0)
    x_rv = srng.normal(0, 1)

    rng_key = jax.random.PRNGKey(0)
    samples = aex.prior_sampler(x_rv)(rng_key, 10)
    assert len(np.unique(samples)) == 10

    srng = at.random.RandomStream(0)
    mu_rv = srng.normal(0, 1)
    sigma_rv = srng.normal(0, 1)
    Y_rv = srng.normal(mu_rv, sigma_rv)

    rng_key = jax.random.PRNGKey(0)
    samples = aex.prior_sampler(mu_rv, Y_rv)(rng_key, 10)
    assert len(samples) == 2
    assert len(np.unique(samples[0])) == 10
