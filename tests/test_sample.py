import aesara.tensor as at
import jax

import aex


def test_sample():
    num_samples = 1000
    num_warmup = 300
    num_chains = 3

    srng = at.random.RandomStream(0)
    mu_rv = srng.halfnormal(0, 1, name="mu")
    Y_rv = srng.normal(mu_rv, 1, name="y")

    rng_key = jax.random.PRNGKey(0)
    sample = aex.mcmc(Y_rv, num_chains)
    samples, info = sample(rng_key, {Y_rv: 10.0}, num_samples, num_warmup)

    assert "mu" in samples
    assert samples["mu"].shape == (num_chains, num_samples)
