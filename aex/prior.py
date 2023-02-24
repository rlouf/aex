import aesara
import jax
from aesara.graph.basic import io_toposort
from aesara.tensor.random.op import RandomVariable


def prior_sampler(*rvs):
    """Returns a function that generates prior predictive samples.

    Parameters
    ----------
    rvs
        The `RandomVariable`s for which we want to generate prior samples.

    Returns
    -------
    A function that generates prior samples.

    """
    prior_fn = aesara.function([], list(rvs), mode="JAX").vm.jit_fn
    num_rvs = count_model_rvs(*rvs)

    def one_sample(rng_key):
        keys = [{"jax_state": key} for key in split_key_to_list(rng_key, num_rvs)]
        return prior_fn(*keys)[: len(rvs)]

    def sample(rng_key, num_samples):
        return jax.vmap(one_sample)(jax.random.split(rng_key, num_samples))

    return sample


def count_model_rvs(*rvs):
    """Count the number of `RandomVariable` in a model"""
    return len(
        [
            node
            for node in io_toposort([], list(rvs))
            if isinstance(node.op, RandomVariable)
        ]
    )


def split_key_to_list(rng_key, num):
    keys = jax.numpy.split(jax.random.split(rng_key, num), num)
    return [key.squeeze() for key in keys]
