import aesara
import jax
from aesara.graph.basic import io_toposort
from aesara.tensor.random.op import RandomVariable


def prior_sampler(*rvs):
    """Returns a function that generates prior predictive samples.

    TODO: We can pre-compile the sampling function given the graph by finding
    the tensor variables whose value needs to be provided. Given that JAX caches
    JITTED function I am not sure this would make a difference, but it is nicer
    than lazily relying on the cache.

    Parameters
    ----------
    rvs
        The `RandomVariable`s for which we want to generate prior samples.

    Returns
    -------
    A function that generates prior samples.

    """

    def compile(inputs):
        prior_fn = aesara.function(list(inputs), list(rvs), mode="JAX").vm.jit_fn
        num_rvs = count_model_rvs(*rvs)

        def sample_fn(rng_key, values=[]):
            keys = [{"jax_state": key} for key in split_key_to_list(rng_key, num_rvs)]
            return prior_fn(*values, *keys)[: len(rvs)]

        return sample_fn

    def sample(rng_key, num_samples, inputs={}):
        sample_fn = compile(inputs.keys())
        in_axes = (0, None)
        if len(inputs) > 0:
            in_axes += (None,) * (len(inputs) - 1)

        samples = jax.vmap(sample_fn, in_axes=in_axes)(
            jax.random.split(rng_key, num_samples), list(inputs.values())
        )
        if len(samples) == 1:
            return samples[0]

        return samples

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
