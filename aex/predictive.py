import aesara
from aesara.graph.basic import io_toposort
from aesara.tensor.random.op import RandomVariable
import jax


def posterior_predictive_sampler(samples, rv):
    """Returns a function that generates posterior predictive samples.

    Parameters
    ----------
    samples
        A dictionary that maps random variable names to their posterior samples.
    rv
        The `RandomVariable` for which we want posterior predictive samples.

    Returns
    -------
    A function that generates posterior predictive samples.
    """

    def compile(inputs):
        posterior_predictive_fn = aesara.function(list(inputs), rv, mode="JAX").vm.jit_fn
        num_rvs = count_model_rvs(rv)

        def sample_fn(rng_key, values=[]):
            rng_key, choice_key = jax.random.split(rng_key)
            sample_values = list(samples.values())
            idx = jax.random.choice(rng_key, sample_values[0].shape[0])
            sample_value = [samples[idx] for samples in sample_values]
            keys = [{"jax_state": key} for key in split_key_to_list(rng_key, num_rvs)]
            return posterior_predictive_fn(*values, *sample_value, *keys)[0]

        return sample_fn

    def sample(rng_key, num_samples, inputs={}):
        sample_fn = compile(inputs.keys())
        in_axes = (0, None)
        if len(inputs) > 0:
            in_axes += (None,) * (len(inputs) - 1)

        samples = jax.vmap(sample_fn, in_axes=in_axes)(
            jax.random.split(rng_key, num_samples), list(inputs.values())
        )

        samples = tuple([s.squeeze() for s in samples])

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
