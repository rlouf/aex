"""Sample from the posterior distribution.

To sample from the posterior distribution we first need to build the sampling
algorithm that we would like to use in the form:

.. code::

   MCMC(
      {[x_rv, mu_rv]: NUTS, sigma_rv: RMH, eta_rv: EllipticalSlice}
   )

It seems reasonable to assume that any unassigned variable will be sampled
using NUTS. Algorithms like SMC are meta-algorithm:

.. code::

   SMC(
      mcmc({[x_rv, mu_rv]: nuts, sigma_rv: rmh, eta_rv: ellipticalslice}),
      particles=1000,
      resampler=100,
   )

Same for Tempered Stuff.


.. code::

   TemperedSMC(
      mcmc({[x_rv, mu_rv]: nuts, sigma_rv: rmh, eta_rv: ellipticalslice}),
      particles=1000,
      resampler=100,
   )

   ParallelTempering(
      mcmc({[x_rv, mu_rv]: nuts, sigma_rv: rmh, eta_rv: ellipticalslice}),
      particles=1000,
      resampler=100,
   )

When we execute these classes we get a single sampling step (where the values of
the parameters need to be specified). A warmup method is attached to them.

sampler = aex.mcmc({x_rv: RMH}, num_chains=4)
samples, info = sampler(rng_key, 1000)
"""
import aeppl
import aesara
import blackjax
import jax
from aeppl.transforms import TransformValuesRewrite, _default_transformed_rv
from aesara.graph.basic import io_toposort
from aesara.tensor.random.op import RandomVariable

from aex.prior import prior_sampler


def mcmc(observed_rv, num_chains=1):
    algorithm = NUTS()

    # We place the observed RVs at the end
    # We could call `joint_logprob` with the `realized` keyword, but this
    # would not allow the user to manage memory as they wish with JAX.
    to_sample_rvs = tuple(
        node.outputs[1]
        for node in io_toposort([], [observed_rv])
        if isinstance(node.op, RandomVariable)
        if node.outputs[1] != observed_rv
    )
    rvs = to_sample_rvs + (observed_rv,)

    value_variables = tuple(rv.clone() for rv in to_sample_rvs)
    transforms = {rv: get_transform(rv) for rv in to_sample_rvs}

    # Values to unconstrained space
    transformed_values = {
        rv.name: transform_forward(rv, vv, transforms[rv])
        for rv, vv in zip(to_sample_rvs, value_variables)
    }
    forward_transform_fn = aesara.function(
        value_variables, transformed_values, mode="JAX"
    ).vm.jit_fn

    # unconstrained space to original space
    untransformed_values = {
        name: transform_backward(rv, vv, transforms[rv])
        for rv, (name, vv) in zip(to_sample_rvs, transformed_values.items())
    }
    backward_transform_fn = aesara.function(
        list(transformed_values.values()), untransformed_values, mode="JAX"
    ).vm.jit_fn

    # Logdensity function
    logprob, vvs = aeppl.joint_logprob(
        *rvs, extra_rewrites=TransformValuesRewrite(transforms)
    )
    logdensity_jax = aesara.function(vvs, logprob, mode="JAX").vm.jit_fn

    def sample(rng_key, observations, num_samples=1000, num_warmup=1000):
        def logdensity_fn(position):
            flat_position = tuple(position) + tuple(observations.values())
            return logdensity_jax(*flat_position)[0]

        def initialize_state(rng_key):
            init_position = find_init_position(to_sample_rvs, rng_key)
            init_position_tr = forward_transform_fn(*init_position)
            return init_position_tr

        def one_chain(rng_key, state, step_size, inverse_mass_matrix):
            """Sample one chain"""

            def kernel(rng_key, state):
                return algorithm.step(
                    rng_key, state, logdensity_fn, step_size, inverse_mass_matrix
                )

            def one_step(state, rng_key):
                state, info = kernel(rng_key, state)
                return state, (state, info)

            keys = jax.random.split(rng_key, num_samples)
            _, (states, infos) = jax.lax.scan(one_step, state, keys)

            return states, infos

        init_key, warmup_key, sample_key = jax.random.split(rng_key, 3)

        init_keys = jax.random.split(init_key, num_chains)
        init_state = jax.vmap(initialize_state)(init_keys)

        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)

        keys = jax.random.split(warmup_key, num_chains)
        (states, parameters), _ = jax.vmap(warmup.run, in_axes=(0, 0, None))(
            keys, init_state, num_warmup
        )

        keys = jax.random.split(sample_key, num_chains)
        states, infos = jax.vmap(one_chain)(
            keys, states, parameters["step_size"], parameters["inverse_mass_matrix"]
        )

        positions = states.position
        positions = jax.vmap(backward_transform_fn)(*positions)
        positions = {
            rv.name: position.squeeze() for rv, position in zip(rvs, positions)
        }

        return positions, infos

    return sample


def get_transform(rv):
    """Get the default transform associated with the random variable."""
    transform = _default_transformed_rv(rv.owner.op, rv.owner)
    if transform:
        return transform.op.transform
    else:
        return None


def transform_forward(rv, vv, transform):
    """Push variables to the transformed space."""
    if transform:
        res = transform.forward(vv, *rv.owner.inputs)
        if vv.name:
            res.name = f"{vv.name}_trans"
        return res
    else:
        return vv


def transform_backward(rv, vv, transform):
    """Pull variables back from the transformed space."""
    if transform:
        res = transform.backward(vv, *rv.owner.inputs)
        return res
    else:
        return vv


def find_init_position(to_sample_rvs, rng_key):
    samples = prior_sampler(*to_sample_rvs)(rng_key, 1)
    if not isinstance(samples, tuple):
        return (samples,)
    return samples


class NUTS(object):
    def __init__(self):
        self.step = blackjax.mcmc.nuts.kernel()
        self.init = blackjax.mcmc.nuts.init
