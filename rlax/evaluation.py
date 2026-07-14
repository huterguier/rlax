import jax
import jax.numpy as jnp
import tqdx
from gxm.core import Environment
from gxm.typing import Key, Policy, PolicyState
from gxm.wrappers import EpisodeCounter, Evaluate


def evaluate_steps(
    key: Key,
    env: Environment,
    policy: Policy,
    policy_state: PolicyState,
    num_steps: int,
    num_envs: int = 1,
):
    """
    Evaluate a policy in the given environment over a specified number of steps.

    Args:
        key: A JAX random key for any stochasticity in the environment and policy.
        env: The environment to perform the evaluation in.
        policy: The policy to evaluate.
        policy_state: The initial state of the policy.
        num_steps: The number of steps to perform in the rollout.
        num_envs: The number of parallel environments to use for evaluation.
    Returns:
        The mean return of the policy over the rollout.
    """
    env = Evaluate(env.unwrapped)

    def step(carry, _):
        key, env_state, policy_state, timestep = carry
        key, key_policy, key_step = jax.random.split(key, 3)
        action, policy_state = policy(key_policy, policy_state, timestep)
        env_state, timestep = env.step(key_step, env_state, action)
        carry = (key, env_state, policy_state, timestep)
        return carry, _

    def _evaluate(key):
        env_state, timestep = env.init(key)
        carry = (key, env_state, policy_state, timestep)
        carry, _ = tqdx.scan(step, carry, None, length=num_steps)
        _, env_state, _, _ = carry
        return env_state.mean_return

    keys = jax.random.split(key, num_envs)
    returns = jax.vmap(_evaluate)(keys)
    mean_return = jnp.mean(returns)
    return mean_return


def evaluate_episodes(
    key: Key,
    env: Environment,
    policy: Policy,
    policy_state: PolicyState,
    num_episodes: int,
    num_envs: int = 1,
):
    """
    Evaluate a policy in the given environment over a specified number of episodes.

    Args:
        key: A JAX random key for any stochasticity in the environment and policy.
        env: The environment to perform the evaluation in.
        policy: The policy to evaluate.
        policy_state: The initial state of the policy.
        num_episodes: The number of episodes to perform the rollout.
        num_envs: The number of parallel environments to use for evaluation.
    Returns:
        The mean return of the policy over the episodes.
    """
    env = EpisodeCounter(env)
    env = Evaluate(env)

    def cond(carry):
        _, _, _, timestep = carry
        return timestep.info["n_episodes"] < num_episodes

    def body(carry):
        key, env_state, policy_state, timestep = carry
        key, key_policy, key_step = jax.random.split(key, 3)
        action, policy_state = policy(key_policy, policy_state, timestep)
        env_state, timestep = env.step(key_step, env_state, action)
        carry = (key, env_state, policy_state, timestep)
        return carry

    def _evaluate(key):
        env_state, timestep = env.init(key)
        carry = (key, env_state, policy_state, timestep)
        carry = jax.lax.while_loop(cond, body, carry)
        _, env_state, _, _ = carry
        return env_state.mean_return

    keys = jax.random.split(key, num_envs)
    returns = jax.vmap(_evaluate)(keys)
    return jnp.mean(returns)
