from collections.abc import Callable, Sequence

from rlax.agent import Agent
from rlax.agents.ppo import PPOAgent, PPOConfig, PPOState
from rlax.algorithms.on_policy import OnPolicy
from rlax.typing import Environment, Optimizer

__all__ = ["PPO", "PPOConfig", "PPOState"]


class PPO(OnPolicy):
    """Proximal Policy Optimization: :class:`~rlax.OnPolicy` around a
    :class:`~rlax.PPOAgent`. ``wrappers`` are applied to the agent in order,
    e.g. ``wrappers=(Normalize,)``."""

    def __init__(
        self,
        config: PPOConfig,
        env: Environment,
        actor,
        critic,
        optimizer: Optimizer,
        wrappers: Sequence[Callable[[Agent], Agent]] = (),
    ):
        agent: Agent = PPOAgent(config, actor, critic, optimizer)
        for wrap in wrappers:
            agent = wrap(agent)
        super().__init__(agent, env, config)
