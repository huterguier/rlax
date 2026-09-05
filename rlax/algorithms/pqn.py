from collections.abc import Callable, Sequence

from rlax.agent import Agent
from rlax.agents.pqn import PQNAgent, PQNConfig, PQNState
from rlax.algorithms.on_policy import OnPolicy
from rlax.typing import Environment, Optimizer

__all__ = ["PQN", "PQNConfig", "PQNState"]


class PQN(OnPolicy):
    """Parallelised Q-Network: :class:`~rlax.OnPolicy` around a
    :class:`~rlax.PQNAgent`. ``wrappers`` are applied to the agent in order."""

    def __init__(
        self,
        config: PQNConfig,
        env: Environment,
        network,
        optimizer: Optimizer,
        wrappers: Sequence[Callable[[Agent], Agent]] = (),
    ):
        agent: Agent = PQNAgent(config, network, optimizer)
        for wrap in wrappers:
            agent = wrap(agent)
        super().__init__(agent, env, config)
