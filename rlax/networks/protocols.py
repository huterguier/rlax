from typing import Protocol

import distrax

from rlax.typing import Array, Key, Observation, Params


class ActorNetwork(Protocol):
    """What :class:`~rlax.PPO` expects of its actor.

    ``apply`` maps a batch of observations to a distrax distribution with
    batch shape ``(batch,)`` over actions. :class:`~rlax.networks.Actor` is
    the stock one.
    """

    def init(self, key: Key, obs: Observation) -> Params: ...

    def apply(self, params: Params, obs: Observation) -> distrax.Distribution: ...


class CriticNetwork(Protocol):
    """What :class:`~rlax.PPO` expects of its critic.

    ``apply`` maps a batch of observations to values of shape ``(batch,)``.
    :class:`~rlax.networks.Critic` is the stock one.
    """

    def init(self, key: Key, obs: Observation) -> Params: ...

    def apply(self, params: Params, obs: Observation) -> Array: ...


class QNetwork(Protocol):
    """What :class:`~rlax.PQN` expects of its network.

    ``apply`` maps a batch of observations to Q-values of shape
    ``(batch, num_actions)``.
    """

    def init(self, key: Key, obs: Observation) -> Params: ...

    def apply(self, params: Params, obs: Observation) -> Array: ...
