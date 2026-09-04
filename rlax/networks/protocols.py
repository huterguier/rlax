from typing import Protocol

import distrax

from rlax.typing import Array, Key, Observation, Params


class ActorCriticNetwork(Protocol):
    """What :class:`~rlax.PPO` expects of its network.

    ``apply`` maps a batch of observations to ``(dist, value)``: a distrax
    distribution with batch shape ``(batch,)`` over actions, and a value
    estimate of shape ``(batch,)``. Any flax module returning that pair
    qualifies; :class:`~rlax.networks.ActorCritic` is the stock one.
    """

    def init(self, key: Key, obs: Observation) -> Params: ...

    def apply(
        self, params: Params, obs: Observation
    ) -> tuple[distrax.Distribution, Array]: ...


class QNetwork(Protocol):
    """What :class:`~rlax.PQN` expects of its network.

    ``apply`` maps a batch of observations to Q-values of shape
    ``(batch, num_actions)``.
    """

    def init(self, key: Key, obs: Observation) -> Params: ...

    def apply(self, params: Params, obs: Observation) -> Array: ...
