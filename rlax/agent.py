from abc import abstractmethod
from typing import Protocol

from rlax.typing import Action, Key, PyTree, Timestep, Transition


class Agent[TState, TCarry](Protocol):
    """A learning rule, decoupled from environment stepping.

    An agent never touches an environment. It answers two questions: given a
    batch of observations, which actions to take (:meth:`act`), and given a
    batch of experience, how to improve (:meth:`update`). An
    :class:`~rlax.Algorithm` such as :class:`~rlax.OnPolicy` owns the
    environment and feeds the agent.

    State is split in two:

    - ``state`` is shared across environments: parameters, optimizer state,
      running statistics. It threads linearly through :meth:`act` and
      :meth:`update`, so both always see the same parameters.
    - ``carry`` is per environment, with a leading batch axis: a recurrent
      hidden state, for example. Its batch size is decided by whoever runs the
      agent, so it is created with :meth:`initialize_carry` rather than
      :meth:`init`. Stateless agents use ``None``.

    Observations passed to :meth:`act` are batched, ``(batch, ...)``, and
    :meth:`update` receives ``(time, batch, ...)`` arrays.
    """

    @abstractmethod
    def init(self, key: Key, timestep: Timestep) -> TState:
        """Create the shared state. ``timestep`` is a batched sample from the
        environment, used for shapes only."""
        ...

    @abstractmethod
    def initialize_carry(self, key: Key, state: TState, timestep: Timestep) -> TCarry:
        """Create the per-environment carry for the batch in ``timestep``."""
        ...

    @abstractmethod
    def act(
        self,
        key: Key,
        state: TState,
        carry: TCarry,
        timestep: Timestep,
        evaluation: bool = False,
    ) -> tuple[Action, TState, TCarry, PyTree]:
        """Choose actions for ``timestep.next_obs``.

        Returns the actions, the updated state and carry, and ``aux``: whatever
        per-step quantities :meth:`update` will need (log-probabilities,
        values, Q-values, ...). ``aux`` is ``None`` when ``evaluation`` is set.
        ``evaluation`` is a static Python bool.
        """
        ...

    @abstractmethod
    def update(
        self, key: Key, state: TState, transitions: Transition, aux: PyTree
    ) -> TState:
        """Learn from a ``(time, batch)`` batch of transitions and the ``aux``
        stacked from the :meth:`act` calls that produced them. ``aux`` may have
        been computed with parameters older than ``state``'s."""
        ...


class AgentBase[TState](Agent[TState, None]):
    """Base for agents without per-environment state."""

    def initialize_carry(self, key: Key, state: TState, timestep: Timestep) -> None:
        del key, state, timestep
        return None
