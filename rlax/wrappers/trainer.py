import jax
import lox

from rlax.algorithms.algorithm import Algorithm, AlgorithmState
from rlax.typing import Key, PyTree


class Trainer[TAlgorithmState: AlgorithmState](Algorithm[TAlgorithmState]):
    """Splits a training run into epochs and evaluates after each one.

    An epoch is a chunk of ``num_steps // num_epochs`` environment steps, not to be
    confused with the update epochs of PPO and PQN, which are passes over a batch of
    collected transitions. Every epoch runs the wrapped algorithm's ``train`` followed
    by its ``evaluate``, and logs the resulting metrics under an ``eval/`` prefix.
    The whole loop is a single scan, so the evaluation curve is obtained by spooling::

        trainer = Trainer(algorithm, num_epochs=10)
        state, logs = lox.spool(trainer.train)(key, state, num_steps=int(1e7))
        logs["eval/return"]  # shape (10,)

    The wrapper is itself an ``Algorithm``, delegating ``init`` and ``evaluate`` to the
    algorithm it wraps.
    """

    algorithm: Algorithm[TAlgorithmState]
    num_epochs: int
    num_steps_eval: int
    num_envs_eval: int

    def __init__(
        self,
        algorithm: Algorithm[TAlgorithmState],
        num_epochs: int,
        num_steps_eval: int = 1000,
        num_envs_eval: int = 16,
    ):
        assert num_epochs >= 1, f"num_epochs ({num_epochs}) must be >= 1"
        self.algorithm = algorithm
        self.num_epochs = num_epochs
        self.num_steps_eval = num_steps_eval
        self.num_envs_eval = num_envs_eval

    def init(self, key: Key) -> TAlgorithmState:
        return self.algorithm.init(key)

    def evaluate(
        self, key: Key, state: TAlgorithmState, num_steps: int, num_envs: int
    ) -> dict[str, PyTree]:
        return self.algorithm.evaluate(key, state, num_steps, num_envs)

    def train(
        self, key: Key, state: TAlgorithmState, num_steps: int
    ) -> TAlgorithmState:
        """Train for ``num_steps`` steps, split into ``num_epochs`` equal epochs.

        Steps that do not divide evenly into epochs are dropped, as are steps that do
        not fill a whole update loop of the wrapped algorithm.
        """
        num_steps_epoch = int(num_steps) // self.num_epochs

        def epoch(state, key):
            key_train, key_eval = jax.random.split(key)
            state = self.algorithm.train(key_train, state, num_steps_epoch)
            metrics = self.algorithm.evaluate(
                key_eval, state, self.num_steps_eval, self.num_envs_eval
            )
            lox.log({f"eval/{k}": v for k, v in metrics.items()})
            return state, None

        keys = jax.random.split(key, self.num_epochs)
        state, _ = jax.lax.scan(epoch, state, keys)

        return state
