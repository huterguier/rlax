from dataclasses import dataclass

import jax
import lox
from lox.loggers import MultiLogger

from rlax.algorithms.algorithm import Algorithm, AlgorithmState
from rlax.typing import Key, PyTree


@jax.tree_util.register_dataclass
@dataclass
class TrainerState[TAlgorithmState: AlgorithmState](AlgorithmState):
    algorithm_state: TAlgorithmState
    logger_state: lox.LoggerState


class Trainer[TAlgorithmState: AlgorithmState](
    Algorithm[TrainerState[TAlgorithmState]]
):
    """Splits a training run into epochs, evaluating and logging after each one."""

    algorithm: Algorithm[TAlgorithmState]
    num_epochs: int
    num_steps_eval: int
    num_envs_eval: int
    logger: lox.Logger

    def __init__(
        self,
        algorithm: Algorithm[TAlgorithmState],
        num_epochs: int,
        num_steps_eval: int = 1000,
        num_envs_eval: int = 16,
        logger: lox.Logger | None = None,
    ):
        assert num_epochs >= 1, f"num_epochs ({num_epochs}) must be >= 1"
        self.algorithm = algorithm
        self.num_epochs = num_epochs
        self.num_steps_eval = num_steps_eval
        self.num_envs_eval = num_envs_eval
        self.logger = MultiLogger() if logger is None else logger

    def init(self, key: Key) -> TrainerState[TAlgorithmState]:
        key_algorithm, key_logger = jax.random.split(key)
        algorithm_state = self.algorithm.init(key_algorithm)

        return TrainerState(
            step=algorithm_state.step,
            algorithm_state=algorithm_state,
            logger_state=self.logger.init(key_logger),
        )

    def evaluate(
        self,
        key: Key,
        state: TrainerState[TAlgorithmState],
        num_steps: int,
        num_envs: int,
    ) -> dict[str, PyTree]:
        return self.algorithm.evaluate(key, state.algorithm_state, num_steps, num_envs)

    def train(
        self, key: Key, state: TrainerState[TAlgorithmState], num_steps: int
    ) -> TrainerState[TAlgorithmState]:
        num_steps_epoch = int(num_steps) // self.num_epochs

        def epoch(state, key):
            key_train, key_eval = jax.random.split(key)
            algorithm_state, logs = lox.spool(self.algorithm.train)(
                key_train, state.algorithm_state, num_steps_epoch
            )
            metrics = self.algorithm.evaluate(
                key_eval, algorithm_state, self.num_steps_eval, self.num_envs_eval
            )
            epoch_logs = {"step": algorithm_state.step}
            epoch_logs |= {f"eval/{k}": v for k, v in metrics.items()}

            logger_state = self.logger.log(
                state.logger_state,
                logs | lox.logdict({k: v[None] for k, v in epoch_logs.items()}),
            )
            lox.log(logs | lox.logdict(epoch_logs))

            state = TrainerState(
                step=algorithm_state.step,
                algorithm_state=algorithm_state,
                logger_state=logger_state,
            )
            return state, None

        keys = jax.random.split(key, self.num_epochs)
        state, _ = jax.lax.scan(epoch, state, keys)

        return state
