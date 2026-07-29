from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import flax.linen as nn
import gxm
import jax
import optax

Array: TypeAlias = jax.Array
Key: TypeAlias = jax.Array

PyTree: TypeAlias = Any

Shape: TypeAlias = Sequence[int]
Module: TypeAlias = nn.Module
Params: TypeAlias = Any
Optimizer: TypeAlias = optax.GradientTransformation
OptimizerState: TypeAlias = optax.OptState
Environment: TypeAlias = gxm.Environment
EnvironmentState: TypeAlias = gxm.EnvironmentState

Timestep: TypeAlias = gxm.Timestep
Transition: TypeAlias = gxm.Transition

Action: TypeAlias = PyTree
Observation: TypeAlias = PyTree

PolicyState: TypeAlias = PyTree
Policy: TypeAlias = Callable[[Key, PolicyState, gxm.Timestep], Action]
