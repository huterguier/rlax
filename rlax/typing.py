from collections.abc import Callable, Sequence
from typing import Any

import flax.linen as nn
import gxm
import jax
import optax

Array = jax.Array
Key = jax.Array

PyTree = Any

Shape = Sequence[int]
Module = nn.Module
Params = Any
Optimizer = optax.GradientTransformation
OptimizerState = optax.OptState
Environment = gxm.Environment
EnvironmentState = gxm.EnvironmentState

Timestep = gxm.Timestep
Transition = gxm.Transition

Action = PyTree
Observation = PyTree

PolicyState = PyTree
Policy = Callable[[Key, PolicyState, gxm.Timestep], Action]
