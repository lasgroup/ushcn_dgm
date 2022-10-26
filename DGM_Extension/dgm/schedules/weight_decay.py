from enum import Enum, auto
from typing import Callable
import jax.numpy as jnp

from jax.experimental.optimizers import constant, piecewise_constant
from dgm.schedules.betas import polynomial_decay, beta_transition_between_values

Schedule = Callable[[int], float]


class WeightDecayType(Enum):
    PIECEWISE_CONSTANT = auto()
    CONSTANT = auto()
    POLYNOMIAL_DECAY = auto()
    TRANSITION_BETWEEN_VALUESE = auto()


def get_weight_decay(wd_type: WeightDecayType, kwargs: dict) -> Schedule:
    if wd_type == WeightDecayType.PIECEWISE_CONSTANT:
        return piecewise_constant(**kwargs)
    elif wd_type == WeightDecayType.CONSTANT:
        return constant(**kwargs)
    elif wd_type == WeightDecayType.POLYNOMIAL_DECAY:
        return polynomial_decay(**kwargs)
    elif wd_type == WeightDecayType.TRANSITION_BETWEEN_VALUESE:
        return beta_transition_between_values(**kwargs)
    raise NotImplementedError(f"weight_decay_type {wd_type} has not been implemented yet.")


def indicator(x):
    return (jnp.sign(x) + 1) / 2


def beta_transition_between_values(transition_start, step_size, decay_steps, final_step_size,
                                   power=1.0):
    initial_beta = constant(step_size)
    later_beta = polynomial_decay(step_size, decay_steps, final_step_size, power=power)

    def beta_schedule(i):
        return indicator(transition_start - i) * initial_beta(i) + indicator(
            i - transition_start) * later_beta(i - transition_start)

    return beta_schedule
