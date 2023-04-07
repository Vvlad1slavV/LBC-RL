from typing import Callable

def linear_schedule(initial_value: float, min_value: float = 0.0) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
           min_value: Min learning rate.
    
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return max(progress_remaining * initial_value, min_value)

    return func

def linear_rectified_schedule(initial_value: float, min_value: float = 0.0, *, total_timesteps: int, rectified_timesteps: int) -> Callable[[float], float]:
    """
    Linear learning rate schedule with recified unit.

    :param initial_value: Initial learning rate.
           min_value: Min learning rate.
           total_timesteps: Total timesteps.
           rectified_timesteps: Timesteps after lr will be rectified.
    
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return max(((progress_remaining - 1)*total_timesteps/rectified_timesteps + 1) * initial_value,
                    min_value)

    return func