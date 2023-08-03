from src.State import State
from src.base.GridActions import GridActions
from src.base.GridRewards import GridRewards, GridRewardParams


class RewardParams(GridRewardParams):
    def __init__(self):
        super().__init__()
        self.data_multiplier = 1.0


ALPHA = 2
BETA  = 1

class Rewards(GridRewards):
    cumulative_reward: float = 0.0

    def __init__(self, reward_params: RewardParams, stats):
        super().__init__(stats)
        self.params = reward_params
        self.reset()

    def calculate_reward(self, state: State, action: GridActions, next_state: State):
        reward = self.calculate_motion_rewards(state, action, next_state)

        reward += self.params.data_multiplier * (ALPHA(state.get_coverage() * state.get_data_rate())/ (BETA * state.get_energy()))

        # Cumulative reward
        self.cumulative_reward += reward

        return reward
