from src.base.GridActions import GridActions


class GridRewardParams:
    def __init__(self):
        self.low_energy = 1.0
        self.collision_1 = 10.0
        self.collision_2 = 10.0
        self.overload = 1.0
        self.loss = 100.0
        self.ceiling = 1
        self.empty_battery_penalty = 3.0
        self.movement_penalty = 0.2


class GridRewards:
    def __init__(self, stats):
        self.params = GridRewardParams()
        self.cumulative_reward: float = 0.0

        stats.add_log_data_callback('cumulative_reward', self.get_cumulative_reward)

    def get_cumulative_reward(self):
        return self.cumulative_reward

    def calculate_motion_rewards(self, state, action: GridActions, next_state):
        reward = 0.0

        # Penalize repeated hovering when the UAV is not serving anything.
        if state.position == next_state.position and not next_state.coverage and action != GridActions.SIR3:
            reward -= self.params.ceiling

        # Penalize depleted battery.
        if hasattr(next_state, "movement_budget") and next_state.movement_budget == 0 and not next_state.coverage:
            reward -= self.params.empty_battery_penalty

        # Small per-move penalty to discourage unnecessary motion.
        reward -= self.params.movement_penalty

        return reward

    def reset(self):
        self.cumulative_reward = 0
