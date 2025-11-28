from src.State import State
from src.base.GridActions import GridActions
from src.base.GridRewards import GridRewards, GridRewardParams


class RewardParams(GridRewardParams):
    def __init__(self):
        super().__init__()
        self.bs_threshold = 5.0
        self.service_threshold = 5.0
        self.qos_threshold = 5.0
        self.lambda_collision_uav = 1.0
        self.lambda_collision_obstacle = 1.0
        self.lambda_bs = 1.0
        self.lambda_service_distance = 1.0
        self.lambda_qos = 1.0
        self.lambda_backhaul = 1.0
        self.eta = 0.5
        self.collision_penalty = 1.0
        self.obstacle_penalty = 1.0
        self.bs_penalty = 1.0
        self.service_penalty = 1.0
        self.qos_penalty = 1.0
        self.backhaul_penalty = 1.0


class Rewards(GridRewards):
    cumulative_reward: float = 0.0

    def __init__(self, reward_params: RewardParams, stats):
        super().__init__(stats)
        self.params = reward_params
        self.reset()

    def calculate_reward(self, state: State, action: GridActions, next_state: State):
        motion_reward = self.calculate_motion_rewards(state, action, next_state)
        penalties = self._compute_weighted_penalties(next_state)
        reward = motion_reward + penalties

        self.cumulative_reward += reward

        return reward

    def _compute_weighted_penalties(self, state: State):
        f1 = self.params.collision_penalty if state.is_occupied() else 0.0
        f2 = self.params.obstacle_penalty if state.is_in_no_fly_zone() else 0.0
        f3 = self.params.bs_penalty if state.get_distance_to_base_station() > self.params.bs_threshold else 0.0
        f4 = self.params.service_penalty if state.get_distance_to_served_vehicle() > self.params.service_threshold else 0.0
        qos_violation = state.get_distance_to_served_vehicle() > self.params.qos_threshold
        f5 = self.params.qos_penalty if qos_violation and state.is_serving() else 0.0
        f6 = self.params.backhaul_penalty if state.get_distance_to_base_station() > self.params.bs_threshold else 0.0

        weighted_sum = (
            self.params.lambda_collision_uav * f1
            + self.params.lambda_collision_obstacle * f2
            + self.params.lambda_bs * f3
            + self.params.lambda_service_distance * f4
            + self.params.lambda_qos * f5
            + self.params.lambda_backhaul * f6
        )

        service_state_penalty = self.params.qos_penalty if qos_violation else 0.0

        return (1 - self.params.eta) * weighted_sum - self.params.eta * service_state_penalty
