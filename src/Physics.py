import numpy as np

from src.Channel import ChannelParams, Channel
from src.State import State
from src.ModelStats import ModelStats
from src.base.GridActions import GridActions
from src.base.GridPhysics import GridPhysics



class PhysicsParams:
    def __init__(self):
        self.channel_params = ChannelParams()
        self.comm_steps = 4


class Physics(GridPhysics):

    def __init__(self, params: PhysicsParams, stats: ModelStats):

        super().__init__()

        self.channel = Channel(params.channel_params)

        self.params = params

        self.register_functions(stats)

    def register_functions(self, stats: ModelStats):
        stats.set_evaluation_value_callback(self.get_cral)

        stats.add_log_data_callback('coverage', self.get_cral)
        stats.add_log_data_callback('date_rate', self.get_service_coverage)
        stats.add_log_data_callback('pdr', self.has_landed)
        stats.add_log_data_callback('delay', self.get_packet_delivery_rate)
        stats.add_log_data_callback('energy_consumption', self.get_landing_attempts)
        stats.add_log_data_callback('overhead', self.get_movement_ratio)

    def reset(self, state: State):
        GridPhysics.reset(self, state)

        self.channel.reset(self.state.shape[0])

    def step(self, action: GridActions):
        old_position = self.state.position
        self.movement_step(action)
        if not self.state.terminal:
            self.comm_step(old_position)

        return self.state

    def comm_step(self, old_position):
        positions = list(
            reversed(np.linspace(self.state.position, old_position, num=self.params.comm_steps, endpoint=False)))

        indices = []
        device_list = self.state.device_list
        for position in positions:
            data_rate, idx = device_list.get_best_data_rate(position, self.channel)
            device_list.collect_data(data_rate, idx)
            indices.append(idx)

        self.state.collected = device_list.get_collected_map(self.state.shape)
        self.state.device_map = device_list.get_data_map(self.state.shape)

        idx = max(set(indices), key=indices.count)
        self.state.set_device_com(idx)

        return idx

    def get_example_action(self):
        return GridActions.HOVER

    def is_in_vulnerable_cell(self):
        return self.state.is_in_vulnerable_cell()

    def get_service_coverage(self):
        return self.state.get_service_coverage()

    def get_movement_budget_used(self):
        return sum(self.state.initial_movement_budgets) - sum(self.state.movement_budgets)

    def get_max_rate(self):
        return self.channel.get_max_rate()

    def get_average_data_rate(self):
        return self.state.get_collected_data() / self.get_movement_budget_used()

    def get_cral(self):
        return self.get_service_coverage() * self.state.all_landed

    def get_packet_delivery_rate(self):
        return self.packet_delivery_rate

    def get_landing_attempts(self):
        return self.landing_attempts

    def get_movement_ratio(self):
        return float(self.get_movement_budget_used()) / float(sum(self.state.initial_movement_budgets))

    def has_landed(self):
        return self.state.all_landed
