import numpy as np
from src.Map.Map import Map
from src.StateUtils import pad_centered
from src.base.BaseState import BaseState


class State(BaseState):
    #def __init__(self, map_init: Map, num_agents: int, multi_agent: bool):
    def __init__(self, map_init: Map, num_agents: int, multi_agent: bool, base_station_position=(0, 0, 0)):
        super().__init__(map_init)
        self.device_list = None
        self.device_map = None  # Floating point sparse matrix showing devices and their data to be collected

        self.active_agent = 0
        self.num_agents = num_agents
        self.multi_agent = multi_agent


        #self.positions = [[0, 0]] * num_agents
        self.positions = [self._normalize_position([0, 0, 0])] * num_agents
        self.movement_budgets = [0] * num_agents
        self.landeds = [False] * num_agents
        self.terminals = [False] * num_agents
        self.device_coms = [-1] * num_agents
        self.base_station_position = np.array(base_station_position, dtype=float)
        self.transmit_power = 0.0
        self.initial_movement_budgets = [0] * num_agents
        self.initial_total_data = 0
        self.collected = None

    @property
    def position(self):
        return self.positions[self.active_agent]

    @property
    def movement_budget(self):
        return self.movement_budgets[self.active_agent]

    @property
    def initial_movement_budget(self):
        return self.initial_movement_budgets[self.active_agent]

    @property
    def landed(self):
        return self.landeds[self.active_agent]

    @property
    def terminal(self):
        return self.terminals[self.active_agent]

    @property
    def all_landed(self):
        return all(self.landeds)

    @property
    def all_terminal(self):
        return all(self.terminals)

    def is_terminal(self):
        return self.all_terminal

    def set_landed(self, landed):
        self.landeds[self.active_agent] = landed

    def set_position(self, position):
        #self.positions[self.active_agent] = position
        self.positions[self.active_agent] = self._normalize_position(position)

    def decrement_movement_budget(self):
        self.movement_budgets[self.active_agent] -= 1

    def set_terminal(self, terminal):
        self.terminals[self.active_agent] = terminal

    def set_device_com(self, device_com):
        self.device_coms[self.active_agent] = device_com

    def get_active_agent(self):
        return self.active_agent

    def get_remaining_data(self):
        return np.sum(self.device_map)

    def get_total_data(self):
        return self.initial_total_data

    def get_scalars(self, give_position=False):
        """
        Return the scalars without position, as it is treated individually
        """
        if give_position:
            return np.array([self.movement_budget, self.position[0], self.position[1]])

        return np.array([self.movement_budget])

    def get_num_scalars(self, give_position=False):
        return len(self.get_scalars(give_position))

    def get_boolean_map(self):
        padded_red = pad_centered(self, np.concatenate([np.expand_dims(self.no_fly_zone, -1),
                                                        np.expand_dims(self.obstacles, -1)], axis=-1), 1)
        if self.multi_agent:
            padded_rest = pad_centered(self,
                                       np.concatenate(
                                           [np.expand_dims(self.vulnerable_cell, -1), self.get_agent_bool_maps()],
                                           axis=-1), 0)
        else:
            padded_rest = pad_centered(self, np.expand_dims(self.vulnerable_cell, -1), 0)
        return np.concatenate([padded_red, padded_rest], axis=-1)

    def get_boolean_map_shape(self):
        return self.get_boolean_map().shape

    def get_float_map(self):
        if self.multi_agent:
            return pad_centered(self, np.concatenate([np.expand_dims(self.device_map, -1),
                                                      self.get_agent_float_maps()], axis=-1), 0)
        else:
            return pad_centered(self, np.expand_dims(self.device_map, -1), 0)

    def get_float_map_shape(self):
        return self.get_float_map().shape

    def is_in_vulnerable_cell(self):
        return self.vulnerable_cell[self.position[1]][self.position[0]]

    def is_in_no_fly_zone(self):
        # Out of bounds is implicitly nfz
        if 0 <= self.position[1] < self.no_fly_zone.shape[0] and 0 <= self.position[0] < self.no_fly_zone.shape[1]:
            # NFZ or occupied
            return self.no_fly_zone[self.position[1], self.position[0]] or self.is_occupied()
        return True

    def is_occupied(self):
        if not self.multi_agent:
            return False
        for i, pos in enumerate(self.positions):
            if self.terminals[i]:
                continue
            if i == self.active_agent:
                continue
            if pos == self.position:
                return True
        return False

    def get_service_coverage(self):
        return np.sum(self.collected) / self.initial_total_data

    def get_collected_data(self):
        return np.sum(self.collected)

    def reset_devices(self, device_list):
        self.device_map = device_list.get_data_map(self.no_fly_zone.shape)
        self.collected = np.zeros(self.no_fly_zone.shape, dtype=float)
        self.initial_total_data = device_list.get_total_data()
        self.device_list = device_list

    def get_agent_bool_maps(self):
        agent_map = np.zeros(self.no_fly_zone.shape + (1,), dtype=bool)
        for agent in range(self.num_agents):
            # agent_map[self.positions[agent][1], self.positions[agent][0]][0] = self.landeds[agent]
            agent_map[self.positions[agent][1], self.positions[agent][0]][0] = not self.terminals[agent]
        return agent_map

    def get_agent_float_maps(self):
        agent_map = np.zeros(self.no_fly_zone.shape + (1,), dtype=float)
        for agent in range(self.num_agents):
            agent_map[self.positions[agent][1], self.positions[agent][0]][0] = self.movement_budgets[agent]
        return agent_map

    def get_device_scalars(self, max_num_devices, relative):
        devices = np.zeros(3 * max_num_devices, dtype=np.float32)
        if relative:
            for k, dev in enumerate(self.device_list.devices):
                devices[k * 3] = dev.position[0] - self.position[0]
                devices[k * 3 + 1] = dev.position[1] - self.position[1]
                devices[k * 3 + 2] = dev.data - dev.collected_data
        else:
            for k, dev in enumerate(self.device_list.devices):
                devices[k * 3] = dev.position[0]
                devices[k * 3 + 1] = dev.position[1]
                devices[k * 3 + 2] = dev.data - dev.collected_data
        return devices

    def get_uav_scalars(self, max_num_uavs, relative):
        uavs = np.zeros(4 * max_num_uavs, dtype=np.float32)
        if relative:
            for k in range(max_num_uavs):
                if k >= self.num_agents:
                    break
                uavs[k * 4] = self.positions[k][0] - self.position[0]
                uavs[k * 4 + 1] = self.positions[k][1] - self.position[1]
                uavs[k * 4 + 2] = self.movement_budgets[k]
                uavs[k * 4 + 3] = not self.terminals[k]
        else:
            for k in range(max_num_uavs):
                if k >= self.num_agents:
                    break
                uavs[k * 4] = self.positions[k][0]
                uavs[k * 4 + 1] = self.positions[k][1]
                uavs[k * 4 + 2] = self.movement_budgets[k]
                uavs[k * 4 + 3] = not self.terminals[k]
        return uavs
    def get_distance_to_base_station(self):
        pos = np.array(self.position, dtype=float)
        bs = self.base_station_position
        if bs.shape[0] == 2:
            bs = np.append(bs, 0)
        return float(np.linalg.norm(pos - bs))

    def get_distance_to_served_vehicle(self):
        if not self.device_list or self.device_coms[self.active_agent] == -1:
            return float('inf')
        target_device = self.device_list.get_device(self.device_coms[self.active_agent])
        device_pos = np.array(self._normalize_position(target_device.position), dtype=float)
        return float(np.linalg.norm(np.array(self.position, dtype=float) - device_pos))

    def is_serving(self):
        return self.device_coms[self.active_agent] != -1

    @property
    def coverage(self):
        return self.is_serving()

    def get_connectivity_indicator(self, bs_threshold):
        return float(self.get_distance_to_base_station() <= bs_threshold)

    def get_service_coverage_flag(self):
        return float(self.is_serving())

    def get_vehicle_locations(self):
        if not self.device_list:
            return []
        return [self._normalize_position(device.position) for device in self.device_list.devices]

    def get_state_space(self, bs_threshold, service_threshold):
        """
        Construct the full state space vector (Definition 6) with the current UAV as reference.
        """
        return {
            's1': np.array(self.position, dtype=float),
            's2': float(self.movement_budget),
            's3': {
                'uav_collision': float(self.is_occupied()),
                'obstacle_collision': float(self.is_in_no_fly_zone())
            },
            's4': self.get_distance_to_base_station(),
            's5': float(self.transmit_power),
            's6': self.get_connectivity_indicator(bs_threshold),
            's7': self.get_distance_to_served_vehicle(),
            's8': self.get_service_coverage_flag(),
            's9': self.get_vehicle_locations(),
            'service_threshold': service_threshold,
        }

