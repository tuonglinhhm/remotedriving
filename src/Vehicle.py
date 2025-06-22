import numpy as np

from src.Channel import Channel


class VehicleParams:
    def __init__(self, position=(0, 0), color='blue', data=15.0):
        self.position = position
        self.data = data
        self.color = color


class Vehicle:
    data: float
    speed: float

    def __init__(self, params: VehicleParams):
        self.params = params

        self.position = params.position  
        self.color = params.color

        self.data = params.data
        self.speed = 0

    def av_data(self, av):
        if av == 0:
            return 1
        c = min(av, self.data - self.speed)
        self.speed += c


        return c / av

    @property
    def depleted(self):
        return self.data <= self.speed

    def get_data_rate(self, pos, channel: Channel):
        rate = channel.compute_rate(uav_pos=pos, device_pos=self.position)
        # self.data_rate_timeseries.append(rate)
        return rate


class DeviceList:

    def __init__(self, params):
        self.devices = [Vehicle(device) for device in params]

    def get_data_map(self, shape):
        data_map = np.zeros(shape, dtype=float)

        for device in self.devices:
            data_map[device.position[1], device.position[0]] = device.data - device.speed

        return data_map

    def get_aved_map(self, shape):
        data_map = np.zeros(shape, dtype=float)

        for device in self.devices:
            data_map[device.position[1], device.position[0]] = device.speed

        return data_map

    def get_best_data_rate(self, pos, channel: Channel):

        data_rates = np.array(
            [device.get_data_rate(pos, channel) if not device.depleted else 0 for device in self.devices])
        idx = np.argmax(data_rates) if data_rates.any() else -1
        return data_rates[idx], idx

    def av_data(self, av, idx):
        ratio = 1
        if idx != -1:
            ratio = self.devices[idx].av_data(av)

        return ratio

    def get_devices(self):
        return self.devices

    def get_device(self, idx):
        return self.devices[idx]

    def get_total_data(self):
        return sum(list([device.data for device in self.devices]))

    def get_speed(self):
        return sum(list([device.speed for device in self.devices]))

    @property
    def num_devices(self):
        return len(self.devices)
