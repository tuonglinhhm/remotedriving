from src.Map.Map import Map


class BaseState:
    def __init__(self, map_init: Map):
        self.no_fly_zone = map_init.nfz
        self.obstacles = map_init.obstacles
        self.vulnerable_cell = map_init.start_land_zone
 
    @property
    def shape(self):
        return self.vulnerable_cell.shape[:2]
