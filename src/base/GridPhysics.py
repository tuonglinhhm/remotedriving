from src.base.GridActions import GridActions


class GridPhysics:
    def __init__(self):
        self.coverage = 0
        self.data_rate = 10
        self.state = None

    def movement_step(self, action: GridActions):
        old_position = self.state.position
        x, y = old_position[0], old_position[1]
        z = self.state.altitude

        # Simple grid motion model based on the provided direction.
        if action == GridActions.STEP1:
            pass
        elif action == GridActions.STEP2:
            x += 1
            y += 1
        elif action == GridActions.DIR1:
            y -= 1
        elif action == GridActions.DIR2:
            y += 1
        elif action == GridActions.DIR3:
            x += 1
            y += 1
        elif action == GridActions.DIR4:
            x += 1
        elif action == GridActions.DIR5:
            x += 1
            y -= 1
        elif action == GridActions.DIR6:
            x -= 1
            y += 1
        elif action == GridActions.DIR7:
            x -= 1
            y -= 1
        elif action == GridActions.DIR8:
            x -= 1
        elif action == GridActions.LIF1:
            z -= 1
        elif action == GridActions.LIF2:
            pass
        elif action == GridActions.LIF3:
            z += 1

        self.state.set_position([x, y, z])

        if self.state.is_in_no_fly_zone():
            # Reset to previous position if the action leads into an invalid area.
            self.state.set_position(old_position)

        self.state.decrement_movement_budget()
        self.state.set_terminal(self.state.landed or (self.state.movement_budget == 0))

        return x, y

    def reset(self, state):
        self.coverage = 0
        self.data_rate = 0
        self.state = state
