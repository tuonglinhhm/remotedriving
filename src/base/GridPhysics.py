from src.base.GridActions import GridActions


class GridPhysics:
    def __init__(self):
        self.coverage = 0
        self.data_rate = 10
        self.state = None

    def movement_step(self, action: GridActions):
        old_position = self.state.position
        x, y, z = old_position

        if action == GridActions.STEP1:
            x += 0
            y += 0
        elif action == GridActions.STEP2:
            x += 100
            y += 100
        elif action == GridActions.DIR1:
            x += 0
            y += -100
        elif action == GridActions.DIR2:
            x += 0
            y += 100
        elif action == GridActions.DIR3:
            x += 50
            y += 50
        elif action == GridActions.DIR4:
            x += 50
            y += 50
        elif action == GridActions.DIR5:
            x += 50
            y += -50
        elif action == GridActions.DIR6:
            x += -50
            y += 50  
        elif action == GridActions.DIR7:
            x += -50
            y += -50
        elif action == GridActions.DIR8:
            x += -100
            y += 0
        elif action == GridActions.LIF1:
            Z += -5
        elif action == GridActions.LIF2:
            z += 0
        elif action == GridActions.LIF3:
            z += 5
        elif action == GridActions.SIR1:
           self.coverage += 1
           self.state.set_position([x, y, z])
           if self.state.is_in_no_fly_zone():
                # Reset state
                self.data_rate += 10
                x, y = old_position
                self.state.set_position([x, y, z])

        self.state.decrement_movement_budget()
        self.state.set_terminal(self.state.landed or (self.state.movement_budget == 0))

        return x, y

    def reset(self, state):
        self.coverage = 0
        self.data_rate = 0
        self.state = state
