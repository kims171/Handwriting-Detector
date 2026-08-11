"""
Adapter that makes a plain optimizer (e.g. AdamW) speak SAM's first_step/
second_step interface, so train_step() doesn't need to change at all.
"""

class PlainOptimizerAdapter:
    def __init__(self, base_optimizer):
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups

    def first_step(self, zero_grad=False):
        if zero_grad:
            self.zero_grad()

    def second_step(self, zero_grad=False):
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
