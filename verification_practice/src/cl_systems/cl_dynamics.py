import torch
import torch.nn as nn
import torch.nn.functional as F
# from auto_LiRPA import PerturbationLpNorm, BoundedParameter
# from nfl_veripy.dynamics import DoubleIntegrator

class ClosedLoopDynamics(nn.Module):
    def __init__(self, controller, dynamics, num_steps=1, device='cpu') -> None:
        super().__init__()
        self.controller = controller
        self.dynamics = dynamics
        self.At = torch.tensor(dynamics.At, dtype=torch.float32, device=device).transpose(0, 1)
        self.bt = torch.tensor(dynamics.bt, dtype=torch.float32, device=device).transpose(0, 1)
        self.ct = torch.tensor(dynamics.ct, dtype=torch.float32, device=device)
        self.num_steps = num_steps

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps

    def to(self, device):
        self.At = self.At.to(device)
        self.bt = self.bt.to(device)
        self.ct = self.ct.to(device)
        self.controller = self.controller.to(device)

        return self

    
    def forward(self, xt):
        num_steps = self.num_steps
        xts = [xt]

        for i in range(num_steps):
            ut = self.controller(xts[-1])
            xt1 = torch.matmul(xts[-1], self.At) + torch.matmul(ut, self.bt) + self.ct
            xts.append(xt1)

        
        return xts[-1]