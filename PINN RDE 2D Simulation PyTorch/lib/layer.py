import torch
import torch.nn as nn
import torch.autograd as autograd


class GradientLayer(nn.Module):
    def __init__(self, model):
        super(GradientLayer, self).__init__()
        self.model = model

    def forward(self, x):
        x.requires_grad_(True)
        c = self.model(x)

        # Compute first order derivatives
        gradients = autograd.grad(c, x, grad_outputs=torch.ones_like(c), create_graph=True)[0]
        dc_dt, dc_dx, dc_dy = gradients[:, 0], gradients[:, 1], gradients[:, 2]

        # Compute second order derivatives
        d2c_dx2 = autograd.grad(dc_dx, x, grad_outputs=torch.ones_like(dc_dx), create_graph=True)[0][:, 1]
        d2c_dy2 = autograd.grad(dc_dy, x, grad_outputs=torch.ones_like(dc_dy), create_graph=True)[0][:, 2]

        return c, dc_dt, dc_dx, dc_dy, d2c_dx2, d2c_dy2


class BoundaryGradientLayer(nn.Module):
    def __init__(self, model):
        super(BoundaryGradientLayer, self).__init__()
        self.model = model

    def forward(self, x):
        x.requires_grad_(True)
        c = self.model(x)

        gradients = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c), create_graph=True)[0]
        dc_dt, dc_dx, dc_dy = gradients[:, 0], gradients[:, 1], gradients[:, 2]
        return c, dc_dt, dc_dx, dc_dy
