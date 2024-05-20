import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import GradientLayer, BoundaryGradientLayer


class PINN(nn.Module):
    def __init__(self, network, no_flux_bnd:bool):
        super(PINN, self).__init__()
        self.network = network
        self.grads = GradientLayer(self.network)
        self.boundaryGrad = BoundaryGradientLayer(self.network)
        self.no_flux_bnd = no_flux_bnd  # whether to use no-flux boundary conditions

    def forward(self, inputs):
        TYR_dmn0, aux1_dmn0, aux2_dmn0, aux3_dmn0, TYR_bnd0, TYR_bnd1, TYR_bnd2, TYR_bnd3, TYR_bnd4 = inputs
        aux1_dmn0, aux2_dmn0, aux3_dmn0 = aux1_dmn0.squeeze(1), aux2_dmn0.squeeze(1), aux3_dmn0.squeeze(1)
        
        Cdmn0, dC_dT_dmn0, dC_dY_dmn0, dC_dR_dmn0, d2C_dY2_dmn0, d2C_dR2_dmn0 = self.grads(TYR_dmn0)
        C_dmn0 = d2C_dY2_dmn0 + d2C_dR2_dmn0 + aux1_dmn0 * dC_dR_dmn0 + aux2_dmn0 * dC_dY_dmn0 - aux3_dmn0 * dC_dR_dmn0
        C_dmn0 = C_dmn0.unsqueeze(1)

        C_bnd0 = self.network(TYR_bnd0)

        Cbnd1, dC_dT_bnd1, dC_dY_bnd1, dC_dR_bnd1 = self.boundaryGrad(TYR_bnd1)
        C_bnd1 = dC_dY_bnd1.unsqueeze(1)

        C_bnd2 = self.network(TYR_bnd2)
        C_bnd3 = self.network(TYR_bnd3)

        if self.no_flux_bnd:
            Cbnd4, dC_dT_bnd4, dC_dY_bnd4, dC_dR_bnd4 = self.boundaryGrad(TYR_bnd4)
            C_bnd4 = dC_dR_bnd4.unsqueeze(1)
        else:
            C_bnd4 = self.network(TYR_bnd4)

        return [C_dmn0, C_bnd0, C_bnd1, C_bnd2, C_bnd3, C_bnd4]
    