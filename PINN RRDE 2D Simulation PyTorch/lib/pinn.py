import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import GradientLayer, BoundaryGradientLayer


class PINN(nn.Module):
    def __init__(self, network, RADIALDIFFUSION, NO_FLUX_BND):
        super(PINN, self).__init__()
        self.network = network
        self.grads = GradientLayer(network)  # Assume imported and adapted to PyTorch
        self.boundaryGrad = BoundaryGradientLayer(network)  # Assume imported and adapted to PyTorch
        self.RADIALDIFFUSION = RADIALDIFFUSION
        self.NO_FLUX_BND = NO_FLUX_BND

    def forward(self, inputs):
        (TYR_dmn0, aux1overR_dmn0, auxY_dmn0, auxR_dmn0, TYR_dmn1, aux1overR_dmn1,
         auxY_dmn1, auxR_dmn1, TYR_bnd0, TYR_bnd1, TYR_bnd2, TYR_bnd3,
         TYR_bnd4, TYR_bnd5, TYR_bnd6) = [torch.tensor(x, dtype=torch.float32) for x in inputs]
        # Domain calculations
        Cdmn0, dC_dT_dmn0, dC_dY_dmn0, dC_dR_dmn0, d2C_dY2_dmn0, d2C_dR2_dmn0 = self.grads(TYR_dmn0)
        Cdmn1, dC_dT_dmn1, dC_dY_dmn1, dC_dR_dmn1, d2C_dY2_dmn1, d2C_dR2_dmn1 = self.grads(TYR_dmn1)

        # Custom operations based on conditions
        if self.RADIALDIFFUSION:
            C_dmn0 = d2C_dY2_dmn0 + d2C_dR2_dmn0 + aux1overR_dmn0 * dC_dR_dmn0 - auxY_dmn0 * dC_dY_dmn0 - auxR_dmn0 * dC_dR_dmn0
            C_dmn1 = d2C_dY2_dmn1 + d2C_dR2_dmn1 + aux1overR_dmn1 * dC_dR_dmn1 - auxY_dmn1 * dC_dY_dmn1 - auxR_dmn1 * dC_dR_dmn1
        else:
            C_dmn0 = d2C_dY2_dmn0 - auxY_dmn0 * dC_dY_dmn0 - auxR_dmn0 * dC_dR_dmn0
            C_dmn1 = d2C_dY2_dmn1 - auxY_dmn1 * dC_dY_dmn1 - auxR_dmn1 * dC_dR_dmn1

        # Boundary calculations
        C_bnd0 = self.network(TYR_bnd0)
        Cbnd1, dC_dT_bnd1, dC_dY_bnd1, dC_dR_bnd1 = self.boundaryGrad(TYR_bnd1)
        C_bnd1 = dC_dY_bnd1
        Cbnd3, dC_dT_bnd3, dC_dY_bnd3, dC_dR_bnd3 = self.boundaryGrad(TYR_bnd3)
        C_bnd3 = dC_dY_bnd3
        Cbnd4, dC_dT_bnd4, dC_dY_bnd4, dC_dR_bnd4 = self.boundaryGrad(TYR_bnd4)
        C_bnd4 = dC_dR_bnd4 if self.NO_FLUX_BND else Cbnd4
        C_bnd2 = self.network(TYR_bnd2)
        C_bnd5 = self.network(TYR_bnd5)
        C_bnd6 = self.network(TYR_bnd6)

        # Output processing
        outputs = [C_dmn0, C_dmn1, C_bnd0, C_bnd1, C_bnd2, C_bnd3, C_bnd4, C_bnd5, C_bnd6]
        return outputs
