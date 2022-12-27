
import torch
from k_diffusion.external import CompVisDenoiser
from typing import Iterable

class KCFGDenoiser(torch.nn.Module):
    inner_model: CompVisDenoiser

    def __init__(self, model: CompVisDenoiser):
        super().__init__()
        self.inner_model = model

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        uncond: torch.Tensor,
        conditions: Iterable[torch.Tensor],
        cond_scale: float,
    ) -> torch.Tensor:
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, *conditions])
        conditions_len = len(conditions)
        uncond, *conditions = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(
            1 + conditions_len
        )
        cond = torch.sum(torch.stack(conditions), dim=0) / conditions_len
        return uncond + (cond - uncond) * cond_scale


class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

