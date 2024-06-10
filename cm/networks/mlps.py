
from multiprocessing.sharedctypes import Value
import einops

import torch
from torch import DictType, nn
from omegaconf import DictConfig
import numpy as np 


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def round_ste(z):
    """Round with straight through gradients."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()

class FSQ(nn.Module):
    """Quantizer."""

    def __init__(self, levels: list[int], eps: float = 1e-3):
        super(FSQ, self).__init__()
        self._levels = levels
        self._eps = eps
        self._levels_np = np.asarray(levels)

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return np.prod(self._levels)

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        device = z.device
        half_l = torch.tensor((self._levels_np - 1) * (1 - self._eps) / 2, device=device)
        offset = torch.where(torch.tensor(np.mod(self._levels_np, 2) == 1, device=device), torch.tensor(0.0, device=device), torch.tensor(0.5, device=device))
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))

        # Renormalize to [-1, 1].
        half_width = torch.tensor(self._levels_np // 2, device=z.device)
        return quantized / half_width

    def forward(self, x):
        return self.quantize(x).float()


class MLPNetwork(nn.Module):
    """
    Simple multi layer perceptron network which can be generated with different 
    activation functions with and without spectral normalization of the weights
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 100,
        num_hidden_layers: int = 1,
        output_dim=1,
        L_fsq=-1,
        device: str = 'cuda'
    ):
        super(MLPNetwork, self).__init__()
        self.L_fsq = L_fsq
        self.network_type = "mlp"
        # define number of variables in an input sequence
        self.input_dim = input_dim
        # the dimension of neurons in the hidden layer
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        # number of samples per batch
        self.output_dim = output_dim
        # set up the network
        self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim)])
        self.layers.extend(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim)
                for i in range(1, self.num_hidden_layers)
            ]
        )
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        if self.L_fsq > 1:
            self.fsq_layer = self.num_hidden_layers // 2 + 1
            self.layers.insert(self.fsq_layer, FSQ([L_fsq] * self.hidden_dim))

        # build the activation layer
        self.act = nn.Mish()
        self._device = device
        self.layers.to(self._device)

    def forward(self, x):

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                out = layer(x)
            else:
                if idx < len(self.layers) - 2:
                    out = layer(out) # + out
                else:
                    out = layer(out)
            if self.L_fsq > 1 and idx == self.fsq_layer:
                continue
            if idx < len(self.layers) - 1:
                out = self.act(out)
        return out

    def get_device(self, device: torch.device):
        self._device = device
        self.layers.to(device)
    
    def get_params(self):
        return self.layers.parameters()


class ConistencyScoreNetwork(nn.Module):

    def __init__(
            self,
            x_dim: int,
            hidden_dim: int,
            time_embed_dim: int,
            cond_dim: int,
            cond_mask_prob: float,
            num_hidden_layers: int = 1,
            output_dim=1,
            device: str = 'cuda',
            cond_conditional: bool = True,
            L_fsq: int = -1,
    ):
        super(ConistencyScoreNetwork, self).__init__()
        #  Gaussian random feature embedding layer for time
        self.embed = GaussianFourierProjection(time_embed_dim).to(device)
        self.time_embed_dim = time_embed_dim
        self.cond_mask_prob = cond_mask_prob
        self.cond_conditional = cond_conditional
        if self.cond_conditional:
            input_dim = self.time_embed_dim +  x_dim  + cond_dim
        else:
            input_dim = self.time_embed_dim +  x_dim  
        # set up the network
        self.layers = MLPNetwork(
                input_dim,
                hidden_dim,
                num_hidden_layers,
                output_dim,
                L_fsq,
                device
            ).to(device)

        # build the activation layer
        self.act = nn.Mish()
        self.device = device
        self.sigma = None
        self.training = True

    def forward(self, x, cond, sigma, uncond=False):
        # Obtain the Gaussian random feature embedding for t
        if len(sigma.shape) == 0:
            sigma  = einops.rearrange(sigma, ' -> 1')
            sigma = sigma.unsqueeze(1)
        elif len(sigma.shape) == 1:
            sigma = sigma.unsqueeze(1)
        embed = self.embed(sigma)
        embed.squeeze_(1)
        if embed.shape[0] != x.shape[0]:
            embed = einops.repeat(embed, '1 d -> (1 b) d', b=x.shape[0])
        # during training randomly mask out the cond
        # to train the conditional model with classifier-free guidance wen need 
        # to 0 out some of the conditional during training with a desrired probability
        # it is usually in the range of 0,1 to 0.2 
        if self.training and cond is not None:
            cond = self.mask_cond(cond)
        # we want to use unconditional sampling during classifier free guidance
        if uncond:
            cond = torch.zeros_like(cond)   # cond
        if self.cond_conditional:
            x = torch.cat([x, cond, embed], dim=-1) 
        else:
            x = torch.cat([x, embed], dim=-1) 
        x = self.layers(x) 
        return x  # / marginal_prob_std(t, self.sigma, self.device)[:, None]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones((bs, d), device=cond.device) * self.cond_mask_prob)# .view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def get_params(self):
        return self.parameters()