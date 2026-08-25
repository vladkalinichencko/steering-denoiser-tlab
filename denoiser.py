"""The activation model, in the two objectives the task puts side by side.

GLP (arXiv:2602.06964) trains flow matching on residual-stream activations and, at
inference, treats a steered activation as a corrupted sample: standardize it, jump to
noise level `t_start`, integrate the learned velocity field back to zero. The task's
own proposal, denoiser(h + eps) -> h, is the same network with one regression step
instead of a trajectory — so both live here and differ only by `predict`.

Worth noticing: the corruption the task suggests, t*h + (1-t)*eps with t ~ U[0,1], is
exactly the flow-matching interpolant. The naive baseline is already halfway there.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """Llama3-style SwiGLU block; t modulates the gate pre-activation, as in GLP."""

    def __init__(self, d_model, d_ff, d_t):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.t_proj = nn.Linear(d_t, d_ff) if d_t else None

    def forward(self, x, temb):
        z = self.norm(x)
        g = self.gate(z)
        if self.t_proj is not None:
            g = g * self.t_proj(temb)
        return x + self.down(F.silu(g) * self.up(z))

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None].float() * freqs[None]
    return torch.cat([ang.cos(), ang.sin()], dim=-1)

class Denoiser(nn.Module):
    """predict="residual": h_hat = h + net(h), the task's denoiser.
       predict="velocity": u_hat(z, t), the GLP flow-matching field.

    Standardisation lives inside the module and travels with the checkpoint, so no
    caller can forget it — GLP normalises activations before training and the scale
    of GPT-2's residual stream is nowhere near unit variance.
    """

    def __init__(self, d_act=768, width=2, expand=2, n_blocks=4, predict="residual"):
        super().__init__()
        d_model, self.predict = width * d_act, predict
        d_t = d_model if predict == "velocity" else 0
        self.inp = nn.Linear(d_act, d_model)
        self.blocks = nn.ModuleList(Block(d_model, expand * d_model, d_t)
                                    for _ in range(n_blocks))
        self.norm = nn.RMSNorm(d_model)
        self.out = nn.Linear(d_model, d_act)
        self.t_mlp = nn.Sequential(nn.Linear(d_t, d_t), nn.SiLU(), nn.Linear(d_t, d_t)) \
            if d_t else None
        self.register_buffer("mean", torch.zeros(d_act))
        self.register_buffer("std", torch.ones(d_act))

    def set_stats(self, acts):
        self.mean.copy_(acts.mean(0))
        self.std.copy_(acts.std(0).clamp_min(1e-6))

    def standardize(self, h):
        return (h - self.mean) / self.std

    def restore(self, z):
        return z * self.std + self.mean

    @torch.no_grad()
    def repair(self, h):
        """One regression, in and out of standardised space.

        The network is trained on standardised activations, so a caller that hands it
        a raw residual-stream vector gets nonsense — and nothing in the shapes says so.
        This is the only entry point inference should use.
        """
        return self.restore(self(self.standardize(h)))

    def forward(self, z, t=None):
        temb = None if self.t_mlp is None else self.t_mlp(
            timestep_embedding(t, self.inp.out_features))
        x = self.inp(z)
        for block in self.blocks:
            x = block(x, temb)
        out = self.out(self.norm(x))
        return z + out if self.predict == "residual" else out

def flow_batch(z0, generator=None):
    """z_t = (1-t) z0 + t eps, target velocity u = eps - z0 (GLP, eq. for flow matching)."""
    t = torch.rand(len(z0), device=z0.device, generator=generator)
    eps = torch.randn(z0.shape, device=z0.device, generator=generator)
    zt = (1 - t[:, None]) * z0 + t[:, None] * eps
    return zt, t, eps - z0

@torch.no_grad()
def sdedit(net, h, t_start=0.5, steps=20, generator=None):
    """GLP inference: noise the edited activation to t_start, integrate back to 0.

    Defaults are the paper's (t_start = 0.5, 20 steps). Lower t_start keeps more of
    the edit and repairs less; that trade-off is the whole knob.
    """
    z = net.standardize(h)
    eps = torch.randn(z.shape, device=z.device, generator=generator)
    z = (1 - t_start) * z + t_start * eps
    grid = torch.linspace(t_start, 0.0, steps + 1, device=z.device)
    for a, b in zip(grid[:-1], grid[1:]):
        z = z + (b - a) * net(z, a.expand(len(z)))
    return net.restore(z)

@torch.no_grad()
def sdedit_onestep(net, h, t_start=0.5, generator=None):
    """One evaluation instead of twenty: the field taken once and followed all the way.

    The task's own complaint about GLP is that sampling is expensive at inference. If
    the velocity at t_start already points at the answer, a single Euler step of length
    t_start lands there — this is the crude version of one-step distillation, and it
    costs one forward pass. How much it loses against the full trajectory is exactly
    what the Pareto front will say.
    """
    z = net.standardize(h)
    eps = torch.randn(z.shape, device=z.device, generator=generator)
    z = (1 - t_start) * z + t_start * eps
    t = torch.full((len(z),), t_start, device=z.device)
    return net.restore(z - t_start * net(z, t))
