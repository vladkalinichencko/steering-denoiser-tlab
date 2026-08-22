"""Activation denoisers and the objectives listed in NOTES.md."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    angle = t[:, None].float() * freq[None]
    return torch.cat((angle.cos(), angle.sin()), dim=-1)


class SwiGLUBlock(nn.Module):
    def __init__(self, width: int, hidden: int, conditioned: bool):
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.condition = nn.Linear(width, hidden) if conditioned else None

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        z = self.norm(x)
        gate = self.gate(z)
        if self.condition is not None:
            gate = gate * self.condition(condition)
        return x + self.down(F.silu(gate) * self.up(z))


class ActivationModel(nn.Module):
    """One model body for MSE, GLP, Consistency, Rectified Flow and MeanFlow."""

    def __init__(self, d_act=768, width=1536, hidden=3072, blocks=4,
                 time_inputs=0, simple=False):
        super().__init__()
        self.config = dict(d_act=d_act, width=width, hidden=hidden, blocks=blocks,
                           time_inputs=time_inputs, simple=simple)
        self.register_buffer("mean", torch.zeros(d_act))
        self.register_buffer("std", torch.ones(d_act))
        self.simple = simple
        self.time_inputs = time_inputs
        if simple:
            self.body = nn.Sequential(
                nn.RMSNorm(d_act), nn.Linear(d_act, hidden), nn.GELU(), nn.Linear(hidden, d_act)
            )
            return
        self.input = nn.Linear(d_act, width)
        self.blocks = nn.ModuleList(SwiGLUBlock(width, hidden, time_inputs > 0)
                                    for _ in range(blocks))
        self.output = nn.Sequential(nn.RMSNorm(width), nn.Linear(width, d_act))
        self.time = (nn.Sequential(nn.Linear(width * time_inputs, width), nn.SiLU(),
                                   nn.Linear(width, width)) if time_inputs else None)

    def set_stats(self, train: torch.Tensor) -> None:
        self.mean.copy_(train.mean(0))
        self.std.copy_(train.std(0).clamp_min(1e-6))

    def standardize(self, h: torch.Tensor) -> torch.Tensor:
        return (h - self.mean) / self.std

    def restore(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean

    def forward(self, z: torch.Tensor, *times: torch.Tensor) -> torch.Tensor:
        if len(times) != self.time_inputs:
            raise ValueError(f"expected {self.time_inputs} time inputs, got {len(times)}")
        if self.simple:
            return self.body(z)
        condition = None
        if times:
            embeddings = [time_embedding(t, self.input.out_features) for t in times]
            condition = self.time(torch.cat(embeddings, dim=-1))
        x = self.input(z)
        for block in self.blocks:
            x = block(x, condition)
        return self.output(x)


def build_model(method: str, d_act: int, reduced: bool) -> ActivationModel:
    if method == "additive_simple":
        return ActivationModel(d_act=d_act, hidden=4 * d_act, simple=True)
    time_inputs = 2 if method == "meanflow" else int(method not in {"additive_capacity"})
    width = d_act if reduced else 2 * d_act
    blocks = 2 if reduced else 4
    return ActivationModel(d_act, width, 2 * width, blocks, time_inputs)


def flow_path(z0: torch.Tensor, generator=None):
    t = torch.rand(len(z0), device=z0.device, generator=generator)
    noise = torch.randn(z0.shape, device=z0.device, generator=generator)
    zt = (1 - t[:, None]) * z0 + t[:, None] * noise
    return zt, t, noise - z0


def loss(method: str, model: ActivationModel, z0: torch.Tensor, *, sigma=1.0,
         target_model=None, teacher=None, generator=None) -> torch.Tensor:
    if method.startswith("additive"):
        noisy = z0 + sigma * torch.randn(z0.shape, device=z0.device, generator=generator)
        return F.mse_loss(noisy + model(noisy), z0)

    if method == "interpolation":
        zt, t, _ = flow_path(z0, generator)
        return F.mse_loss(zt + model(zt, t), z0)

    if method in {"glp", "rectified"}:
        if method == "rectified":
            if teacher is None:
                raise ValueError("rectified flow needs a frozen GLP teacher")
            noise = torch.randn(z0.shape, device=z0.device, generator=generator)
            with torch.no_grad():
                endpoint = integrate(teacher, noise, 1.0, 20)[-1]
            t = torch.rand(len(z0), device=z0.device, generator=generator)
            zt = (1 - t[:, None]) * endpoint + t[:, None] * noise
            target = noise - endpoint
        else:
            zt, t, target = flow_path(z0, generator)
        return F.mse_loss(model(zt, t), target)

    if method == "consistency":
        if target_model is None:
            raise ValueError("consistency training needs an EMA target")
        n = torch.randint(0, 19, (len(z0),), device=z0.device, generator=generator)
        low, high = n / 19, (n + 1) / 19
        noise = torch.randn(z0.shape, device=z0.device, generator=generator)
        z_low = (1 - low[:, None]) * z0 + low[:, None] * noise
        z_high = (1 - high[:, None]) * z0 + high[:, None] * noise
        online = z_high + high[:, None] * model(z_high, high)
        with torch.no_grad():
            target = z_low + low[:, None] * target_model(z_low, low)
        return F.mse_loss(online, target)

    if method == "meanflow":
        a = torch.sigmoid(torch.randn(len(z0), device=z0.device, generator=generator) - 0.4)
        b = torch.sigmoid(torch.randn(len(z0), device=z0.device, generator=generator) - 0.4)
        r, t = torch.minimum(a, b), torch.maximum(a, b)
        same = torch.rand(len(z0), device=z0.device, generator=generator) >= 0.25
        r = torch.where(same, t, r)
        noise = torch.randn(z0.shape, device=z0.device, generator=generator)
        velocity = noise - z0
        zt = (1 - t[:, None]) * z0 + t[:, None] * noise

        def field(z, start, end):
            return model(z, end, end - start)

        prediction, derivative = jvp(field, (zt, r, t),
                                     (velocity, torch.zeros_like(r), torch.ones_like(t)))
        target = velocity - (t - r)[:, None] * derivative
        error = prediction - target.detach()
        squared = error.square().sum(-1)
        weight = (squared + 1e-3).reciprocal().detach()
        return (weight * error.square().mean(-1)).mean()

    raise ValueError(method)


@torch.no_grad()
def update_ema(target: ActivationModel, online: ActivationModel, decay=0.999) -> None:
    for target_value, online_value in zip(target.state_dict().values(), online.state_dict().values()):
        if target_value.is_floating_point():
            target_value.lerp_(online_value, 1 - decay)
        else:
            target_value.copy_(online_value)


def make_ema(model: ActivationModel) -> ActivationModel:
    target = copy.deepcopy(model).eval()
    target.requires_grad_(False)
    return target


@torch.no_grad()
def integrate(model: ActivationModel, z: torch.Tensor, start: float, steps: int):
    states = [z]
    grid = torch.linspace(start, 0, steps + 1, device=z.device)
    for current, following in zip(grid[:-1], grid[1:]):
        t = current.expand(len(z))
        z = z + (following - current) * model(z, t)
        states.append(z)
    return states


@torch.no_grad()
def repair(method: str, model: ActivationModel, h: torch.Tensor, *, t_start=0.5,
           steps=20, generator=None):
    z = model.standardize(h)
    if method.startswith("additive"):
        return model.restore(z + model(z)), [z, z + model(z)]
    if method == "interpolation":
        t = torch.full((len(z),), t_start, device=z.device)
        endpoint = z + model(z, t)
        return model.restore(endpoint), [z, endpoint]
    noise = torch.randn(z.shape, device=z.device, generator=generator)
    zt = (1 - t_start) * z + t_start * noise
    t = torch.full((len(z),), t_start, device=z.device)
    if method == "consistency":
        states = [zt, zt + t[:, None] * model(zt, t)]
    elif method == "meanflow":
        zero = torch.zeros_like(t)
        states = [zt, zt - t[:, None] * model(zt, t, t - zero)]
    else:
        states = integrate(model, zt, t_start, steps)
    return model.restore(states[-1]), states
