"""Nonlinear steering maps from the methods listed in NOTES.md."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import KernelPCA


class Curveball:
    def __init__(self, components=20, degree=2, gamma=0.001):
        self.kpca = KernelPCA(components, kernel="poly", degree=degree,
                              gamma=gamma, coef0=1)

    def fit(self, positive: torch.Tensor, negative: torch.Tensor):
        self.values = torch.cat((positive, negative)).float().cpu()
        coordinates = torch.from_numpy(self.kpca.fit_transform(self.values.numpy())).float()
        split = len(positive)
        direction = coordinates[:split].mean(0) - coordinates[split:].mean(0)
        self.direction = direction / direction.norm()
        self.coordinates = coordinates
        distance = torch.pdist(coordinates)
        self.bandwidth = distance.median().clamp_min(1e-6)
        return self

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        result = self.kpca.transform(values.detach().float().cpu().numpy())
        return torch.from_numpy(np.asarray(result)).float()

    def preimage(self, coordinates: torch.Tensor) -> torch.Tensor:
        distance = torch.cdist(coordinates, self.coordinates).square()
        weight = torch.softmax(-distance / (2 * self.bandwidth.square()), dim=-1)
        return weight @ self.values

    def steer(self, values: torch.Tensor, amount: float) -> torch.Tensor:
        target_device = values.device
        source = values.detach().float().cpu()
        coordinates = self.transform(source)
        residual = source - self.preimage(coordinates)
        output = self.preimage(coordinates + amount * self.direction) + residual
        return output.to(target_device)


class Coupling(nn.Module):
    def __init__(self, dim: int, hidden: int, flip: bool):
        super().__init__()
        self.flip = flip
        self.net = nn.Sequential(nn.Linear(dim // 2, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 2 * (dim - dim // 2)))

    def forward(self, x: torch.Tensor, inverse=False):
        if self.flip:
            x = x.flip(-1)
        first, second = x.split((x.shape[-1] // 2, x.shape[-1] - x.shape[-1] // 2), -1)
        scale, shift = self.net(first).chunk(2, -1)
        scale = 0.75 * scale.tanh()
        if inverse:
            second = (second - shift) * (-scale).exp()
            logdet = -scale.sum(-1)
        else:
            second = second * scale.exp() + shift
            logdet = scale.sum(-1)
        output = torch.cat((first, second), -1)
        return (output.flip(-1) if self.flip else output), logdet


class ActNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, inverse=False):
        if inverse:
            return x * (-self.log_scale).exp() - self.bias, -self.log_scale.sum().expand(len(x))
        return (x + self.bias) * self.log_scale.exp(), self.log_scale.sum().expand(len(x))


class INNSteer(nn.Module):
    def __init__(self, dim=768, hidden=256, blocks=4):
        super().__init__()
        self.config = {"dim": dim, "hidden": hidden, "blocks": blocks}
        self.blocks = nn.ModuleList(Coupling(dim, hidden, i % 2 == 1)
                                    for i in range(blocks))
        self.norms = nn.ModuleList(ActNorm(dim) for _ in range(blocks))

    def forward(self, x: torch.Tensor, inverse=False):
        logdet = torch.zeros(len(x), device=x.device)
        indices = range(len(self.blocks) - 1, -1, -1) if inverse else range(len(self.blocks))
        for index in indices:
            if inverse:
                x, current = self.norms[index](x, True)
                logdet += current
            x, current = self.blocks[index](x, inverse)
            logdet += current
            if not inverse:
                x, current = self.norms[index](x)
                logdet += current
        return x, logdet


def inn_loss(model: INNSteer, positive: torch.Tensor, negative: torch.Tensor,
             direction_weight=1.0, logdet_weight=0.1):
    first, first_logdet = model(positive)
    second, second_logdet = model(negative)
    dim = positive.shape[-1]
    nll = torch.cat((0.5 * first.square().sum(-1) - first_logdet,
                     0.5 * second.square().sum(-1) - second_logdet)).mean() / dim
    direction = -(first.mean(0) - second.mean(0)).norm() / dim ** 0.5
    logdet = torch.cat((first_logdet, second_logdet))
    determinant = logdet.mean().square() + logdet.var()
    return nll + direction_weight * direction + logdet_weight * determinant, {
        "nll": nll, "direction": direction, "logdet": determinant}


class FiLMBlock(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.modulation = nn.Linear(width, 2 * hidden)

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        z = self.norm(x)
        gamma, beta = self.modulation(condition).chunk(2, -1)
        gate = (1 + gamma) * self.gate(z) + beta
        return x + self.down(torch.nn.functional.silu(gate) * self.up(z))


class ConditionalFlow(nn.Module):
    def __init__(self, dim=768, condition_dim=768, width=768, hidden=1536, blocks=2):
        super().__init__()
        self.config = {"dim": dim, "condition_dim": condition_dim, "width": width,
                       "hidden": hidden, "blocks": blocks}
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.input = nn.Linear(dim, width)
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.condition = nn.Linear(condition_dim, width)
        self.blocks = nn.ModuleList(FiLMBlock(width, hidden) for _ in range(blocks))
        self.output = nn.Sequential(nn.RMSNorm(width), nn.Linear(width, dim))

    def set_stats(self, values: torch.Tensor):
        self.mean.copy_(values.mean(0))
        self.std.copy_(values.std(0).clamp_min(1e-6))

    def standardize(self, values: torch.Tensor):
        return (values - self.mean) / self.std

    def restore(self, values: torch.Tensor):
        return values * self.std + self.mean

    def forward(self, x: torch.Tensor, time: torch.Tensor, condition: torch.Tensor):
        from tmp.methods import time_embedding
        state = self.input(x)
        context = self.time(time_embedding(time, state.shape[-1])) + self.condition(condition)
        for block in self.blocks:
            state = block(state, context)
        return self.output(state)


def conditional_flow_loss(model: ConditionalFlow, values: torch.Tensor,
                          condition: torch.Tensor, generator=None):
    time = torch.rand(len(values), device=values.device, generator=generator)
    noise = torch.randn(values.shape, device=values.device, generator=generator)
    state = (1 - time[:, None]) * noise + time[:, None] * values
    return torch.nn.functional.mse_loss(model(state, time, condition), values - noise)


@torch.no_grad()
def conditional_transport(model: ConditionalFlow, values: torch.Tensor,
                          source: torch.Tensor, target: torch.Tensor,
                          strength: float, steps=10, return_path=False):
    state = model.standardize(values)
    path = [values]
    middle = 1 - strength
    for start, end, condition in ((1.0, middle, source), (middle, 1.0, target)):
        grid = torch.linspace(start, end, steps + 1, device=values.device)
        for current, following in zip(grid[:-1], grid[1:]):
            time = current.expand(len(values))
            state = state + (following - current) * model(
                state, time, condition.expand(len(values), -1))
            path.append(model.restore(state))
    output = model.restore(state)
    return (output, path) if return_path else output
