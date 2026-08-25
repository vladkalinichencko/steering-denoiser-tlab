"""Неудачный нелинейный стиринг: INNSteer (копия inn_mode из screening.py)."""

from pathlib import Path

import torch

from nonlinear import INNSteer

def inn_mode(path: Path, target: torch.device):
    blob = torch.load(path, map_location=target, weights_only=False)
    model = INNSteer(**blob["model_config"]).to(target).eval()
    model.load_state_dict(blob["model"])
    mean, std, direction = blob["mean"].to(target), blob["std"].to(target), \
        blob["direction"].to(target)
    amounts = {0.0: 0.0}

    def raw(h, amount):
        latent = model((h - mean) / std)[0] + amount * direction
        return model(latent, inverse=True)[0] * std + mean

    def apply(h, strength, generator):
        key = float(strength)
        if key not in amounts:
            low, high = 0.0, 1.0
            while (raw(h, high) - h).norm(dim=-1).mean() < strength and high < 128:
                high *= 2
            for _ in range(10):
                middle = (low + high) / 2
                if (raw(h, middle) - h).norm(dim=-1).mean() < strength:
                    low = middle
                else:
                    high = middle
            amounts[key] = (low + high) / 2
        fixed = h if key == 0 else raw(h, amounts[key])
        return fixed, [h, fixed]

    def jacobian(h, strength):
        low, high = 0.0, 1.0
        while (raw(h, high) - h).norm(dim=-1).mean() < strength and high < 128:
            high *= 2
        for _ in range(10):
            middle = (low + high) / 2
            if (raw(h, middle) - h).norm(dim=-1).mean() < strength:
                low = middle
            else:
                high = middle
        amount = (low + high) / 2
        with torch.enable_grad():
            value = h[0].detach().requires_grad_(True)
            matrix = torch.func.jacrev(lambda x: raw(x[None], amount)[0])(value)
        return torch.linalg.svdvals(matrix).detach().cpu().tolist()

    return {"apply": apply, "checkpoint": str(path),
            "parameters": sum(value.numel() for value in model.parameters()),
            "coordinates": lambda h: model((h - mean) / std)[0].detach().cpu().tolist(),
            "jacobian": jacobian}

def build(args, nets, bank, v, alpha):
    path = Path(getattr(args, "inn", "runs/mac_reduced_inn/best.pt"))
    if not path.exists():
        return []
    mode = inn_mode(path, bank.device)
    return [("inn", lambda h, s=alpha, m=mode: m["apply"](h, s, None)[0])]
