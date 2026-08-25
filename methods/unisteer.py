"""Неудачный нелинейный стиринг: UniSteer (копия unisteer_mode из screening.py)."""

from pathlib import Path

import torch

from nonlinear import ConditionalFlow, conditional_transport

def unisteer_mode(path: Path, target: torch.device):
    blob = torch.load(path, map_location=target, weights_only=False)
    model = ConditionalFlow(**blob["model_config"]).to(target).eval()
    model.load_state_dict(blob["model"])
    positive, null = blob["conditions"][[0, 2]].to(target)
    amounts = {0.0: 0.0}

    def endpoint(h, amount, path=False):
        return conditional_transport(model, h, null, positive, amount, 10, path)

    def apply(h, strength, generator):
        key = float(strength)
        if key not in amounts:
            low, high = 0.0, 1.0
            for _ in range(10):
                middle = (low + high) / 2
                if (endpoint(h, middle) - h).norm(dim=-1).mean() < strength:
                    low = middle
                else:
                    high = middle
            amounts[key] = (low + high) / 2
        fixed, states = endpoint(h, amounts[key], True)
        return fixed, states

    return {"apply": apply, "checkpoint": str(path),
            "parameters": sum(value.numel() for value in model.parameters())}

def build(args, nets, bank, v, alpha):
    path = Path(getattr(args, "unisteer", "runs/mac_reduced_unisteer/best.pt"))
    if not path.exists():
        return []
    mode = unisteer_mode(path, bank.device)
    return [("unisteer", lambda h, s=alpha, m=mode: m["apply"](h, s, None)[0])]
