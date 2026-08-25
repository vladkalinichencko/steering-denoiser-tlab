"""Неудачный нелинейный стиринг: Curveball (копия curveball_mode из screening.py)."""

import pickle
from pathlib import Path

def curveball_mode(path: Path):
    with path.open("rb") as file:
        blob = pickle.load(file)
    model, mean, std = blob["model"], blob["mean"], blob["std"]
    amounts = {0.0: 0.0}

    def apply(h, strength, generator):
        key = float(strength)
        source = ((h.detach().cpu() - mean) / std).float()

        def restore(amount):
            return model.steer(source, amount) * std + mean

        if key not in amounts:
            low, high = 0.0, 1.0
            while (restore(high) - h.cpu()).norm(dim=-1).mean() < strength and high < 128:
                high *= 2
            for _ in range(10):
                middle = (low + high) / 2
                if (restore(middle) - h.cpu()).norm(dim=-1).mean() < strength:
                    low = middle
                else:
                    high = middle
            amounts[key] = (low + high) / 2
        fixed = h if key == 0 else restore(amounts[key]).to(h.device)
        return fixed, [h, fixed]

    return {"apply": apply, "checkpoint": str(path), "parameters": 0,
            "coordinates": lambda h: model.transform(((h.cpu() - mean) / std).float()).tolist()}

def build(args, nets, bank, v, alpha):
    path = Path(getattr(args, "curveball", "runs/mac_reduced_curveball/model.pkl"))
    if not path.exists():
        return []
    mode = curveball_mode(path)
    return [("curveball", lambda h, s=alpha, m=mode: m["apply"](h, s, None)[0])]
