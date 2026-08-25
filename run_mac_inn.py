"""Train the reduced INNSteer map on the fixed SST-5 contrast activations."""

import json
import math
import pathlib

import torch

from nonlinear import INNSteer, inn_loss
from training import schedule

def main() -> None:
    torch.manual_seed(0)
    source = torch.load("datasets/sst5_layer6_e51bdcd.pt", weights_only=False)
    positive, negative = source["positive"].float(), source["negative"].float()
    train = torch.cat((positive[:2048], negative[:2048]))
    mean, std = train.mean(0), train.std(0).clamp_min(1e-6)
    positive, negative = (positive - mean) / std, (negative - mean) / std
    model = INNSteer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: schedule(step, 300, 3))
    output = pathlib.Path("runs/mac_reduced_inn")
    output.mkdir(parents=True, exist_ok=True)
    config = {"train_pairs": 2048, "val_pairs": 256, "steps": 300, "batch": 64,
              "lr": 5e-4, "direction_weight": 1.0, "logdet_weight": 0.1,
              "model": model.config, "source": source["meta"]}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    history, best = [], math.inf
    for step in range(300):
        index = torch.randint(2048, (64,))
        value, parts = inn_loss(model, positive[index], negative[index])
        value.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if step % 30 == 0 or step == 299:
            with torch.no_grad():
                val, val_parts = inn_loss(model, positive[2048:2304], negative[2048:2304])
            row = {"step": step, "train_loss": float(value), "val_loss": float(val),
                   "nll": float(val_parts["nll"]), "direction": float(val_parts["direction"]),
                   "logdet": float(val_parts["logdet"]), "grad_norm": float(grad),
                   "lr": optimizer.param_groups[0]["lr"]}
            history.append(row)
            print(json.dumps(row), flush=True)
            if val < best:
                best = float(val)
                save(output / "best.pt", model, mean, std, positive[:2048],
                     negative[:2048], config, step, best)
    (output / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    save(output / "final.pt", model, mean, std, positive[:2048], negative[:2048],
         config, 299, best)

def save(path, model, mean, std, positive, negative, config, step, best):
    with torch.no_grad():
        first = model(positive)[0].mean(0)
        second = model(negative)[0].mean(0)
    torch.save({"model": model.state_dict(), "model_config": model.config,
                "mean": mean, "std": std, "direction": first - second,
                "config": config, "step": step, "best_val_loss": best}, path)

if __name__ == "__main__":
    main()
