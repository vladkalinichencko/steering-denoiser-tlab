"""One explicit data and training path for every activation model."""

import hashlib
import json
import math
import pathlib
import time

import torch

from tmp import methods

FINEWEB_REVISION = "9bb295d"
HOOK = "blocks.6.hook_resid_post"


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def collect_activations(path: pathlib.Path, n_train: int, n_val: int, seq_len=128) -> None:
    """Collect non-BOS activations and split complete documents by text hash."""
    from datasets import load_dataset
    import transformer_lens

    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", center_writing_weights=False, device=str(device()))
    stream = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train",
                          streaming=True, revision=FINEWEB_REVISION)
    split = {"train": [], "val": []}
    totals = {"train": 0, "val": 0}
    targets = {"train": n_train, "val": n_val}
    documents = {"train": 0, "val": 0}
    for example in stream:
        text = example["text"]
        part = "val" if hashlib.sha256(text.encode()).digest()[0] < 5 else "train"
        if totals[part] >= targets[part]:
            continue
        tokens = model.to_tokens(text)[:, :seq_len]
        if tokens.shape[1] < 8:
            continue
        _, cache = model.run_with_cache(tokens, names_filter=HOOK)
        activations = cache[HOOK][0, 1:].half().cpu()
        split[part].append(activations)
        totals[part] += len(activations)
        documents[part] += 1
        if sum(documents.values()) % 100 == 0:
            print(f"activations train={totals['train']} val={totals['val']}", flush=True)
        if all(totals[key] >= targets[key] for key in targets):
            break
    if not all(totals[key] >= targets[key] for key in targets):
        raise RuntimeError(f"FineWeb stream ended early: {totals}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train": torch.cat(split["train"])[:n_train],
                "val": torch.cat(split["val"])[:n_val],
                "meta": {"dataset": "HuggingFaceFW/fineweb", "subset": "sample-10BT",
                         "revision": FINEWEB_REVISION, "split": "sha256[0] < 5",
                         "documents": documents, "hook": HOOK, "bos": "excluded"}}, path)


def load_checkpoint(path: pathlib.Path, target_device: torch.device):
    blob = torch.load(path, map_location=target_device, weights_only=False)
    model = methods.ActivationModel(**blob["model_config"]).to(target_device)
    model.load_state_dict(blob["model"])
    return model.eval(), blob


@torch.no_grad()
def prepare_rectified_pairs(path: pathlib.Path, checkpoint: pathlib.Path,
                            n_train=4096, n_val=512, batch=64, seed=0) -> None:
    """Sample GLP endpoints once, then reuse them throughout Reflow training."""
    target_device = device()
    teacher, _ = load_checkpoint(checkpoint, target_device)
    generator = torch.Generator(device=target_device).manual_seed(seed)
    total = n_train + n_val
    endpoints, noises = [], []
    for start in range(0, total, batch):
        noise = torch.randn(min(batch, total - start), teacher.mean.numel(),
                            device=target_device, generator=generator)
        endpoint = methods.integrate(teacher, noise, 1.0, 20)[-1]
        endpoints.append(endpoint.half().cpu())
        noises.append(noise.half().cpu())
        print(f"rectified pairs {min(start + batch, total)}/{total}", flush=True)
    endpoint, noise = torch.cat(endpoints), torch.cat(noises)
    torch.save({"train": (endpoint[:n_train], noise[:n_train]),
                "val": (endpoint[n_train:], noise[n_train:]),
                "meta": {"teacher": str(checkpoint), "steps": 20,
                         "train": n_train, "val": n_val, "seed": seed}}, path)


def schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup - 1, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train(config: dict) -> pathlib.Path:
    target_device = device()
    torch.manual_seed(config["seed"])
    data = torch.load(config["data"], map_location="cpu", weights_only=False)
    train_data, val_data = data["train"].float(), data["val"].float()
    model = methods.build_model(config["method"], train_data.shape[1], config["reduced"])
    model.set_stats(train_data)
    model = model.to(target_device)
    ema = methods.make_ema(model) if config["method"] == "consistency" else None
    teacher = None
    if config["method"] == "rectified":
        teacher = (load_checkpoint(pathlib.Path(config["teacher"]), target_device)[0]
                   if "teacher" in config else None)
    pairs = (torch.load(config["pairs"], map_location="cpu", weights_only=False)
             if "pairs" in config else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    warmup = max(1, round(config["steps"] * 0.01))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: schedule(step, config["steps"], warmup))
    output = pathlib.Path("runs") / config["tag"]
    output.mkdir(parents=True, exist_ok=True)
    effective = {**config, "device": str(target_device), "dtype": "float32",
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "data_meta": data["meta"], "model": model.config}
    print(json.dumps(effective), flush=True)
    (output / "config.json").write_text(json.dumps(effective, indent=2))
    history = (output / "history.jsonl").open("w")
    best = float("inf")
    started = time.time()

    for step in range(config["steps"]):
        source = pairs["train"][0] if pairs else train_data
        index = torch.randint(len(source), (config["batch"],))
        batch = (model.standardize(train_data[index].to(target_device))
                 if pairs is None else None)
        pair = (tuple(values[index].float().to(target_device) for values in pairs["train"])
                if pairs else None)
        value = methods.loss(config["method"], model, batch, sigma=config["sigma"],
                             target_model=ema, teacher=teacher, pair=pair)
        value.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if ema is not None:
            methods.update_ema(ema, model)

        if step % config["log_every"] == 0 or step + 1 == config["steps"]:
            with torch.no_grad():
                val = (model.standardize(val_data.to(target_device))
                       if pairs is None else None)
                val_pair = (tuple(values.float().to(target_device) for values in pairs["val"])
                            if pairs else None)
                fixed = torch.Generator(device=target_device).manual_seed(10_000 + config["seed"])
                val_loss = methods.loss(config["method"], model, val, sigma=config["sigma"],
                                        target_model=ema, teacher=teacher,
                                        pair=val_pair,
                                        generator=fixed).item()
            row = {"step": step, "train_loss": value.item(), "val_loss": val_loss,
                   "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(grad),
                   "seconds": round(time.time() - started, 2)}
            if config["method"] == "meanflow":
                fixed = torch.Generator(device=target_device).manual_seed(10_000 + config["seed"])
                row["val_unweighted_mse"] = methods.loss(
                    "meanflow", model, val, generator=fixed, meanflow_weighted=False).item()
            history.write(json.dumps(row) + "\n")
            history.flush()
            print(json.dumps(row), flush=True)
            if val_loss < best:
                best = val_loss
                save_checkpoint(output / "best.pt", model, ema, effective, step, best)

    history.close()
    save_checkpoint(output / "final.pt", model, ema, effective, config["steps"] - 1, best)
    return output


def save_checkpoint(path, model, ema, config, step, best) -> None:
    chosen = ema if ema is not None else model
    torch.save({"model": chosen.state_dict(), "model_config": chosen.config,
                "config": config, "step": step, "best_val_loss": best}, path)
