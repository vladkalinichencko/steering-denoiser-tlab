"""One response-only Mac evaluation and one report for every steering method."""

import gc
import hashlib
import json
import math
import pathlib
import pickle
import subprocess
import time

import torch

import steering
from tmp import methods
from tmp.diagnostics import sentiment
from tmp.training import device, load_checkpoint

RUN = pathlib.Path("runs/mac_screening")
DATA = pathlib.Path("datasets/fineweb_layer6_mac_full.pt")
RATIOS = tuple(i / 5 for i in range(11))
K = 256
RANK = 16
PROMPTS = tuple(json.loads(line)["prompt"]["text"] for line in
                pathlib.Path("datasets/dexperts_neutral_prompts_100.jsonl").read_text().splitlines())
SEEDS = (0, 1)

CHECKPOINTS = {
    "Additive MSE simple": ("runs/mac_full_additive_simple/best.pt", 0.5, 1),
    "Additive MSE capacity": ("runs/mac_full_additive_capacity/best.pt", 0.5, 1),
    "Interpolation MSE": ("runs/mac_full_interpolation/best.pt", 0.5, 1),
    "GLP 20 steps": ("runs/mac_full_glp/best.pt", 0.2, 20),
    "GLP 20 steps t=0.35": ("runs/mac_full_glp/best.pt", 0.35, 20),
    "GLP 20 steps t=0.5": ("runs/mac_full_glp/best.pt", 0.5, 20),
    "GLP one Euler": ("runs/mac_full_glp/best.pt", 0.2, 1),
    "Consistency": ("runs/mac_reduced_consistency/best.pt", 0.2, 1),
    "Consistency t=0.35": ("runs/mac_reduced_consistency/best.pt", 0.35, 1),
    "Consistency t=0.5": ("runs/mac_reduced_consistency/best.pt", 0.5, 1),
    "Rectified 1 step": ("runs/mac_reduced_rectified/best.pt", 0.2, 1),
    "Rectified 2 steps": ("runs/mac_reduced_rectified/best.pt", 0.2, 2),
    "Rectified 4 steps": ("runs/mac_reduced_rectified/best.pt", 0.2, 4),
    "MeanFlow": ("runs/mac_reduced_meanflow/best.pt", 0.2, 1),
    "MeanFlow t=0.35": ("runs/mac_reduced_meanflow/best.pt", 0.35, 1),
    "MeanFlow t=0.5": ("runs/mac_reduced_meanflow/best.pt", 0.5, 1),
    "Tangent-preserving MSE": ("runs/mac_reduced_tangent_mse/best.pt", 0.5, 1),
}


def digest(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def synchronize(target: torch.device) -> None:
    if target.type == "cuda":
        torch.cuda.synchronize()
    elif target.type == "mps":
        torch.mps.synchronize()


def load_state(target: torch.device):
    data = torch.load(DATA, map_location="cpu", weights_only=False)
    normalizer, _ = load_checkpoint(
        pathlib.Path("runs/mac_full_additive_capacity/best.pt"), target)
    source = steering.load_model(str(target))
    vector = steering.diffmean_vector("sentiment", source, target)
    vector = vector / vector.norm()
    bank = normalizer.standardize(data["train"][:20_000].float().to(target))
    validation_raw = data["val"].float().to(target)
    heldout_raw = validation_raw[:8]
    heldout = normalizer.standardize(heldout_raw)
    reference = normalizer.standardize(validation_raw)
    centre = reference.mean(0)
    _, _, basis = torch.pca_lowrank(reference - centre, q=2)
    clean_geometry = methods.local_geometry(bank, heldout, K, RANK)
    scale = validation_raw.norm(dim=-1).mean()
    return data, normalizer, source, vector, bank, heldout_raw, heldout, centre, basis, \
        clean_geometry, scale


def checkpoint_mode(name: str, target: torch.device, vector: torch.Tensor):
    path, start, steps = CHECKPOINTS[name]
    model, checkpoint = load_checkpoint(pathlib.Path(path), target)

    def apply(h, strength, generator):
        edited = h + strength * vector
        fixed, states = methods.repair(
            checkpoint["config"]["method"], model, edited, t_start=start,
            steps=steps, generator=generator)
        return fixed, [h, edited] + [model.restore(state) for state in states]

    return {"apply": apply, "checkpoint": path,
            "parameters": checkpoint["config"]["parameters"]}


def geometry_modes(normalizer, bank, vector):
    direction = vector / normalizer.std

    def naive(h, strength, generator):
        edited = h + strength * vector
        return edited, [h, edited]

    def noise(h, strength, generator):
        perturbation = torch.randn(h.shape, device=h.device, generator=generator)
        perturbation = perturbation / perturbation.norm(dim=-1, keepdim=True) * strength
        return h + perturbation, [h, h + perturbation]

    def nearest(h, strength, generator):
        start = normalizer.standardize(h)
        edited = normalizer.standardize(h + strength * vector)
        fixed = methods.nearest(bank, edited)
        return normalizer.restore(fixed), [h, h + strength * vector, normalizer.restore(fixed)]

    def segment(h, strength, generator):
        start = normalizer.standardize(h)
        edited = normalizer.standardize(h + strength * vector)
        fixed = methods.segment_nearest(bank, start, edited)
        return normalizer.restore(fixed), [h, h + strength * vector, normalizer.restore(fixed)]

    def split(h, strength, part):
        start = normalizer.standardize(h)
        geometry = methods.local_geometry(bank, start, K, RANK)
        tangent, normal = methods.split_local(direction.expand_as(start), geometry["basis"])
        chosen = tangent if part == "tangent" else normal
        chosen = chosen / chosen.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        displacement = strength * direction.norm() * chosen
        fixed = normalizer.restore(start + displacement)
        return fixed, [h, fixed]

    def tangent(h, strength, generator):
        return split(h, strength, "tangent")

    def normal(h, strength, generator):
        return split(h, strength, "normal")

    def geodesic(h, strength, generator):
        start = normalizer.standardize(h)
        distance = float(strength * direction.norm())
        fixed, states = methods.local_geodesic(
            bank, start, direction / direction.norm(), distance, 8, K, RANK)
        return normalizer.restore(fixed), [normalizer.restore(state) for state in states]

    return {
        "Naive": {"apply": naive, "checkpoint": None, "parameters": 0},
        "Isotropic noise": {"apply": noise, "checkpoint": None, "parameters": 0},
        "Nearest activation": {"apply": nearest, "checkpoint": None, "parameters": 0},
        "Segment-kNN": {"apply": segment, "checkpoint": None, "parameters": 0},
        "Local tangent": {"apply": tangent, "checkpoint": None, "parameters": 0},
        "Local normal": {"apply": normal, "checkpoint": None, "parameters": 0},
        "Local geodesic": {"apply": geodesic, "checkpoint": None, "parameters": 0},
    }


def safe_mode(capacity, vector):
    unit = vector / vector.norm()

    def apply(h, strength, generator):
        edited = h + strength * vector
        fixed, _ = methods.repair("additive_capacity", capacity, edited)
        correction = fixed - edited
        correction = correction - (correction @ unit)[:, None] * unit
        fixed = edited + correction
        return fixed, [h, edited, fixed]

    return {"apply": apply, "checkpoint": "runs/mac_full_additive_capacity/best.pt",
            "parameters": sum(parameter.numel() for parameter in capacity.parameters())}


def curveball_mode(path: pathlib.Path):
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


def inn_mode(path: pathlib.Path, target: torch.device):
    from tmp.nonlinear import INNSteer
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


def unisteer_mode(path: pathlib.Path, target: torch.device):
    from tmp.nonlinear import ConditionalFlow, conditional_transport
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


def geometry(candidate, clean, normalizer, bank, clean_geometry, capacity):
    candidate_z = normalizer.standardize(candidate)
    current = methods.local_geometry(bank, candidate_z, K, RANK)
    tangent, normal = methods.split_local(candidate_z - clean, clean_geometry["basis"])
    energy = capacity(candidate_z).norm(dim=-1)
    return {"knn_distance": float(current["distance"].mean()),
            "local_pca_residual": float(current["residual"].mean()),
            "denoiser_energy": float(energy.mean()),
            "tangent_displacement": float(tangent.norm(dim=-1).mean()),
            "normal_displacement": float(normal.norm(dim=-1).mean()),
            "spectrum": current["spectrum"][0].cpu().tolist()}


def diversity(texts, n):
    grams, total = set(), 0
    for text in texts:
        words = text.lower().split()
        current = list(zip(*(words[i:] for i in range(n))))
        grams.update(current)
        total += len(current)
    return len(grams) / max(total, 1)


def hook(apply, strength, target, seed):
    generator = torch.Generator(device=target).manual_seed(seed)

    def intervention(residual, hook=None):
        output = residual.clone()
        output[:, -1] = apply(output[:, -1], strength, generator)[0]
        return output

    return intervention


@torch.no_grad()
def generate_many(model, prompts, apply, strength, target, seed):
    groups = {}
    for index, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        groups.setdefault(tokens.shape[1], []).append((index, tokens))
    output = [None] * len(prompts)
    for length, rows in groups.items():
        for start in range(0, len(rows), 16):
            current = rows[start:start + 16]
            sequence = torch.cat([tokens for _, tokens in current])
            torch.manual_seed(seed)
            intervention = hook(apply, strength, target, seed + length + start)
            with model.hooks(fwd_hooks=[(steering.HOOK, intervention)]):
                sequence = model.generate(sequence, max_new_tokens=20, do_sample=True,
                                          temperature=1.0, top_k=50, verbose=False,
                                          use_past_kv_cache=True)
            for row, (index, _) in zip(sequence, current):
                output[index] = model.to_string(row[length:])
    return output


@torch.no_grad()
def nll_many(model, prompts, texts):
    groups = {}
    for prompt, text in zip(prompts, texts):
        tokens = model.to_tokens(prompt + text)
        prompt_length = model.to_tokens(prompt).shape[1]
        groups.setdefault(tokens.shape[1], []).append((tokens, prompt_length))
    values = []
    for rows in groups.values():
        for start in range(0, len(rows), 32):
            current = rows[start:start + 32]
            losses = model(torch.cat([tokens for tokens, _ in current]),
                           return_type="loss", loss_per_token=True)
            values.extend(float(loss[prompt_length - 1:].mean())
                          for loss, (_, prompt_length) in zip(losses, current))
    return values


def bootstrap(values, seed=0):
    sample = torch.tensor(values)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(len(sample), (1_000, len(sample)), generator=generator)
    means = sample[index].mean(1)
    return [float(value) for value in torch.quantile(means, torch.tensor([0.025, 0.975]))]


@torch.no_grad()
def downstream(model, apply, strength, target):
    tokens = model.to_tokens(PROMPTS[0])
    names = [f"blocks.{layer}.hook_resid_post" for layer in range(7, 12)]
    clean_logits, clean = model.run_with_cache(tokens, names_filter=lambda name: name in names)
    intervention = hook(apply, strength, target, 0)
    with model.hooks(fwd_hooks=[(steering.HOOK, intervention)]):
        edited_logits, edited = model.run_with_cache(
            tokens, names_filter=lambda name: name in names)
    first = clean_logits[0, -1].log_softmax(-1)
    second = edited_logits[0, -1].log_softmax(-1)
    return {"layers": list(range(7, 12)),
            "drift": [float((edited[name][0, -1] - clean[name][0, -1]).norm())
                      for name in names],
            "logit_kl": float((first.exp() * (first - second)).sum()),
            "clean_top": model.to_string(first.topk(10).indices),
            "edited_top": model.to_string(second.topk(10).indices)}


@torch.no_grad()
def evaluate(name, mode, ratio, state, capacity):
    data, normalizer, source, vector, bank, heldout_raw, heldout, centre, basis, \
        clean_geometry, scale = state
    target = heldout.device
    strength = ratio * scale
    generator = torch.Generator(device=target).manual_seed(0)
    synchronize(target)
    started = time.perf_counter()
    candidate, path = mode["apply"](heldout_raw, strength, generator)
    synchronize(target)
    elapsed = (time.perf_counter() - started) * 1000 / len(heldout_raw)
    texts = []
    for seed in SEEDS:
        texts.extend(generate_many(source, PROMPTS, mode["apply"], strength, target, seed))
    text_prompts = PROMPTS * len(SEEDS)
    scores = sentiment(texts)
    nll_values = nll_many(source, text_prompts, texts)
    nll = sum(nll_values) / len(nll_values)
    project = lambda values: ((normalizer.standardize(values) - centre) @ basis).cpu().tolist()
    own = ([mode["coordinates"](value[:1])[0] for value in path]
           if "coordinates" in mode else None)
    slug = "".join(value.lower() if value.isalnum() else "_" for value in name).strip("_")
    state_path = pathlib.Path("states") / f"{slug}_{ratio:.1f}.pt"
    (RUN / "states").mkdir(parents=True, exist_ok=True)
    torch.save(torch.stack([value.detach().half().cpu() for value in path]), RUN / state_path)
    return {"method": name, "ratio": ratio, "nll": nll, "ppl": math.exp(nll),
            "property": sum(scores) / len(scores),
            "property_scores": scores, "nll_scores": nll_values,
            "uncertainty": {"property": bootstrap(scores), "nll": bootstrap(nll_values)},
            "dist1": diversity(texts, 1), "dist2": diversity(texts, 2),
            "dist3": diversity(texts, 3), "latency_ms": elapsed,
            "geometry": geometry(candidate, heldout, normalizer, bank,
                                 clean_geometry, capacity),
            "trajectory": project(torch.cat([value[:1] for value in path])),
            "method_coordinates": own,
            "jacobian_spectrum": mode["jacobian"](heldout_raw[:1], strength)
            if "jacobian" in mode else None,
            "checkpoint": mode["checkpoint"], "parameters": mode["parameters"],
            "states": state_path.as_posix(),
            "texts": texts}


def save(artifact):
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "screening.json").write_text(json.dumps(artifact, indent=2))
    report = pathlib.Path(__file__).with_name("screening_report.html").read_text()
    (RUN / "screening.html").write_text(
        report.replace("__DATA__", json.dumps(artifact).replace("</", "<\\/")))


def training_histories():
    output = {}
    for name, (checkpoint, _, _) in CHECKPOINTS.items():
        history = pathlib.Path(checkpoint).parent / "history.jsonl"
        if history.as_posix() in (run["source"] for run in output.values()):
            continue
        rows = [json.loads(line) for line in history.read_text().splitlines()]
        output[name] = {"key": "val_unweighted_mse" if name == "MeanFlow" else "val_loss",
                        "rows": rows, "source": history.as_posix()}
    for name, directory in (("INNSteer", "runs/mac_reduced_inn"),
                            ("Conditional field / UniSteer", "runs/mac_reduced_unisteer")):
        history = pathlib.Path(directory) / "history.jsonl"
        if history.exists():
            output[name] = {"key": "val_loss",
                            "rows": [json.loads(line) for line in history.read_text().splitlines()],
                            "source": history.as_posix()}
    return output


def run(selected=None, ratios=RATIOS):
    target = device()
    state = load_state(target)
    data, normalizer, source, vector, bank, heldout_raw, heldout, centre, basis, \
        clean_geometry, scale = state
    capacity, _ = load_checkpoint(
        pathlib.Path("runs/mac_full_additive_capacity/best.pt"), target)
    geometry = geometry_modes(normalizer, bank, vector)
    geometry["Safe capacity MSE"] = safe_mode(capacity, vector)
    curveball = pathlib.Path("runs/mac_reduced_curveball/model.pkl")
    if curveball.exists():
        geometry["Curveball"] = curveball_mode(curveball)
    inn = pathlib.Path("runs/mac_reduced_inn/best.pt")
    if inn.exists():
        geometry["INNSteer"] = inn_mode(inn, target)
    unisteer = pathlib.Path("runs/mac_reduced_unisteer/best.pt")
    if unisteer.exists():
        geometry["Conditional field / UniSteer"] = unisteer_mode(unisteer, target)
    chosen = tuple(selected) if selected is not None else tuple(CHECKPOINTS) + tuple(geometry)
    checkpoint_paths = {CHECKPOINTS[name][0] for name in chosen if name in CHECKPOINTS}
    checkpoint_paths.add("runs/mac_full_additive_capacity/best.pt")
    checkpoint_paths.update(mode["checkpoint"] for name, mode in geometry.items()
                            if name in chosen and mode["checkpoint"])
    contract = {"hook": steering.HOOK, "scope": "response tokens",
                "ratios": ratios, "prompts": PROMPTS, "seeds": SEEDS,
                "direction": f"SetFit/sst5 train@{steering.SST5_REVISION}; labels 3/4 minus 0/1",
                "data": data["meta"], "bank": len(bank), "k": K, "rank": RANK,
                "alpha_scale": "mean norm over full denoiser validation split",
                "projection": "fixed clean validation PCA",
                "provenance": {
                    "git_revision": subprocess.check_output(
                        ("git", "rev-parse", "HEAD"), text=True).strip(),
                    "sources": {path: digest(path) for path in
                                ("steering.py", "tmp/methods.py", "tmp/nonlinear.py",
                                 "tmp/screening.py")},
                    "checkpoints": {path: digest(path) for path in sorted(checkpoint_paths)}}}
    path = RUN / "screening.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    contract = json.loads(json.dumps(contract))
    compatible = False
    if previous:
        old = json.loads(json.dumps(previous["contract"]))
        old["provenance"].pop("git_revision", None)
        current = json.loads(json.dumps(contract))
        current["provenance"].pop("git_revision", None)
        compatible = old == current
    artifact = (previous if compatible else
                {"background": ((heldout - centre) @ basis).cpu().tolist(),
                 "points": []})
    artifact["contract"] = contract
    completed = {(point["method"], point["ratio"]) for point in artifact["points"]}
    artifact["training"] = training_histories()
    save(artifact)
    print(RUN / "screening.html", flush=True)
    for name in chosen:
        mode = checkpoint_mode(name, target, vector) if name in CHECKPOINTS else geometry[name]
        for ratio in ratios:
            if (name, ratio) in completed:
                continue
            point = evaluate(name, mode, ratio, state, capacity)
            artifact["points"].append(point)
            save(artifact)
            print(json.dumps({key: point[key] for key in
                              ("method", "ratio", "nll", "property", "latency_ms")}), flush=True)
        del mode
        gc.collect()


@torch.no_grad()
def enrich():
    """Add full states and method-specific diagnostics without regenerating text."""
    target = device()
    state = load_state(target)
    _, normalizer, source, vector, bank, heldout_raw, _, _, _, _, _ = state
    capacity, _ = load_checkpoint(
        pathlib.Path("runs/mac_full_additive_capacity/best.pt"), target)
    extra = geometry_modes(normalizer, bank, vector)
    extra["Safe capacity MSE"] = safe_mode(capacity, vector)
    paths = {"Curveball": pathlib.Path("runs/mac_reduced_curveball/model.pkl"),
             "INNSteer": pathlib.Path("runs/mac_reduced_inn/best.pt"),
             "Conditional field / UniSteer": pathlib.Path("runs/mac_reduced_unisteer/best.pt")}
    if paths["Curveball"].exists():
        extra["Curveball"] = curveball_mode(paths["Curveball"])
    if paths["INNSteer"].exists():
        extra["INNSteer"] = inn_mode(paths["INNSteer"], target)
    if paths["Conditional field / UniSteer"].exists():
        extra["Conditional field / UniSteer"] = unisteer_mode(
            paths["Conditional field / UniSteer"], target)
    artifact = json.loads((RUN / "screening.json").read_text())
    scale = state[-1]
    names = [point["method"] for point in artifact["points"]]
    for name in dict.fromkeys(names):
        mode = checkpoint_mode(name, target, vector) if name in CHECKPOINTS else extra[name]
        for point in (row for row in artifact["points"] if row["method"] == name):
            strength = point["ratio"] * scale
            _, path = mode["apply"](
                heldout_raw, strength, torch.Generator(device=target).manual_seed(0))
            slug = "".join(value.lower() if value.isalnum() else "_"
                           for value in name).strip("_")
            state_path = pathlib.Path("states") / f"{slug}_{point['ratio']:.1f}.pt"
            (RUN / "states").mkdir(parents=True, exist_ok=True)
            torch.save(torch.stack([value.detach().half().cpu() for value in path]),
                       RUN / state_path)
            point.update({"states": state_path.as_posix(),
                          "checkpoint": mode["checkpoint"],
                          "parameters": mode["parameters"],
                          "method_coordinates": ([mode["coordinates"](value[:1])[0]
                                                  for value in path]
                                                 if "coordinates" in mode else None),
                          "jacobian_spectrum": (mode["jacobian"](
                              heldout_raw[:1], strength) if "jacobian" in mode else None),
                          "downstream": downstream(source, mode["apply"], strength, target)})
        save(artifact)
        del mode
        gc.collect()
    artifact["training"] = training_histories()
    save(artifact)




if __name__ == "__main__":
    run()
