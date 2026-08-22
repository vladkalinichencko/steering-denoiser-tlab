"""One response-only Mac evaluation and one report for every steering method."""

import gc
import json
import math
import pathlib
import pickle
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
    "GLP one Euler": ("runs/mac_full_glp/best.pt", 0.2, 1),
    "Consistency": ("runs/mac_reduced_consistency/best.pt", 0.2, 1),
    "Rectified 1 step": ("runs/mac_reduced_rectified/best.pt", 0.2, 1),
    "Rectified 2 steps": ("runs/mac_reduced_rectified/best.pt", 0.2, 2),
    "Rectified 4 steps": ("runs/mac_reduced_rectified/best.pt", 0.2, 4),
    "MeanFlow": ("runs/mac_reduced_meanflow/best.pt", 0.2, 1),
    "Tangent-preserving MSE": ("runs/mac_reduced_tangent_mse/best.pt", 0.5, 1),
}


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

    def apply(h, strength, generator):
        source = ((h.detach().cpu() - mean) / std).float()

        def restore(amount):
            return model.steer(source, amount) * std + mean

        if strength == 0:
            fixed = h
        else:
            low, high = 0.0, 1.0
            while (restore(high) - h.cpu()).norm(dim=-1).mean() < strength and high < 128:
                high *= 2
            for _ in range(10):
                middle = (low + high) / 2
                if (restore(middle) - h.cpu()).norm(dim=-1).mean() < strength:
                    low = middle
                else:
                    high = middle
            fixed = restore((low + high) / 2).to(h.device)
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

    def raw(h, amount):
        latent = model((h - mean) / std)[0] + amount * direction
        return model(latent, inverse=True)[0] * std + mean

    def apply(h, strength, generator):
        low, high = 0.0, 1.0
        if strength:
            while (raw(h, high) - h).norm(dim=-1).mean() < strength and high < 128:
                high *= 2
            for _ in range(10):
                middle = (low + high) / 2
                if (raw(h, middle) - h).norm(dim=-1).mean() < strength:
                    low = middle
                else:
                    high = middle
        fixed = h if strength == 0 else raw(h, (low + high) / 2)
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

    def endpoint(h, amount, path=False):
        return conditional_transport(model, h, null, positive, amount, 10, path)

    def apply(h, strength, generator):
        low, high = 0.0, 1.0
        if strength:
            for _ in range(10):
                middle = (low + high) / 2
                if (endpoint(h, middle) - h).norm(dim=-1).mean() < strength:
                    low = middle
                else:
                    high = middle
        amount = 0 if strength == 0 else (low + high) / 2
        fixed, states = endpoint(h, amount, True)
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
            for _ in range(20):
                with model.hooks(fwd_hooks=[(steering.HOOK, intervention)]):
                    logits = model(sequence)[:, -1]
                top = logits.topk(50)
                choice = torch.multinomial(top.values.softmax(-1), 1)
                sequence = torch.cat((sequence, top.indices.gather(1, choice)), dim=1)
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
    (RUN / "screening.html").write_text(
        REPORT.replace("__DATA__", json.dumps(artifact).replace("</", "<\\/")))


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
    contract = {"hook": steering.HOOK, "scope": "response tokens",
                "ratios": ratios, "prompts": PROMPTS, "seeds": SEEDS,
                "direction": f"SetFit/sst5 train@{steering.SST5_REVISION}; labels 3/4 minus 0/1",
                "data": data["meta"], "bank": len(bank), "k": K, "rank": RANK,
                "alpha_scale": "mean norm over full denoiser validation split",
                "projection": "fixed clean validation PCA"}
    path = RUN / "screening.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    contract = json.loads(json.dumps(contract))
    artifact = (previous if previous and previous["contract"] == contract
                else {"contract": contract,
                      "background": ((heldout - centre) @ basis).cpu().tolist(),
                      "points": []})
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


REPORT = r'''<!doctype html><meta charset="utf-8"><title>Mac steering screening</title>
<style>
:root{--fg:#17202a;--mut:#687078;--line:#d7dce1;--blue:#2563eb;--orange:#d97706}
*{box-sizing:border-box}body{font:14px/1.45 system-ui;margin:0;color:var(--fg)}
header{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid var(--line);padding:10px 18px;display:flex;gap:18px;align-items:center}
h1{font-size:16px;margin:0}select,input{font:inherit}.readout{margin-left:auto;font-variant-numeric:tabular-nums}
main{padding:18px;display:grid;grid-template-columns:minmax(560px,1.4fr) minmax(380px,1fr);gap:18px}.wide{grid-column:1/-1}
section{border-top:1px solid var(--line);padding-top:10px}h2{font-size:14px;margin:0 0 8px}.plots{display:flex;flex-wrap:wrap;gap:12px}
svg{width:100%;height:auto;background:#fafbfc}.small{width:320px}.large{min-height:430px}.axis{stroke:#aab1b8;stroke-width:1}.cloud{fill:#aab1b8;opacity:.28}.path{fill:none;stroke:var(--blue);stroke-width:1.5}.naive{stroke:var(--orange)}
.caption{color:var(--mut);margin:5px 0}.legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin:6px 0;color:var(--mut)}.swatch{display:inline-block;width:9px;height:9px;margin-right:4px}.sample{border-top:1px solid var(--line);padding:8px 0}.prompt{font-weight:600}.metric{font-variant-numeric:tabular-nums}
@media(max-width:960px){main{grid-template-columns:1fr}.wide{grid-column:auto}}
</style>
<header><h1>Steering Mac screening</h1><label>method <select id="method"></select></label><label>alpha <input id="ratio" type="range" min="0" max="0" step="1"></label><a id="state-link">states</a><span class="readout" id="readout"></span></header>
<main><section class="wide"><h2>Pareto: clean-model NLL and positive sentiment</h2><div class="legend" id="legend"></div><svg id="pareto" viewBox="0 0 1080 360"></svg></section>
<section class="wide"><h2>Quality, property and diversity over alpha</h2><div class="plots" id="quality"></div></section>
<section><h2>Fixed clean-validation coordinates</h2><svg class="large" id="trajectory" viewBox="0 0 620 430"></svg><p class="caption">Grey is the same held-out reference cloud for every method. Orange is clean to naive. Blue is the selected method path.</p></section>
<section><h2>Full-dimensional geometry</h2><div class="plots" id="geometry"></div></section>
<section id="intrinsic-section"><h2>Method coordinates</h2><svg class="large" id="intrinsic" viewBox="0 0 620 430"></svg><p class="caption">The selected method's own first two coordinates; common conclusions still use decoded and full-dimensional axes.</p></section>
<section class="wide"><h2>Training dynamics</h2><div class="plots" id="training"></div></section>
<section class="wide"><h2>Decoded continuations</h2><div id="samples"></div></section></main>
<script>
const data=__DATA__, points=data.points, methods=[...new Set(points.map(x=>x.method))];
const colors=["#2563eb","#d97706","#059669","#7c3aed","#dc2626","#0891b2","#4f46e5","#65a30d","#9333ea","#ea580c","#0f766e","#be123c","#475569","#15803d","#6d28d9","#0369a1","#a16207"];
const color=Object.fromEntries(methods.map((x,i)=>[x,colors[i%colors.length]]));
const S="http://www.w3.org/2000/svg"; function node(tag,a={},text=""){const n=document.createElementNS(S,tag);for(const k in a)n.setAttribute(k,a[k]);if(text)n.textContent=text;return n}
function bounds(values){let lo=Math.min(...values),hi=Math.max(...values);if(lo===hi){lo-=.5;hi+=.5}const p=(hi-lo)*.08;return[lo-p,hi+p]}
function chart(svg,series,xkey,ykey){svg.replaceChildren();const all=series.flatMap(s=>s.rows);if(!all.length)return;const xb=bounds(all.map(x=>x[xkey])),yb=bounds(all.map(x=>x[ykey]));const W=+svg.viewBox.baseVal.width,H=+svg.viewBox.baseVal.height,p=42,X=x=>p+(x-xb[0])/(xb[1]-xb[0])*(W-p-12),Y=y=>H-p-(y-yb[0])/(yb[1]-yb[0])*(H-p-12);svg.append(node("line",{x1:p,y1:8,x2:p,y2:H-p,class:"axis"}),node("line",{x1:p,y1:H-p,x2:W-8,y2:H-p,class:"axis"}),node("text",{x:p,y:H-8,fill:"#687078","font-size":10},xb[0].toPrecision(3)),node("text",{x:W-10,y:H-8,fill:"#687078","font-size":10,"text-anchor":"end"},xb[1].toPrecision(3)),node("text",{x:p-5,y:14,fill:"#687078","font-size":10,"text-anchor":"end"},yb[1].toPrecision(3)),node("text",{x:p-5,y:H-p,fill:"#687078","font-size":10,"text-anchor":"end"},yb[0].toPrecision(3)));for(const s of series){const rows=s.rows.slice().sort((a,b)=>a.ratio-b.ratio);svg.append(node("path",{d:rows.map((r,i)=>(i?"L":"M")+X(r[xkey])+","+Y(r[ykey])).join(" "),fill:"none",stroke:s.color,"stroke-width":"1.5"}));for(const r of rows){const c=node("circle",{cx:X(r[xkey]),cy:Y(r[ykey]),r:3,fill:s.color});c.append(node("title",{},`${r.method||"spectrum"} r=${r.ratio}: ${xkey}=${r[xkey].toFixed(3)}, ${ykey}=${r[ykey].toFixed(3)}`));svg.append(c)}}}
function metric(name,key,rows,xkey="ratio"){const box=document.createElement("div"),title=document.createElement("div"),svg=node("svg",{viewBox:"0 0 320 190",class:"small"});title.textContent=name;box.append(title,svg);chart(svg,[{rows,color:"#2563eb"}],xkey,key);return box}
function project(svg,background,path,naive){svg.replaceChildren();const all=background.concat(path,naive),xb=bounds(all.map(x=>x[0])),yb=bounds(all.map(x=>x[1])),X=x=>42+(x-xb[0])/(xb[1]-xb[0])*560,Y=y=>394-(y-yb[0])/(yb[1]-yb[0])*370;svg.append(node("line",{x1:42,y1:14,x2:42,y2:394,class:"axis"}),node("line",{x1:42,y1:394,x2:608,y2:394,class:"axis"}),node("text",{x:608,y:416,fill:"#687078","font-size":11,"text-anchor":"end"},"PC1"),node("text",{x:8,y:16,fill:"#687078","font-size":11},"PC2"));for(const p of background)svg.append(node("circle",{cx:X(p[0]),cy:Y(p[1]),r:1.5,class:"cloud"}));for(const [values,cls] of [[naive,"path naive"],[path,"path"]]){svg.append(node("polyline",{points:values.map(p=>X(p[0])+","+Y(p[1])).join(" "),class:cls}));values.forEach((p,i)=>svg.append(node("circle",{cx:X(p[0]),cy:Y(p[1]),r:3,fill:cls.includes("naive")?"#d97706":"#2563eb"})))}}
const method=document.querySelector("#method"),slider=document.querySelector("#ratio");methods.forEach(x=>method.append(new Option(x,x)));
function render(){const rows=points.filter(x=>x.method===method.value).sort((a,b)=>a.ratio-b.ratio);slider.max=Math.max(0,rows.length-1);const row=rows[+slider.value]||rows[0];if(!row)return;const naive=points.find(x=>x.method==="Naive"&&x.ratio===row.ratio),ci=row.uncertainty;document.querySelector("#state-link").href=row.states;document.querySelector("#readout").textContent=`r ${row.ratio.toFixed(1)}  NLL ${row.nll.toFixed(3)} [${ci.nll.map(x=>x.toFixed(3)).join(", ")}]  property ${row.property.toFixed(3)} [${ci.property.map(x=>x.toFixed(3)).join(", ")}]  kNN ${row.geometry.knn_distance.toFixed(2)}  residual ${row.geometry.local_pca_residual.toFixed(2)}`;project(document.querySelector("#trajectory"),data.background,row.trajectory,naive?naive.trajectory:row.trajectory);const own=document.querySelector("#intrinsic-section");own.style.display=row.method_coordinates?"block":"none";if(row.method_coordinates)project(document.querySelector("#intrinsic"),[],row.method_coordinates,row.method_coordinates.slice(0,1));const q=document.querySelector("#quality");q.replaceChildren();[["NLL","nll"],["positive probability","property"],["dist-1","dist1"],["dist-2","dist2"],["dist-3","dist3"]].forEach(([n,k])=>q.append(metric(n,k,rows)));const g=document.querySelector("#geometry");g.replaceChildren();const flat=rows.map(x=>Object.assign({ratio:x.ratio},x.geometry));[["kNN distance","knn_distance"],["local PCA residual","local_pca_residual"],["capacity-MSE energy","denoiser_energy"],["tangent displacement","tangent_displacement"],["normal displacement","normal_displacement"],["latency ms","latency_ms"]].forEach(([n,k])=>g.append(metric(n,k,k==="latency_ms"?rows:flat)));g.append(metric("singular spectrum","value",row.geometry.spectrum.map((value,ratio)=>({ratio,value}))));if(row.jacobian_spectrum)g.append(metric("edit Jacobian spectrum","value",row.jacobian_spectrum.map((value,ratio)=>({ratio,value}))));const samples=document.querySelector("#samples");samples.replaceChildren();row.texts.slice(0,20).forEach((text,i)=>{const d=document.createElement("div");d.className="sample";d.innerHTML=`<div class="prompt"></div><div></div>`;d.children[0].textContent=data.contract.prompts[i%data.contract.prompts.length];d.children[1].textContent=text;samples.append(d)})}
method.onchange=()=>{slider.value=0;render()};slider.oninput=render;const legend=document.querySelector("#legend");methods.forEach(name=>{const item=document.createElement("span");item.innerHTML=`<i class="swatch"></i>${name}`;item.firstChild.style.background=color[name];legend.append(item)});chart(document.querySelector("#pareto"),methods.map(x=>({rows:points.filter(p=>p.method===x),color:color[x]})),"nll","property");const training=document.querySelector("#training");for(const [name,run] of Object.entries(data.training)){training.append(metric(`${name} validation`,run.key,run.rows,"step"));if(run.rows[0].grad_norm!==undefined)training.append(metric(`${name} gradient norm`,"grad_norm",run.rows,"step"));training.append(metric(`${name} learning rate`,"lr",run.rows,"step"))}render();
</script>'''


if __name__ == "__main__":
    run()
