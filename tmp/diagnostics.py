"""Real-activation trajectories and decoded causal checks for one Mac run."""

import html
import json
import math
import pathlib
import sys

import torch

import steering
from tmp import methods
from tmp.training import device, load_checkpoint

PROMPTS = ["The weather today is", "I spent the afternoon", "The report concluded that"]


def response_hook(vector, alpha, repair_fn=None):
    """Change only the last position used to predict the next response token."""
    def hook(residual, hook=None):
        output = residual.clone()
        edited = output[:, -1] + alpha * vector
        output[:, -1] = repair_fn(edited) if repair_fn else edited
        return output
    return hook


@torch.no_grad()
def generate(model, prompt, hook, seed, tokens=20):
    torch.manual_seed(seed)
    sequence = model.to_tokens(prompt)
    prompt_length = sequence.shape[1]
    for _ in range(tokens):
        with model.hooks(fwd_hooks=[(steering.HOOK, hook)]):
            logits = model(sequence)[:, -1]
        top = logits.topk(50)
        choice = torch.multinomial(top.values.softmax(-1), 1)
        sequence = torch.cat((sequence, top.indices.gather(1, choice)), dim=1)
    return model.to_string(sequence[0, prompt_length:])


@torch.no_grad()
def continuation_nll(model, prompt, continuation):
    tokens = model.to_tokens(prompt + continuation)
    prompt_tokens = model.to_tokens(prompt).shape[1]
    losses = model(tokens, return_type="loss", loss_per_token=True)[0]
    return float(losses[prompt_tokens - 1:].mean())


_judge = None


def sentiment(texts):
    from transformers import pipeline
    global _judge
    if _judge is None:
        _judge = pipeline("sentiment-analysis",
                          model="distilbert-base-uncased-finetuned-sst-2-english",
                          device=-1, truncation=True)
    rows = _judge([text or "." for text in texts])
    return [row["score"] if row["label"] == "POSITIVE" else 1 - row["score"] for row in rows]


def points(values, width, height, pad=30):
    xs = [value[0] for value in values]
    ys = [value[1] for value in values]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    return [(pad + (x - xmin) / dx * (width - 2 * pad),
             height - pad - (y - ymin) / dy * (height - 2 * pad)) for x, y in values]


def line_chart(rows, key, width=520, height=230):
    scaled = points([(row["step"], row[key]) for row in rows], width, height)
    path = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}"
                    for i, (x, y) in enumerate(scaled))
    return f'<svg viewBox="0 0 {width} {height}"><path d="{path}"/></svg>'


def trajectory_svg(background, paths, width=520, height=390):
    all_values = background + [point for path in paths.values() for point in path]
    scaled = points(all_values, width, height)
    background_scaled = scaled[:len(background)]
    cursor = len(background)
    body = "".join(f'<circle class="cloud" cx="{x:.1f}" cy="{y:.1f}" r="1.4"/>'
                   for x, y in background_scaled)
    colours = ["#d97706", "#2563eb", "#059669", "#7c3aed"]
    for colour, (name, path) in zip(colours, paths.items()):
        current = scaled[cursor:cursor + len(path)]
        cursor += len(path)
        coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in current)
        body += f'<polyline points="{coordinates}" style="stroke:{colour}"/>'
        for i, (x, y) in enumerate(current):
            body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" style="fill:{colour}"><title>{html.escape(name)} step {i}</title></circle>'
    return f'<svg viewBox="0 0 {width} {height}">{body}</svg>'


@torch.no_grad()
def diagnose(tag: str) -> pathlib.Path:
    target_device = device()
    run = pathlib.Path("runs") / tag
    model, checkpoint = load_checkpoint(run / "best.pt", target_device)
    method = checkpoint["config"]["method"]
    data = torch.load(checkpoint["config"]["data"], map_location="cpu", weights_only=False)
    heldout = data["val"].float()[:512].to(target_device)
    source_model = steering.load_model(str(target_device))
    vector = steering.diffmean_vector("sentiment", source_model, target_device)
    vector = vector / vector.norm()
    alpha = 0.8 * heldout.norm(dim=-1).mean()

    clean = heldout[0:1]
    edited = clean + alpha * vector
    preview_start = 0.2 if method in {"glp", "consistency", "rectified", "meanflow"} else 0.5
    preview_steps = 1 if method == "rectified" else 20
    fixed, path = methods.repair(
        method, model, edited, t_start=preview_start, steps=preview_steps,
        generator=torch.Generator(device=target_device).manual_seed(0))
    standardized = model.standardize(heldout)
    centre = standardized.mean(0)
    _, _, basis = torch.pca_lowrank(standardized - centre, q=2)
    project = lambda value: ((value - centre) @ basis).cpu().tolist()
    path_raw = [model.restore(value) for value in path]
    paths = {"clean→steered": project(model.standardize(torch.cat((clean, edited)))),
             method: project(torch.cat(path))}
    background = project(standardized)[::4]

    unit = vector
    correction = fixed - edited
    geometry = {"alpha": float(alpha), "clean_norm": float(clean.norm()),
                "steered_distance": float((edited - clean).norm()),
                "repair_residual": float(correction.norm()),
                "distance_after_repair": float((fixed - clean).norm()),
                "steering_coordinate_after_repair": float(((fixed - clean) @ unit).item()),
                "path_steps": len(path_raw) - 1}

    generations = []
    for ratio in (0.0, 0.8, 1.6):
        strength = ratio * heldout.norm(dim=-1).mean()
        modes = [("clean", None, None)] if ratio == 0 else [("naive", None, None)]
        if ratio:
            if method == "rectified":
                modes += [(f"repair_{steps}step", 0.2, steps) for steps in (1, 2, 4)]
            else:
                starts = ((0.2, 0.35, 0.5)
                          if method in {"glp", "consistency", "meanflow"} else (0.5,))
                modes += [(f"repair_t{start:g}", start, 20) for start in starts]
        for mode, t_start, steps in modes:
            texts = []
            for i, prompt in enumerate(PROMPTS):
                repair_fn = None
                if t_start is not None:
                    noise = torch.Generator(device=target_device).manual_seed(i)
                    repair_fn = lambda value, noise=noise, start=t_start, count=steps: methods.repair(
                        method, model, value, t_start=start, steps=count, generator=noise)[0]
                texts.append(generate(source_model, prompt,
                                      response_hook(vector, strength, repair_fn), seed=i))
            scores = sentiment(texts)
            generations.append({"ratio": ratio, "mode": mode, "texts": texts,
                                "positive_probability": sum(scores) / len(scores),
                                "nll": sum(continuation_nll(source_model, p, text)
                                           for p, text in zip(PROMPTS, texts)) / len(texts)})

    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    loss_key = "val_unweighted_mse" if method == "meanflow" else "val_loss"
    artifact = {"tag": tag, "method": method, "checkpoint": str(run / "best.pt"),
                "checkpoint_step": checkpoint["step"], "data": data["meta"],
                "geometry": geometry, "paths": paths, "generations": generations}
    (run / "diagnostics.json").write_text(json.dumps(artifact, indent=2))
    rows = "".join(f"<tr><td>{key}</td><td>{value:.5g}</td></tr>"
                   for key, value in geometry.items())
    examples = "".join(
        f'<h3>r={row["ratio"]}, {row["mode"]}: NLL {row["nll"]:.3f}, positive {row["positive_probability"]:.3f}</h3>'
        + "".join(f"<p><b>{html.escape(prompt)}</b> {html.escape(text)}</p>"
                  for prompt, text in zip(PROMPTS, row["texts"])) for row in generations)
    page = f"""<!doctype html><meta charset="utf-8"><title>{tag} diagnostics</title>
<style>body{{font:14px/1.5 system-ui;max-width:1120px;margin:auto;padding:24px;color:#17202a}}h1{{font-size:22px}}h2{{margin-top:28px}}svg{{width:100%;max-width:520px;background:#fafafa;border:1px solid #ddd}}path,polyline{{fill:none;stroke:#2563eb;stroke-width:2}}.cloud{{fill:#aaa;opacity:.35}}table{{border-collapse:collapse}}td{{padding:4px 18px 4px 0;border-bottom:1px solid #eee}}code{{background:#f3f4f6;padding:2px 4px}}</style>
<h1>{tag}: реальные активации и causal steering</h1>
<p>Checkpoint <code>{checkpoint['step']}</code>, data revision <code>{data['meta']['revision']}</code>, BOS excluded. JSON: <a href="diagnostics.json">diagnostics.json</a>. Config: <a href="config.json">config.json</a>. Training log: <a href="history.jsonl">history.jsonl</a>.</p>
<h2>Динамика validation objective ({loss_key})</h2>{line_chart(history, loss_key)}
<h2>Одна held-out проекция, fit только на clean validation activations</h2>{trajectory_svg(background, paths)}
<p>Серое облако содержит clean held-out activations. Оранжевая линия показывает clean и steered state; цветная траектория показывает denoiser input, промежуточные состояния и output.</p>
<h2>Величины в полном 768-мерном пространстве</h2><table>{rows}</table>
<h2>Decoded response-only alpha ablation</h2>{examples}
"""
    output = run / "diagnostics.html"
    output.write_text(page)
    return output


if __name__ == "__main__":
    names = sys.argv[1:] or ("mac_additive_simple", "mac_additive_capacity",
                            "mac_interpolation", "mac_glp")
    for name in names:
        print(diagnose(name))
