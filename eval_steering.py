"""Парето-кривая стиринга с денойзером и без него.

Главный эксперимент задания: сдвигает ли денойзер компромисс между связностью и
присутствием концепта. Считаем те же метрики, что и baseline.py, но с интервенцией
h -> denoiser(h + alpha*v) и сравниваем с h -> h + alpha*v.

    python eval_steering.py --latent 27677 --concept-words paris eiffel france \
        --denoiser runs/mlp_interp/denoiser.pt --alphas 0 20 40 60 80 120
"""

import argparse
import json
import os
import pathlib

import mlflow
import torch
import transformer_lens

LAYER = 6
HOOK = f"blocks.{LAYER}.hook_resid_post"
PROMPTS = [
    "The weather today is",
    "My favourite thing about this city is",
    "I spent the afternoon",
    "He opened the door and",
    "The report concluded that",
    "She told me that",
]


def sae_vector(latent, device):
    cache = pathlib.Path("datasets") / f"v_latent{latent}_layer{LAYER}.pt"
    if cache.exists():
        return torch.load(cache, map_location=device).float()
    import blobfile as bf
    import sparse_autoencoder
    with bf.BlobFile(sparse_autoencoder.paths.v5_32k("resid_post_mlp", LAYER), "rb") as f:
        ae = sparse_autoencoder.Autoencoder.from_state_dict(torch.load(f))
    v = ae.decoder.weight[:, latent].detach().float()
    v = v / v.norm()
    cache.parent.mkdir(exist_ok=True)
    torch.save(v, cache)
    return v.to(device)


def load_denoiser(path, device):
    from train_denoiser import Denoiser
    blob = torch.load(path, map_location=device, weights_only=False)
    m = Denoiser(blob["d_model"], blob["d_hidden"], blob["n_layers"]).to(device)
    m.load_state_dict(blob["model"])
    m.eval()
    return m


def make_hook(v, alpha, denoiser, steps):
    def hook(resid, hook):
        h = resid + alpha * v
        if denoiser is not None:
            for _ in range(steps):
                h = denoiser(h)
        return h
    return hook


@torch.no_grad()
def generate(model, hook, n_samples, max_new_tokens, seed):
    torch.manual_seed(seed)
    samples = []
    with model.hooks(fwd_hooks=[(HOOK, hook)] if hook else []):
        for prompt in PROMPTS:
            tokens = model.to_tokens([prompt] * n_samples)
            out = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=1.0, top_k=50, verbose=False)
            n_prompt = tokens.shape[1]
            samples += [{"text": model.to_string(r), "cont": model.to_string(r[n_prompt:]),
                         "n_prompt": n_prompt} for r in out]
    return samples


@torch.no_grad()
def perplexity(model, samples):
    losses = []
    for s in samples:
        loss = model(model.to_tokens(s["text"]), return_type="loss", loss_per_token=True)[0]
        losses.append(loss[s["n_prompt"] - 1:].mean().item())
    return float(torch.tensor(losses).mean().exp())


def dist_n(texts, n):
    grams, total = set(), 0
    for t in texts:
        w = t.split()
        for i in range(len(w) - n + 1):
            grams.add(tuple(w[i:i + n])); total += 1
    return len(grams) / max(total, 1)


def concept_score(texts, words):
    words = [w.lower() for w in words]
    return sum(any(w in t.lower() for w in words) for t in texts) / max(len(texts), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent", type=int, default=27677)
    p.add_argument("--concept-words", nargs="+", required=True)
    p.add_argument("--denoiser", default="runs/mlp_interp/denoiser.pt")
    p.add_argument("--denoise-steps", type=int, default=1)
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 20, 40, 60, 80, 120])
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default="steering_vs_denoised")
    args = p.parse_args()

    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", center_writing_weights=False, device=args.device)
    model.eval()
    v = sae_vector(args.latent, args.device)
    denoiser = load_denoiser(args.denoiser, args.device)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("steering")
    mlflow.start_run(run_name=args.tag)
    mlflow.log_params({k: str(v_)[:250] for k, v_ in vars(args).items()})

    rows = []
    print(f"{'режим':>12} {'alpha':>6} {'ppl':>9} {'dist2':>7} {'concept':>8}")
    for mode, den in (("без денойзера", None), ("с денойзером", denoiser)):
        for alpha in args.alphas:
            hook = make_hook(v, alpha, den, args.denoise_steps) if (alpha or den) else None
            samples = generate(model, hook, args.n_samples, args.max_new_tokens, args.seed)
            conts = [s["cont"] for s in samples]
            row = {"режим": mode, "alpha": alpha, "ppl": perplexity(model, samples),
                   "dist2": dist_n(conts, 2), "concept": concept_score(conts, args.concept_words),
                   "пример": samples[0]["cont"][:80]}
            rows.append(row)
            print(f"{mode:>12} {alpha:>6.0f} {row['ppl']:>9.2f} {row['dist2']:>7.3f} "
                  f"{row['concept']:>8.2f}")
            mlflow.log_metrics({f"{'den' if den else 'raw'}_ppl": row["ppl"],
                                f"{'den' if den else 'raw'}_concept": row["concept"]},
                               step=int(alpha))

    out = pathlib.Path("runs") / f"{args.tag}.json"
    out.write_text(json.dumps({"config": vars(args), "rows": rows},
                              ensure_ascii=False, indent=2))
    mlflow.log_artifact(str(out))
    mlflow.end_run()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
