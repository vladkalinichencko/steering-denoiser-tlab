"""Naive steering baseline: h <- h + alpha * v after layer 6 of GPT-2.

Sweeps alpha and reports the fluency / concept trade-off, i.e. the Pareto front
every improved method has to beat. Writes runs/<tag>.json.

    python baseline.py --latent 0 --alphas 0 20 40 80 160
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
LOCATION = "resid_post_mlp"

PROMPTS = [
    "The weather today is",
    "My favourite thing about this city is",
    "I spent the afternoon",
    "He opened the door and",
    "The report concluded that",
    "She told me that",
]


def sae_vector(latent: int, device: str) -> torch.Tensor:
    """Unit-norm decoder column `latent` of the OpenAI v5_32k SAE."""
    import blobfile as bf
    import sparse_autoencoder

    with bf.BlobFile(sparse_autoencoder.paths.v5_32k(LOCATION, LAYER), mode="rb") as f:
        state_dict = torch.load(f)
    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
    v = autoencoder.decoder.weight[:, latent].detach().to(device).float()
    return v / v.norm()


def steer(v: torch.Tensor, alpha: float):
    def hook(resid, hook):
        return resid + alpha * v

    return hook


@torch.no_grad()
def generate(model, v, alpha, n_samples, max_new_tokens, seed):
    """n_samples continuations per prompt, steered with alpha * v."""
    torch.manual_seed(seed)
    hooks = [(HOOK, steer(v, alpha))] if alpha != 0 else []
    samples = []
    with model.hooks(fwd_hooks=hooks):
        for prompt in PROMPTS:
            tokens = model.to_tokens([prompt] * n_samples)
            out = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                verbose=False,
            )
            n_prompt = tokens.shape[1]
            samples += [
                {
                    "text": model.to_string(row),
                    "cont": model.to_string(row[n_prompt:]),
                    "n_prompt": n_prompt,
                }
                for row in out
            ]
    return samples


@torch.no_grad()
def perplexity(model, samples):
    """Perplexity of the continuations (prompt excluded) under the *clean* model."""
    losses = []
    for s in samples:
        tokens = model.to_tokens(s["text"])
        loss = model(tokens, return_type="loss", loss_per_token=True)[0]
        losses.append(loss[s["n_prompt"] - 1 :].mean().item())
    return float(torch.tensor(losses).mean().exp())


def dist_n(texts, n):
    grams, total = set(), 0
    for text in texts:
        words = text.split()
        for i in range(len(words) - n + 1):
            grams.add(tuple(words[i : i + n]))
            total += 1
    return len(grams) / max(total, 1)


def concept_score(texts, words):
    """Placeholder for an LLM judge: share of generations mentioning the concept."""
    words = [w.lower() for w in words]
    hits = sum(any(w in t.lower() for w in words) for t in texts)
    return hits / max(len(texts), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent", type=int, required=True, help="SAE decoder column")
    p.add_argument("--concept-words", nargs="+", required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 20, 40, 80, 160])
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", center_writing_weights=False, device=args.device
    )
    model.eval()
    v = sae_vector(args.latent, args.device)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("steering")
    mlflow.start_run(run_name=args.tag or f"baseline_latent{args.latent}")
    mlflow.log_params({k: str(v)[:250] for k, v in vars(args).items()})

    rows = []
    for alpha in args.alphas:
        samples = generate(model, v, alpha, args.n_samples, args.max_new_tokens, args.seed)
        conts = [s["cont"] for s in samples]
        rows.append(
            {
                "alpha": alpha,
                "ppl": perplexity(model, samples),
                "dist1": dist_n(conts, 1),
                "dist2": dist_n(conts, 2),
                "dist3": dist_n(conts, 3),
                "concept": concept_score(conts, args.concept_words),
                "sample": samples[0]["text"],
            }
        )
        r = rows[-1]
        print(f"alpha={alpha:6.1f}  ppl={r['ppl']:8.2f}  d1={r['dist1']:.3f} "
              f"d2={r['dist2']:.3f} d3={r['dist3']:.3f}  concept={r['concept']:.2f}")
        mlflow.log_metrics(
            {k: v for k, v in r.items() if k != "sample"}, step=int(alpha)
        )

    tag = args.tag or f"baseline_latent{args.latent}"
    out = pathlib.Path("runs") / f"{tag}.json"
    out.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
    mlflow.log_artifact(str(out))
    mlflow.end_run()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
