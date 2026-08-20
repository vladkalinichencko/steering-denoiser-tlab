"""Validate a steering vector before steering with it.

Step 3 of the task is "check you get a picture like the example", and the picture is
only meaningful if the vector means what its name says. Two checks:

    --tokens   the tokens that fire this SAE latent hardest, straight from text
    --lens     the tokens the direction itself promotes, through the unembedding
    --sweep    the naive h -> h + alpha*v front (same numbers as eval_steering.py)

    python baseline.py --vector sae:27677 --tokens --lens
    python baseline.py --vector diffmean:sentiment --sweep
"""

import argparse
import json
import pathlib

import torch

import steering


@torch.no_grad()
def top_tokens(latent, model, device, n_texts=400, k=20, seq_len=128):
    """Tokens with the largest activation of SAE latent `latent`, with context."""
    import blobfile as bf
    import sparse_autoencoder
    from datasets import load_dataset

    with bf.BlobFile(sparse_autoencoder.paths.v5_32k("resid_post_mlp", steering.LAYER), "rb") as f:
        ae = sparse_autoencoder.Autoencoder.from_state_dict(torch.load(f)).to(device)

    stream = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                          split="train", streaming=True)
    hits = []
    for i, example in enumerate(stream):
        if i >= n_texts:
            break
        tokens = model.to_tokens(example["text"])[:, :seq_len]
        if tokens.shape[1] < 8:
            continue
        _, out = model.run_with_cache(tokens, names_filter=steering.HOOK)
        acts = ae.encode(out[steering.HOOK][0])[0][:, latent]  # encode -> (latents, info)
        for pos in acts.topk(min(3, len(acts))).indices.tolist():
            hits.append((float(acts[pos]), model.to_string(tokens[0, pos]),
                         model.to_string(tokens[0, max(0, pos - 8):pos + 1])))
    hits.sort(reverse=True)
    return hits[:k]


@torch.no_grad()
def logit_lens(v, model, k=15):
    """Tokens the direction pushes towards, read straight off the unembedding.

    Cheap and independent of any text sample: what steering along v adds to the logits
    if the rest of the network passed it through untouched. Disagreement with the
    top-activating tokens is informative — it means the latent reads one thing and
    writes another."""
    logits = model.ln_final(v[None]) @ model.W_U
    top = logits[0].topk(k)
    return [(round(float(x), 3), model.to_string(i)) for x, i in zip(top.values, top.indices)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vector", required=True, help="sae:<i> | diffmean:<c>")
    p.add_argument("--concept-words", nargs="+", default=None)
    p.add_argument("--tokens", action="store_true", help="что латент вообще ловит")
    p.add_argument("--lens", action="store_true", help="куда направление толкает логиты")
    p.add_argument("--sweep", action="store_true", help="наивный стиринг по alpha")
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 10, 20, 40, 80, 160])
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model = steering.load_model(args.device)
    out = {"vector": args.vector}

    if args.tokens:
        kind, name = args.vector.split(":")
        assert kind == "sae", "топ-токены есть только у латента SAE"
        out["top_tokens"] = top_tokens(int(name), model, args.device)
        for act, tok, ctx in out["top_tokens"]:
            print(f"{act:8.3f}  {tok!r:20} …{ctx}")

    if args.lens:
        out["logit_lens"] = logit_lens(steering.vector(args.vector, model, args.device), model)
        print("  ".join(f"{tok!r}:{val}" for val, tok in out["logit_lens"]))

    if args.sweep:
        v = steering.vector(args.vector, model, args.device)
        out["rows"] = []
        for alpha in args.alphas:
            hooks = [(steering.HOOK, steering.make_hook(v, alpha))]
            samples = steering.generate(model, hooks, args.n_samples,
                                        args.max_new_tokens, args.seed)
            row = {"alpha": alpha,
                   **steering.measure(model, samples, args.vector, args.concept_words),
                   "sample": samples[0]["cont"]}
            out["rows"].append(row)
            print(f"alpha={alpha:6.1f}  ppl={row['ppl']:8.2f}  d2={row['dist2']:.3f}  "
                  f"concept={row['concept']:.3f}  {row['sample']!r}")

    path = pathlib.Path("runs") / f"check_{args.vector.replace(':', '')}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
