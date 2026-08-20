"""Everything the three steering scripts share: the hook point, the vectors, the axes.

The task asks for a Pareto front in (fluency, concept), so fluency and concept have to
be measured the same way for the naive baseline, the denoiser and GLP — otherwise the
fronts are not on the same plot. One module, one definition of each number.

Two vector families:
    sae:<i>        decoder column i of the OpenAI v5_32k SAE, as the task suggests;
                   scored by keyword hits, which is as weak as it sounds.
    diffmean:<c>   difference of means between two contrast sets, the vector GLP itself
                   steers with; scored by an off-the-shelf sentiment classifier, so the
                   concept axis is a real number and not a word list.
"""

import pathlib

import torch
import transformer_lens

LAYER = 6
HOOK = f"blocks.{LAYER}.hook_resid_post"
CACHE = pathlib.Path("datasets")

PROMPTS = [
    "The weather today is",
    "My favourite thing about this city is",
    "I spent the afternoon",
    "He opened the door and",
    "The report concluded that",
    "She told me that",
]

CONTRAST = {
    "sentiment": (
        ["I loved every minute of it.", "This is wonderful news.", "A delightful, warm film.",
         "She is brilliant and kind.", "The food was excellent.", "What a beautiful morning.",
         "Everything went perfectly.", "I am so happy about this."],
        ["I hated every minute of it.", "This is terrible news.", "A dull, cold film.",
         "She is stupid and cruel.", "The food was disgusting.", "What an awful morning.",
         "Everything went wrong.", "I am so upset about this."],
    ),
}


def load_model(device):
    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", center_writing_weights=False, device=device)
    model.eval()
    return model


def sae_vector(latent, device):
    cache = CACHE / f"v_sae{latent}_layer{LAYER}.pt"
    if not cache.exists():
        import blobfile as bf
        import sparse_autoencoder
        with bf.BlobFile(sparse_autoencoder.paths.v5_32k("resid_post_mlp", LAYER), "rb") as f:
            ae = sparse_autoencoder.Autoencoder.from_state_dict(torch.load(f))
        v = ae.decoder.weight[:, latent].detach().float()
        CACHE.mkdir(exist_ok=True)
        torch.save(v / v.norm(), cache)
    return torch.load(cache, map_location=device).float()


@torch.no_grad()
def diffmean_vector(concept, model, device):
    """Mean activation of the positive set minus the negative one, over all positions."""
    cache = CACHE / f"v_diffmean_{concept}_layer{LAYER}.pt"
    if not cache.exists():
        sides = []
        for texts in CONTRAST[concept]:
            acts = [model.run_with_cache(model.to_tokens(t), names_filter=HOOK)[1][HOOK][0]
                    for t in texts]
            sides.append(torch.cat(acts).mean(0))
        v = (sides[0] - sides[1]).float().cpu()
        CACHE.mkdir(exist_ok=True)
        torch.save(v / v.norm(), cache)
    return torch.load(cache, map_location=device).float()


def vector(spec, model, device):
    kind, name = spec.split(":")
    return sae_vector(int(name), device) if kind == "sae" else diffmean_vector(name, model, device)


def make_hook(v, alpha, repair=None):
    """h -> repair(h + alpha*v). repair=None is the naive baseline from the task."""
    def hook(resid, hook):
        out = resid + alpha * v
        if repair is None:
            return out
        shape = out.shape
        return repair(out.reshape(-1, shape[-1])).reshape(shape)
    return hook


@torch.no_grad()
def generate(model, hooks, n_samples, max_new_tokens, seed):
    torch.manual_seed(seed)
    samples = []
    with model.hooks(fwd_hooks=hooks):
        for prompt in PROMPTS:
            tokens = model.to_tokens([prompt] * n_samples)
            out = model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=1.0, top_k=50, verbose=False)
            n_prompt = tokens.shape[1]
            samples += [{"text": model.to_string(row), "cont": model.to_string(row[n_prompt:]),
                         "n_prompt": n_prompt} for row in out]
    return samples


@torch.no_grad()
def perplexity(model, samples):
    """Perplexity of the continuations under the *clean* model — the fluency axis."""
    losses = []
    for s in samples:
        loss = model(model.to_tokens(s["text"]), return_type="loss", loss_per_token=True)[0]
        losses.append(loss[s["n_prompt"] - 1:].mean().item())
    return float(torch.tensor(losses).mean().exp())


def dist_n(texts, n):
    grams, total = set(), 0
    for text in texts:
        words = text.split()
        for i in range(len(words) - n + 1):
            grams.add(tuple(words[i:i + n]))
            total += 1
    return len(grams) / max(total, 1)


_judge = None


def sentiment_score(texts):
    """P(positive) averaged over continuations, from a local SST-2 classifier."""
    global _judge
    if _judge is None:
        from transformers import pipeline
        _judge = pipeline("sentiment-analysis",
                          model="distilbert-base-uncased-finetuned-sst-2-english",
                          truncation=True)
    out = _judge([t if t.strip() else "." for t in texts])
    return sum(r["score"] if r["label"] == "POSITIVE" else 1 - r["score"] for r in out) / len(out)


def keyword_score(texts, words):
    words = [w.lower() for w in words]
    return sum(any(w in t.lower() for w in words) for t in texts) / max(len(texts), 1)


def measure(model, samples, spec, words):
    conts = [s["cont"] for s in samples]
    return {"ppl": perplexity(model, samples),
            "dist1": dist_n(conts, 1), "dist2": dist_n(conts, 2), "dist3": dist_n(conts, 3),
            "concept": sentiment_score(conts) if spec.startswith("diffmean")
                       else keyword_score(conts, words or [])}
