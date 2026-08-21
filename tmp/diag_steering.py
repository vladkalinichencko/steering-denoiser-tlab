"""Диагностика стиринга: что происходит с состоянием, а не «работает или нет».

Четыре среза на одних и тех же промптах:

1. Уход с многообразия. Норма, z-оценка и расстояние до ближайших настоящих
   активаций — для чистого h, для застиренного h+av и для расшумленного денойзером.
2. Распространение возмущения по слоям 6->12. Возмущение гасится сетью или растёт?
3. Logit lens: что модель предсказывает, если читать логиты с каждого слоя.
4. PCA: облако настоящих активаций, куда из него уезжает стиринг и возвращает ли
   денойзер.

    python tmp/diag_steering.py --latent 27677 --alphas 0 20 40 60 80 120
"""

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import transformer_lens

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

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
    """Единичный столбец декодера SAE. Кэшируем: качать 200 МБ ради одного вектора глупо."""
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
    if not path or not pathlib.Path(path).exists():
        return None
    from train_denoiser import Denoiser
    blob = torch.load(path, map_location=device, weights_only=False)
    model = Denoiser(blob["d_model"], blob["d_hidden"], blob["n_layers"]).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


@torch.no_grad()
def offmanifold(h, real, mean, std, ref):
    """Насколько точка не похожа на настоящие активации."""
    z = ((h - mean) / std).abs().mean().item()
    # расстояние до ближайшей из подвыборки настоящих активаций
    d = torch.cdist(h.reshape(-1, h.shape[-1]), ref).min(dim=1).values.mean().item()
    return {"норма": h.norm(dim=-1).mean().item(), "z": z, "до_ближайшей": d}


@torch.no_grad()
def collect(model, prompts):
    """Активации слоя 6 и все резидуалы по слоям.

    Позицию 0 (BOS) выбрасываем: у GPT-2 её норма на порядок больше остальных,
    она перетягивает на себя и среднее, и базис PCA, и статистику расстояний.
    """
    hs, per_layer = [], []
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        hs.append(cache[HOOK][1:])
        per_layer.append([cache[f"blocks.{i}.hook_resid_post"]
                          for i in range(model.cfg.n_layers)])
    return hs, per_layer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent", type=int, default=27677)
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 20, 40, 60, 80, 120])
    p.add_argument("--denoiser", default="runs/mlp_directional/denoiser.pt")
    p.add_argument("--acts", default="datasets/acts_layer6_100000.pt")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", center_writing_weights=False, device=args.device)
    model.eval()
    v = sae_vector(args.latent, args.device)
    denoiser = load_denoiser(args.denoiser, args.device)
    print(f"денойзер: {'загружен' if denoiser else 'нет, пропускаю'}")

    real = torch.load(args.acts, map_location=args.device).float()
    # тот же фильтр, что и для промптов: убираем выбросы с аномальной нормой
    norms = real.norm(dim=-1)
    keep = norms < norms.median() * 5
    print(f"настоящих активаций: {len(real)}, после отсева выбросов: {int(keep.sum())} "
          f"(медиана нормы {norms.median():.1f})")
    real = real[keep]
    mean, std = real.mean(0), real.std(0) + 1e-6
    ref = real[torch.randperm(len(real))[:4000]]

    hs, _ = collect(model, PROMPTS)
    h = torch.cat(hs)

    print(f"\n1. Уход с многообразия (латент {args.latent}):")
    print(f"{'alpha':>6} {'что':>12} {'норма':>9} {'z-оценка':>10} {'до ближайшей':>14}")
    rows = []
    for alpha in args.alphas:
        steered = h + alpha * v
        variants = {"застирено": steered}
        if denoiser is not None:
            variants["расшумлено"] = denoiser(steered)
        if alpha == 0:
            variants = {"чистое": h, **variants}
        for name, x in variants.items():
            m = offmanifold(x, real, mean, std, ref)
            rows.append({"alpha": alpha, "вариант": name, **m})
            print(f"{alpha:>6.0f} {name:>12} {m['норма']:>9.1f} {m['z']:>10.3f} "
                  f"{m['до_ближайшей']:>14.1f}")

    # 2. распространение возмущения по слоям
    print(f"\n2. Распространение возмущения по слоям (относительно нормы слоя):")
    prompt = PROMPTS[0]
    tokens = model.to_tokens(prompt)
    _, clean_cache = model.run_with_cache(tokens, remove_batch_dim=True)
    print(f"{'alpha':>6}" + "".join(f"{f'слой {i}':>9}" for i in range(LAYER, 12)))
    prop = []
    for alpha in args.alphas:
        if alpha == 0:
            continue
        hook = lambda resid, hook, a=alpha: resid + a * v
        with model.hooks(fwd_hooks=[(HOOK, hook)]):
            _, dirty = model.run_with_cache(tokens, remove_batch_dim=True)
        line, row = f"{alpha:>6.0f}", {"alpha": alpha}
        for i in range(LAYER, 12):
            key = f"blocks.{i}.hook_resid_post"
            rel = ((dirty[key] - clean_cache[key]).norm(dim=-1)
                   / clean_cache[key].norm(dim=-1)).mean().item()
            row[f"слой{i}"] = rel
            line += f"{rel:>9.3f}"
        prop.append(row)
        print(line)

    # 3. logit lens на застиренном потоке
    print(f"\n3. Logit lens, последняя позиция промпта {prompt!r}:")
    for alpha in (0, args.alphas[len(args.alphas) // 2], args.alphas[-1]):
        hooks = [(HOOK, lambda resid, hook, a=alpha: resid + a * v)] if alpha else []
        with torch.no_grad(), model.hooks(fwd_hooks=hooks):
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        print(f"  alpha={alpha:>5.0f}: ", end="")
        for i in (LAYER, 8, 11):
            resid = cache[f"blocks.{i}.hook_resid_post"][-1]
            logits = model.unembed(model.ln_final(resid[None, None]))[0, 0]
            top = logits.topk(3).indices
            toks = "/".join(model.to_string(t.view(1)).strip() or "␣" for t in top)
            print(f"слой{i}: {toks:<26}", end="")
        print()

    # 4. PCA
    with torch.no_grad():
        sub = real[torch.randperm(len(real))[:3000]]
        centre = sub.mean(0)
        _, _, basis = torch.pca_lowrank(sub - centre, q=2)
        proj = lambda x: ((x.reshape(-1, x.shape[-1]) - centre) @ basis).cpu()

        fig, ax = plt.subplots(figsize=(7, 6))
        pr = proj(sub)
        ax.scatter(pr[:, 0], pr[:, 1], s=3, alpha=.15, color="grey",
                   label="настоящие активации")
        colors = plt.cm.viridis(torch.linspace(0, .9, len(args.alphas)).numpy())
        for alpha, c in zip(args.alphas, colors):
            ps = proj(h + alpha * v)
            ax.scatter(ps[:, 0], ps[:, 1], s=10, color=c, label=f"стиринг a={alpha:.0f}")
            if denoiser is not None and alpha > 0:
                pd_ = proj(denoiser(h + alpha * v))
                ax.scatter(pd_[:, 0], pd_[:, 1], s=10, color=c, marker="x")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("куда уезжает стиринг (x — после денойзера)")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout(); fig.savefig("tmp/steer_pca.png", dpi=140)

    pathlib.Path("tmp/diag_steering.json").write_text(json.dumps(
        {"латент": args.latent, "многообразие": rows, "распространение": prop},
        ensure_ascii=False, indent=2))
    print("\n-> tmp/steer_pca.png, tmp/diag_steering.json")


if __name__ == "__main__":
    main()
