"""Траектория представлений: куда уводит стиринг и куда возвращает денойзер.

Всё рисуется в ОДНОЙ системе координат: базис PCA, центр и пределы осей считаются
один раз по настоящим активациям и сохраняются в datasets/pca_frame.pt. Любой
следующий график берёт их оттуда, поэтому картинки сравнимы между собой и между
запусками, а движение видно по-настоящему, а не из-за перенастройки осей.

Денойзер применяется итеративно: h <- denoiser(h). Даже если он обучался как
одношаговый, это показывает, есть ли у него неподвижная точка и где она.

    python tmp/diag_denoise_path.py --alphas 0 20 40 60 80 120 --denoise-steps 5
"""

import argparse
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
FRAME = pathlib.Path("datasets/pca_frame.pt")
PROMPTS = ["The weather today is", "He opened the door and", "The report concluded that"]


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
    if not path or not pathlib.Path(path).exists():
        return None
    from train_denoiser import Denoiser
    blob = torch.load(path, map_location=device, weights_only=False)
    m = Denoiser(blob["d_model"], blob["d_hidden"], blob["n_layers"]).to(device)
    m.load_state_dict(blob["model"])
    m.eval()
    return m


def get_frame(real, device, rebuild=False):
    """Единая система координат для всех графиков: центр, базис и пределы осей."""
    if FRAME.exists() and not rebuild:
        f = torch.load(FRAME, map_location=device)
        return f["centre"], f["basis"], f["lim"]

    centre = real.mean(0)
    _, _, basis = torch.pca_lowrank(real - centre, q=2)
    proj = (real - centre) @ basis
    # пределы по 99-й перцентили, чтобы редкие выбросы не сжимали картинку
    lim = float(proj.abs().quantile(0.99).item()) * 1.6
    FRAME.parent.mkdir(exist_ok=True)
    torch.save({"centre": centre.cpu(), "basis": basis.cpu(), "lim": lim}, FRAME)
    print(f"система координат построена и сохранена: {FRAME}, предел осей ±{lim:.1f}")
    return centre, basis, lim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent", type=int, default=27677)
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 20, 40, 60, 80, 120])
    p.add_argument("--denoisers", nargs="+",
                   default=["runs/mlp_directional/denoiser.pt", "runs/mlp_interp/denoiser.pt"])
    p.add_argument("--denoise-steps", type=int, default=5)
    p.add_argument("--acts", default="datasets/acts_layer6_100000.pt")
    p.add_argument("--tokens", type=int, default=12, help="сколько токенов рисовать")
    p.add_argument("--rebuild-frame", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    real = torch.load(args.acts, map_location=args.device).float()
    norms = real.norm(dim=-1)
    real = real[norms < norms.median() * 5]          # без BOS-выбросов
    centre, basis, lim = get_frame(real, args.device, args.rebuild_frame)
    proj = lambda x: ((x.reshape(-1, x.shape[-1]) - centre) @ basis).cpu()

    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", center_writing_weights=False, device=args.device)
    model.eval()
    v = sae_vector(args.latent, args.device)

    with torch.no_grad():
        hs = []
        for prompt in PROMPTS:
            _, cache = model.run_with_cache(model.to_tokens(prompt), remove_batch_dim=True)
            hs.append(cache[HOOK][1:])               # без BOS
        h = torch.cat(hs)[: args.tokens]

    denoisers = [(pathlib.Path(d).parent.name, load_denoiser(d, args.device))
                 for d in args.denoisers]
    denoisers = [(n, m) for n, m in denoisers if m is not None]
    print("денойзеры:", ", ".join(n for n, _ in denoisers) or "нет")

    sub = real[torch.randperm(len(real))[:4000]]
    cloud = proj(sub)

    ncol = 1 + len(denoisers)
    fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 5.6), squeeze=False)
    axes = axes[0]

    # --- слева: траектория стиринга ---
    ax = axes[0]
    ax.scatter(cloud[:, 0], cloud[:, 1], s=3, alpha=.12, color="grey")
    path = torch.stack([proj(h + a * v) for a in args.alphas])   # (шаги, токены, 2)
    for t in range(path.shape[1]):
        ax.plot(path[:, t, 0], path[:, t, 1], lw=1.0, color="tab:red", alpha=.7)
    ax.scatter(path[0, :, 0], path[0, :, 1], s=28, color="black", zorder=3,
               label="alpha=0 (чистое)")
    ax.scatter(path[-1, :, 0], path[-1, :, 1], s=28, color="tab:red", zorder=3,
               label=f"alpha={args.alphas[-1]:.0f}")
    ax.set_title(f"стиринг: alpha 0 -> {args.alphas[-1]:.0f}")
    ax.legend(fontsize=8, loc="upper left")

    # --- справа: итеративное расшумление из самой дальней точки ---
    start = h + args.alphas[-1] * v
    for ax, (name, den) in zip(axes[1:], denoisers):
        ax.scatter(cloud[:, 0], cloud[:, 1], s=3, alpha=.12, color="grey")
        with torch.no_grad():
            steps = [start]
            for _ in range(args.denoise_steps):
                steps.append(den(steps[-1]))
        dpath = torch.stack([proj(s) for s in steps])
        for t in range(dpath.shape[1]):
            ax.plot(dpath[:, t, 0], dpath[:, t, 1], lw=1.0, color="tab:blue", alpha=.7)
        ax.scatter(dpath[0, :, 0], dpath[0, :, 1], s=28, color="tab:red", zorder=3,
                   label="вход (застирено)")
        ax.scatter(dpath[-1, :, 0], dpath[-1, :, 1], s=28, color="tab:blue", zorder=3,
                   label=f"после {args.denoise_steps} шагов")
        ax.scatter(path[0, :, 0], path[0, :, 1], s=28, color="black", marker="*",
                   zorder=4, label="цель (чистое)")
        moved = (steps[-1] - steps[0]).norm(dim=-1).mean().item()
        gap0 = (steps[0] - h).norm(dim=-1).mean().item()
        gap1 = (steps[-1] - h).norm(dim=-1).mean().item()
        ax.set_title(f"{name}: сдвиг {moved:.1f}, до цели {gap0:.1f} -> {gap1:.1f}")
        ax.legend(fontsize=8, loc="upper left")
        print(f"{name}: сместил на {moved:.2f}; расстояние до чистого "
              f"{gap0:.2f} -> {gap1:.2f}")

    for ax in axes:
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle("единая система координат: центр и базис по настоящим активациям")
    fig.tight_layout()
    fig.savefig("tmp/denoise_path.png", dpi=140)
    print("-> tmp/denoise_path.png")


if __name__ == "__main__":
    main()
