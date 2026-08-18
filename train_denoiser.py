"""Денойзер активаций GPT-2: учим возвращать испорченное h на многообразие.

Обучение идёт только на шуме — про векторы стиринга денойзер ничего не знает.
Это главное методологическое условие задания, см. NOTES.

Схема зашумления по умолчанию — интерполяция из условия:
    h_noisy = t * h + (1 - t) * eps,   t ~ U[0, 1],  eps ~ N(0, sigma^2)
Так модель видит весь спектр повреждений, а не одну амплитуду.

    python train_denoiser.py --tag mlp --steps 4000
    python train_denoiser.py --tag mlp --noise additive     # h + eps, для сравнения
"""

import argparse
import json
import os
import pathlib

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER = 6
HOOK = f"blocks.{LAYER}.hook_resid_post"
CACHE = pathlib.Path("datasets")


class Denoiser(nn.Module):
    """MLP с остаточной связью: чинит вектор, а не пересобирает его заново."""

    def __init__(self, d_model=768, d_hidden=2048, n_layers=2):
        super().__init__()
        self.inp = nn.Linear(d_model, d_hidden)
        self.hidden = nn.ModuleList(nn.Linear(d_hidden, d_hidden) for _ in range(n_layers - 1))
        self.out = nn.Linear(d_hidden, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h):
        x = F.gelu(self.inp(self.norm(h)))
        for layer in self.hidden:
            x = F.gelu(layer(x))
        return h + self.out(x)  # остаток: на чистом входе легко выучить тождество


def denoise_loss(pred, target, noisy, mode):
    if mode == "mse":
        return F.mse_loss(pred, target)
    # ошибка относительно того, насколько вход был испорчен
    damage = (target - noisy).pow(2).mean(-1) + 1e-6
    return ((pred - target).pow(2).mean(-1) / damage).mean()


def collect_activations(n_vectors, device, seq_len=128):
    """Кэш активаций слоя 6 на тексте FineWeb. Считается один раз."""
    cache = CACHE / f"acts_layer{LAYER}_{n_vectors}.pt"
    if cache.exists():
        print(f"беру активации из кэша: {cache}")
        return torch.load(cache, map_location=device)

    import transformer_lens
    from datasets import load_dataset

    print(f"собираю {n_vectors} активаций (это разово)")
    model = transformer_lens.HookedTransformer.from_pretrained(
        "gpt2", center_writing_weights=False, device=device)
    model.eval()

    stream = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                          split="train", streaming=True)
    chunks, total = [], 0
    with torch.no_grad():
        for example in stream:
            tokens = model.to_tokens(example["text"])[:, :seq_len]
            if tokens.shape[1] < 8:
                continue
            _, cache_out = model.run_with_cache(tokens, remove_batch_dim=True,
                                                names_filter=HOOK)
            chunks.append(cache_out[HOOK].cpu())
            total += chunks[-1].shape[0]
            if total >= n_vectors:
                break
    acts = torch.cat(chunks)[:n_vectors]
    CACHE.mkdir(exist_ok=True)
    torch.save(acts, cache)
    print(f"сохранено: {cache}  {tuple(acts.shape)}")
    return acts.to(device)


def corrupt(h, mode, sigma, generator=None, alpha_max=4.0):
    if mode == "directional":
        # порча, похожая на стиринг: сдвиг вдоль одного случайного направления.
        # Направления случайные, валидационные v денойзер по-прежнему не видит.
        u = torch.randn(h.shape, device=h.device, generator=generator)
        u = u / u.norm(dim=-1, keepdim=True)
        alpha = torch.rand(h.shape[0], 1, device=h.device, generator=generator)
        return h + alpha * alpha_max * h.norm(dim=-1, keepdim=True) * u

    eps = torch.randn(h.shape, device=h.device, generator=generator) * sigma
    if mode == "additive":
        return h + eps
    t = torch.rand(h.shape[0], 1, device=h.device, generator=generator)
    return t * h + (1 - t) * eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--n-vectors", type=int, default=200_000)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--d-hidden", type=int, default=2048)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--noise", choices=["interp", "additive", "directional"],
                   default="interp")
    p.add_argument("--loss", choices=["mse", "relative"], default="mse",
                   help="relative делит ошибку на величину повреждения, чтобы слабо "
                        "зашумлённые примеры не терялись на фоне сильно зашумлённых")
    p.add_argument("--sigma", type=float, default=None,
                   help="масштаб шума; по умолчанию берётся из нормы активаций")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    acts = collect_activations(args.n_vectors, args.device).float()
    n_val = int(len(acts) * args.val_frac)
    val, train = acts[:n_val], acts[n_val:]

    # шум соизмеряем с типичной нормой компоненты, иначе он либо незаметен, либо всё сносит
    sigma = args.sigma if args.sigma is not None else float(train.std())
    print(f"активаций: обучение {len(train)}, валидация {len(val)}; sigma={sigma:.3f}")

    model = Denoiser(acts.shape[-1], args.d_hidden, args.n_layers).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("denoiser")
    mlflow.start_run(run_name=args.tag)
    mlflow.log_params({**vars(args), "sigma": sigma})

    out = pathlib.Path("runs") / args.tag
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for step in range(args.steps):
        idx = torch.randint(0, len(train), (args.batch_size,), device=args.device)
        h = train[idx]
        noisy = corrupt(h, args.noise, sigma)
        loss = denoise_loss(model(noisy), h, noisy, args.loss)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 200 == 0 or step == args.steps - 1:
            with torch.no_grad():
                val_noisy = corrupt(val, args.noise, sigma)
                vl = denoise_loss(model(val_noisy), val, val_noisy, args.loss).item()
                # контроль: денойзер не должен ломать чистые активации
                clean = F.mse_loss(model(val), val).item()
            print(f"шаг {step:>5}  train {loss.item():.4f}  val {vl:.4f}  "
                  f"на чистых {clean:.4f}")
            mlflow.log_metrics({"train_mse": loss.item(), "val_mse": vl,
                                "clean_mse": clean}, step=step)
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "d_model": acts.shape[-1],
                            "d_hidden": args.d_hidden, "n_layers": args.n_layers,
                            "sigma": sigma, "noise": args.noise, "step": step},
                           out / "denoiser.pt")

    mlflow.log_metrics({"best_val_mse": best})
    mlflow.end_run()
    (out / "config.json").write_text(json.dumps({**vars(args), "sigma": sigma,
                                                 "best_val_mse": best}, indent=2))
    print(f"лучший val MSE {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
