"""Train the activation model on GPT-2 layer-6 activations, in one of two objectives.

    python train_denoiser.py --tag glp  --objective flow --steps 20000
    python train_denoiser.py --tag mse  --objective mse  --steps 20000

Both see only noise-corrupted activations and never a validation steering vector —
that is the methodological condition of the task. Everything else (architecture,
standardisation, data) is shared, so the comparison is objective against objective.
"""

import argparse
import json
import os
import pathlib

import mlflow
import torch
import torch.nn.functional as F

import denoiser
import steering

CACHE = pathlib.Path("datasets")


def collect_activations(n_vectors, device, seq_len=128):
    """FineWeb text through GPT-2, layer-6 residual stream, cached in fp16."""
    cache = CACHE / f"acts_layer{steering.LAYER}_{n_vectors}.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu")

    from datasets import load_dataset
    model = steering.load_model(device)
    stream = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                          split="train", streaming=True)
    chunks, total = [], 0
    with torch.no_grad():
        for example in stream:
            tokens = model.to_tokens(example["text"])[:, :seq_len]
            if tokens.shape[1] < 8:
                continue
            _, out = model.run_with_cache(tokens, remove_batch_dim=True,
                                          names_filter=steering.HOOK)
            chunks.append(out[steering.HOOK].half().cpu())
            total += chunks[-1].shape[0]
            if total >= n_vectors:
                break
            if len(chunks) % 500 == 0:
                print(f"  {total}/{n_vectors}")
    acts = torch.cat(chunks)[:n_vectors]
    CACHE.mkdir(exist_ok=True)
    torch.save(acts, cache)
    print(f"сохранено: {cache}  {tuple(acts.shape)}")
    return acts


def corrupt(z, mode, sigma, generator=None, basis=None):
    """Порча в стандартизованном пространстве, поэтому sigma=1 — это масштаб данных.

    -> (испорченное, цель). У всех схем цель это сам z, кроме tangent: там сдвиг вдоль
    главных направлений считается допустимым и остаётся в цели, а убирать денойзер
    должен только сдвиг наружу. Так его учат не «возвращать в точку», а «возвращать на
    поверхность», что ближе к тому, зачем он нужен при стиринге.
    """
    eps = torch.randn(z.shape, device=z.device, generator=generator) * sigma
    if mode == "additive":
        return z + eps, z
    if mode == "tangent":
        eps2 = torch.randn(z.shape, device=z.device, generator=generator) * sigma
        along = (eps @ basis) @ basis.T
        out = eps2 - (eps2 @ basis) @ basis.T
        return z + along + out, z + along
    t = torch.rand(len(z), 1, device=z.device, generator=generator)
    return t * z + (1 - t) * eps, z


def batch_loss(net, z, args, generator=None, basis=None):
    if args.objective == "flow":
        zt, t, u = denoiser.flow_batch(z, generator)
        return F.mse_loss(net(zt, t), u)
    noisy, target = corrupt(z, args.noise, args.sigma, generator, basis)
    return F.mse_loss(net(noisy), target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--objective", choices=["flow", "mse"], default="flow")
    p.add_argument("--n-vectors", type=int, default=500_000)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--width", type=int, default=2, help="d_model = width * d_act (GLP: 2)")
    p.add_argument("--expand", type=int, default=2, help="d_ff = expand * d_model (GLP: 2)")
    p.add_argument("--noise", choices=["interp", "additive", "tangent"], default="interp",
                   help="только для регрессии. interp и additive — формулы из условия; "
                        "tangent сохраняет сдвиг вдоль главных направлений и убирает "
                        "только сдвиг наружу")
    p.add_argument("--tangent-dim", type=int, default=64)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps")
    p.add_argument("--log-every", type=int, default=250)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    acts = collect_activations(args.n_vectors, args.device)
    n_val = int(len(acts) * args.val_frac)
    val, train = acts[:n_val].to(args.device).float(), acts[n_val:]
    print(f"активаций: обучение {len(train)}, валидация {len(val)}")

    predict = "velocity" if args.objective == "flow" else "residual"
    net = denoiser.Denoiser(acts.shape[-1], args.width, args.expand, args.n_blocks, predict)
    net.set_stats(train[::13].float())
    net = net.to(args.device)
    basis = (steering.tangent_basis(args.tangent_dim, args.device)
             if args.noise == "tangent" else None)
    val = net.standardize(val)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("denoiser")
    mlflow.start_run(run_name=args.tag)
    mlflow.log_params({**vars(args), "params": sum(p.numel() for p in net.parameters())})

    out = pathlib.Path("runs") / args.tag
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "history.jsonl").open("w")
    best = float("inf")

    for step in range(args.steps):
        idx = torch.randint(0, len(train), (args.batch_size,))
        z = net.standardize(train[idx].to(args.device).float())
        loss = batch_loss(net, z, args, basis=basis)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                g = torch.Generator(device=args.device).manual_seed(0)  # один и тот же шум
                vl = batch_loss(net, val, args, g, basis).item()
            row = {"step": step, "train_loss": loss.item(), "val_loss": vl}
            print(f"шаг {step:>6}  train {loss.item():.4f}  val {vl:.4f}")
            log.write(json.dumps(row) + "\n")
            log.flush()
            mlflow.log_metrics(row, step=step)
            if vl < best:
                best = vl
                torch.save({"model": net.state_dict(), "args": vars(args),
                            "d_act": acts.shape[-1], "step": step}, out / "denoiser.pt")

    log.close()
    mlflow.log_metrics({"best_val_loss": best})
    mlflow.end_run()
    (out / "config.json").write_text(json.dumps({**vars(args), "best_val_loss": best}, indent=2))
    print(f"лучший val {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
    os._exit(0)
