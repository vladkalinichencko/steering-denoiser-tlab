"""Pareto fronts of the repair methods, on one plot and one set of numbers.

The whole task is: does repairing the steered activation move the trade-off between
fluency and concept presence. So every method is swept over the same alphas with the
same prompts, seed and metrics, and the only difference is what happens to h + alpha*v.

    none  h + alpha*v                            наивный бейзлайн из условия
    mse   denoiser(h + alpha*v)                  предложение условия
    glp   SDEdit по выученному полю скоростей    метод GLP, arXiv:2602.06964

    python eval_steering.py --vector diffmean:sentiment --repair none mse glp \
        --mse runs/mse/denoiser.pt --glp runs/glp/denoiser.pt
"""

import argparse
import json
import os
import pathlib

import mlflow
import torch

import denoiser
import steering


def load(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    a = blob["args"]
    net = denoiser.Denoiser(blob["d_act"], a["width"], a["expand"], a["n_blocks"],
                            "velocity" if a["objective"] == "flow" else "residual")
    net.load_state_dict(blob["model"])
    return net.to(device).eval()


def methods(args, nets):
    """-> [(подпись, функция починки)]. У GLP t_start — главный рычаг: он решает,
    сколько от правки остаётся и сколько чинится, поэтому это отдельные точки."""
    out = []
    for kind in args.repair:
        if kind == "none":
            out.append(("none", None))
        elif kind == "mse":
            out.append(("mse", lambda h: nets["mse"].repair(h)))
        else:
            for t in args.t_start:
                out.append((f"glp_t{t:g}",
                            lambda h, t=t: denoiser.sdedit(nets["glp"], h, t, args.steps)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vector", default="diffmean:sentiment", help="sae:<i> | diffmean:<c>")
    p.add_argument("--concept-words", nargs="+", default=None, help="только для sae:*")
    p.add_argument("--concept", default="auto", choices=["auto", "lens", "latent", "words"],
                   help="чем мерить концепт: lens — доля токенов, которые продвигает само "
                        "направление; latent — активация латента; см. NOTES про то, что это "
                        "разные вещи")
    p.add_argument("--repair", nargs="+", default=["none"], choices=["none", "mse", "glp"])
    p.add_argument("--mse", default=None, help="чекпойнт denoiser(h+eps)->h")
    p.add_argument("--glp", default=None, help="чекпойнт flow matching")
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 10, 20, 40, 80, 160])
    p.add_argument("--t-start", type=float, nargs="+", default=[0.5],
                   help="GLP: уровни шума SDEdit; в статье 0.5")
    p.add_argument("--steps", type=int, default=20, help="GLP: шагов обратного ОДУ")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", required=True)
    args = p.parse_args()

    model = steering.load_model(args.device)
    v = steering.vector(args.vector, model, args.device)
    nets = {k: load(getattr(args, k), args.device) for k in ("mse", "glp")
            if k in args.repair and getattr(args, k)}

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("steering")
    mlflow.start_run(run_name=args.tag)
    mlflow.log_params({k: str(x)[:250] for k, x in vars(args).items()})

    out = pathlib.Path("runs") / f"{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    rows = []
    for kind, repair in methods(args, nets):
        for alpha in args.alphas:
            hooks = [(steering.HOOK, steering.make_hook(v, alpha, repair))]
            samples = steering.generate(model, hooks, args.n_samples,
                                        args.max_new_tokens, args.seed)
            row = {"repair": kind, "alpha": alpha,
                   **steering.measure(model, samples, args.vector, args.concept_words,
                                      args.concept, v),
                   "sample": samples[0]["cont"]}
            rows.append(row)
            print(f"{kind:>8} alpha={alpha:6.1f}  ppl={row['ppl']:8.2f}  "
                  f"d2={row['dist2']:.3f}  concept={row['concept']:.3f}", flush=True)
            # пишем после каждой точки: фронт считается часами, и падение на
            # последней точке не должно стоить всех предыдущих
            out.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
            try:
                mlflow.log_metrics({f"{kind}_{k}": x for k, x in row.items()
                                    if isinstance(x, float)}, step=int(alpha))
            except Exception as exc:  # логирование не имеет права ронять эксперимент
                print(f"  mlflow не записал: {exc}", flush=True)
    mlflow.log_artifact(str(out))
    mlflow.end_run()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
    os._exit(0)
