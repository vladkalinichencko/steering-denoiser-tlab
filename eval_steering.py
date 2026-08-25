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
import methods
import steering

def load(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    a = blob["args"]
    net = denoiser.Denoiser(blob["d_act"], a["width"], a["expand"], a["n_blocks"],
                            "velocity" if a["objective"] == "flow" else "residual")
    net.load_state_dict(blob["model"])
    return net.to(device).eval()

def methods(args, nets, bank, v, alpha):
    """-> [(подпись, функция починки)]."""
    return [point for kind in args.repair
            for point in methods.BUILDERS[kind](args, nets, bank, v, alpha)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vector", default="diffmean:sentiment", help="sae:<i> | diffmean:<c>")
    p.add_argument("--concept-words", nargs="+", default=None, help="только для sae:*")
    p.add_argument("--concept", default="auto", choices=["auto", "lens", "latent", "words"],
                   help="чем мерить концепт: lens — доля токенов, которые продвигает само "
                        "направление; latent — активация латента; см. NOTES про то, что это "
                        "разные вещи")
    p.add_argument("--repair", nargs="+", default=["none"],
                   choices=["none", "mse", "glp", "glp1", "knn"],
                   help="glp1 — один шаг вместо двадцати; knn — притянуть к ближайшей "
                        "настоящей активации рядом с отрезком стиринга")
    p.add_argument("--mse", default=None, help="чекпойнт denoiser(h+eps)->h")
    p.add_argument("--glp", default=None, help="чекпойнт flow matching")
    p.add_argument("--geodesic", type=int, default=0,
                   help="стирить не одним прыжком, а N маленькими шагами, пересчитывая "
                        "локальное касательное направление")
    p.add_argument("--bank", type=int, default=100_000,
                   help="сколько настоящих активаций держать для knn и геодезической")
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 10, 20, 40, 80, 160])
    p.add_argument("--split", nargs="+", default=["none"],
                   choices=["none", "tangent", "normal"],
                   help="стирить полным вектором, его касательной или нормальной частью")
    p.add_argument("--tangent-dim", type=int, default=64)
    p.add_argument("--safe", action="store_true",
                   help="не давать починке отменять саму правку: убрать из поправки "
                        "составляющую вдоль v")
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
            if getattr(args, k)}
    bank = (steering.activation_bank(args.bank, args.device)
            if "knn" in args.repair or args.geodesic else None)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("steering")
    mlflow.start_run(run_name=args.tag)
    mlflow.log_params({k: str(x)[:250] for k, x in vars(args).items()})

    out = pathlib.Path("runs") / f"{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    basis = (steering.tangent_basis(args.tangent_dim, args.device)
             if set(args.split) - {"none"} else None)
    rows = []
    for part in args.split:
        vec = steering.split_vector(v, basis, part) if part != "none" else v
        for alpha in args.alphas:
            for kind, repair in methods(args, nets, bank, vec, alpha):
                hook = (steering.geodesic_hook(vec, alpha, bank, args.geodesic)
                        if args.geodesic and alpha
                        else steering.make_hook(vec, alpha, repair, args.safe))
                samples = steering.generate(model, [(steering.HOOK, hook)], args.n_samples,
                                            args.max_new_tokens, args.seed)
                label = "/".join(x for x in (kind, part if part != "none" else None,
                                             "geo" if args.geodesic else None) if x)
                row = {"repair": label, "alpha": alpha,
                       **steering.measure(model, samples, args.vector, args.concept_words,
                                          args.concept, vec),
                       "sample": samples[0]["cont"]}
                rows.append(row)
                print(f"{label:>16} alpha={alpha:6.1f}  ppl={row['ppl']:8.2f}  "
                      f"d2={row['dist2']:.3f}  concept={row['concept']:.3f}", flush=True)
                # пишем после каждой точки: фронт считается часами, и падение на
                # последней точке не должно стоить всех предыдущих
                out.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
                try:
                    mlflow.log_metrics({f"{label.replace('/', '_')}_{k}": x
                                        for k, x in row.items() if isinstance(x, float)},
                                       step=int(alpha))
                except Exception as exc:  # логирование не имеет права ронять эксперимент
                    print(f"  mlflow не записал: {exc}", flush=True)

    mlflow.log_artifact(str(out))
    mlflow.end_run()
    print(f"-> {out}")

if __name__ == "__main__":
    main()
    os._exit(0)
