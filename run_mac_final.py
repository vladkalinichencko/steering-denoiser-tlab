"""Train and evaluate the methods retained by the Mac screening gate."""

import json
import pathlib
import signal
import time

from tmp import screening
from tmp.training import train


DATA = "datasets/fineweb_layer6_mac_full.pt"
METHODS = (
    ("Additive MSE simple", "additive_simple"),
    ("Interpolation MSE", "interpolation"),
    ("GLP 20 steps", "glp"),
    ("MeanFlow", "meanflow"),
)
SEEDS = (0, 1, 2)
LIMIT_SECONDS = 6 * 60 * 60 + 45 * 60
EARLIER_METHODS = (
    "Additive MSE capacity", "Consistency",
    "Rectified 1 step", "Rectified 2 steps", "Rectified 4 steps",
    "Tangent-preserving MSE", "Nearest activation", "Segment-kNN",
    "Local tangent", "Local normal", "Local geodesic", "Safe capacity MSE",
    "Curveball", "INNSteer", "Conditional field / UniSteer",
)


def config(method: str, seed: int) -> dict:
    return {"tag": f"mac_final_{method}_seed{seed}", "method": method,
            "data": DATA, "steps": 2_000, "batch": 64, "lr": 5e-5,
            "sigma": 1.0, "seed": seed, "log_every": 200, "reduced": False}


def checkpoint(method: str, seed: int) -> pathlib.Path:
    return pathlib.Path("runs") / config(method, seed)["tag"] / "best.pt"


def add_earlier_screening() -> None:
    artifact_path = pathlib.Path("runs/screening.json")
    earlier_path = pathlib.Path("runs/screening.json")
    artifact = json.loads(artifact_path.read_text())
    earlier = json.loads(earlier_path.read_text())
    keys = ("method", "ratio", "nll", "property", "dist1", "dist2", "dist3",
            "latency_ms")
    artifact["earlier"] = {
        "label": "Earlier Mac screening · one seed or reduced budget",
        "source": "../mac_screening/screening.html",
        "points": [{key: point[key] for key in keys} for point in earlier["points"]
                   if point["method"] in EARLIER_METHODS],
    }
    screening.RUN = artifact_path.parent
    screening.save(artifact)


def main() -> None:
    started = time.monotonic()

    def timeout(_signum, _frame):
        raise TimeoutError("final Mac run reached its 6h45m safety deadline")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(LIMIT_SECONDS)
    for name, method in METHODS:
        for seed in SEEDS:
            path = checkpoint(method, seed)
            if not path.with_name("final.pt").exists():
                print(f"TRAIN {name} seed={seed} remaining={LIMIT_SECONDS-time.monotonic()+started:.0f}s",
                      flush=True)
                train(config(method, seed))

    chosen = ["Naive"]
    final = {}
    for name, method in METHODS:
        for seed in SEEDS:
            label = f"{name} · seed {seed}"
            path = checkpoint(method, seed).as_posix()
            final[label] = (path, 0.2 if method in {"glp", "meanflow"} else 0.5,
                            20 if method == "glp" else 1)
            chosen.append(label)
            if method == "glp":
                one = f"GLP one Euler · seed {seed}"
                final[one] = (path, 0.2, 1)
                chosen.append(one)
    screening.CHECKPOINTS = final
    screening.RUN = pathlib.Path("runs")
    screening.SEEDS = (0, 1, 2, 3, 4)
    screening.run(chosen)
    add_earlier_screening()
    print(f"DONE seconds={time.monotonic()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
