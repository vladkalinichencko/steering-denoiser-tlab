"""Fixed short Mac run for the four baselines that must work before newer methods."""

import pathlib

from tmp.training import collect_activations, train

DATA = pathlib.Path("datasets/fineweb_layer6_mac_full.pt")

BASELINES = [
    ("mac_full_additive_simple", "additive_simple"),
    ("mac_full_additive_capacity", "additive_capacity"),
    ("mac_full_interpolation", "interpolation"),
    ("mac_full_glp", "glp"),
]


def config(tag: str, method: str) -> dict:
    return {"tag": tag, "method": method, "data": str(DATA), "steps": 2_000,
            "batch": 64, "lr": 5e-5, "sigma": 1.0, "seed": 0,
            "log_every": 200, "reduced": False}


def main() -> None:
    if not DATA.exists():
        collect_activations(DATA, n_train=100_000, n_val=2_000)
    for tag, method in BASELINES:
        train(config(tag, method))


if __name__ == "__main__":
    main()
