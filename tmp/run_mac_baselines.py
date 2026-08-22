"""Fixed short Mac run for the four baselines that must work before newer methods."""

import pathlib

from tmp.training import collect_activations, train

DATA = pathlib.Path("datasets/fineweb_layer6_mac.pt")

BASELINES = [
    ("mac_additive_simple", "additive_simple"),
    ("mac_additive_capacity", "additive_capacity"),
    ("mac_interpolation", "interpolation"),
    ("mac_glp", "glp"),
]


def config(tag: str, method: str) -> dict:
    return {"tag": tag, "method": method, "data": str(DATA), "steps": 1_000,
            "batch": 128, "lr": 5e-5, "sigma": 1.0, "seed": 0,
            "log_every": 100, "reduced": True}


def main() -> None:
    if not DATA.exists():
        collect_activations(DATA, n_train=20_000, n_val=1_000)
    for tag, method in BASELINES:
        train(config(tag, method))


if __name__ == "__main__":
    main()
