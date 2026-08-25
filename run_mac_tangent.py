"""Train the reduced tangent-preserving repair used in the Mac screen."""

import pathlib

from training import prepare_tangent_bases, train

DATA = pathlib.Path("datasets/fineweb_layer6_mac.pt")
BASES = pathlib.Path("datasets/tangent_bases_mac_reduced.pt")

def main() -> None:
    if not BASES.exists():
        prepare_tangent_bases(BASES, DATA)
    train({"tag": "mac_reduced_tangent_mse", "method": "tangent_mse",
           "data": str(DATA), "bases": str(BASES), "steps": 300,
           "batch": 64, "lr": 5e-5, "sigma": 1.0, "seed": 0,
           "log_every": 30, "reduced": True})

if __name__ == "__main__":
    main()
