"""Run the selected few-step methods after the GLP baseline passes diagnostics."""

import pathlib

from tmp.training import prepare_rectified_pairs, train

DATA = "datasets/fineweb_layer6_mac.pt"
GLP = pathlib.Path("runs/mac_glp/best.pt")
PAIRS = pathlib.Path("datasets/rectified_pairs_mac_reduced.pt")


def config(tag: str, method: str) -> dict:
    return {"tag": tag, "method": method, "data": DATA, "steps": 300,
            "batch": 64, "lr": 5e-5, "sigma": 1.0, "seed": 0,
            "log_every": 30, "reduced": True}


def main() -> None:
    train(config("mac_reduced_consistency", "consistency"))
    train(config("mac_reduced_meanflow", "meanflow"))
    if not PAIRS.exists():
        prepare_rectified_pairs(PAIRS, GLP, n_train=512, n_val=128)
    rectified = config("mac_reduced_rectified", "rectified")
    rectified["pairs"] = str(PAIRS)
    train(rectified)


if __name__ == "__main__":
    main()
