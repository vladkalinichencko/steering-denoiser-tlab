"""Fit the reduced Curveball map on the fixed SST-5 contrast activations."""

import pathlib
import pickle

import torch

from tmp.nonlinear import Curveball


def main() -> None:
    source = torch.load("datasets/sst5_layer6_e51bdcd.pt", weights_only=False)
    positive = source["positive"][:256].float()
    negative = source["negative"][:256].float()
    mean = torch.cat((positive, negative)).mean(0)
    std = torch.cat((positive, negative)).std(0).clamp_min(1e-6)
    model = Curveball().fit((positive - mean) / std, (negative - mean) / std)
    output = pathlib.Path("runs/mac_reduced_curveball")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "model.pkl").open("wb") as file:
        pickle.dump({"model": model, "mean": mean, "std": std,
                     "config": {"train_per_class": 256, "components": 20,
                                "degree": 2, "gamma": 0.001,
                                "preimage": "Nadaraya-Watson"}}, file)


if __name__ == "__main__":
    main()
