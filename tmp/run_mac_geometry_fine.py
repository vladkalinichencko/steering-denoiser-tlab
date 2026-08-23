import pathlib

from tmp import screening


METHODS = (
    "Naive",
    "Isotropic noise",
    "Additive MSE simple",
    "Additive MSE capacity",
    "Interpolation MSE",
    "GLP 20 steps",
    "GLP one Euler",
    "Consistency",
    "Rectified 1 step",
    "MeanFlow",
    "Tangent-preserving MSE",
    "Safe capacity MSE",
)


def main():
    screening.RUN = pathlib.Path("runs/mac_geometry_fine")
    screening.PROMPTS = screening.PROMPTS[:16]
    screening.SEEDS = (0,)
    screening.run(METHODS, (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8))


if __name__ == "__main__":
    main()
