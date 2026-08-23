import pathlib

from tmp import screening


METHODS = (
    "Naive",
    "Isotropic noise",
    "Curveball",
    "INNSteer",
    "Conditional field / UniSteer",
)


def main():
    screening.RUN = pathlib.Path("runs/mac_geometry_diagnostic")
    screening.PROMPTS = screening.PROMPTS[:16]
    screening.SEEDS = (0,)
    screening.run(METHODS, (0.0, 0.4, 0.8))


if __name__ == "__main__":
    main()
