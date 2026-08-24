"""Pareto front and repair cost from the final Mac screening artifacts."""

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = Path("runs/screening.json")
COLORS = {
    "Naive": "#64748b",
    "Additive MSE simple": "#0f766e",
    "Interpolation MSE": "#7c3aed",
    "GLP 20 steps": "#c2410c",
    "GLP one Euler": "#1d4ed8",
    "MeanFlow": "#b45309",
}


def grouped():
    points = json.loads(SOURCE.read_text())["points"]
    groups = {}
    for point in points:
        groups.setdefault(point["method"].split(" · seed")[0], []).append(point)
    return groups


def main():
    groups = grouped()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for method, points in groups.items():
        by_ratio = {}
        for point in points:
            by_ratio.setdefault(point["ratio"], []).append(point)
        ratios = sorted(by_ratio)
        nll = [st.mean(p["nll"] for p in by_ratio[r]) for r in ratios]
        prop = [st.mean(p["property"] for p in by_ratio[r]) for r in ratios]
        axes[0].plot(nll, prop, "o-", color=COLORS.get(method, "#94a3b8"),
                     label=method, linewidth=1.8, markersize=4)
    axes[0].set_xlabel("NLL (ниже лучше)")
    axes[0].set_ylabel("property (выше лучше)")
    axes[0].set_title("Pareto: качество текста против концепта")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)

    names, latency, colors = [], [], []
    for method, points in groups.items():
        target = min(points, key=lambda p: abs(p["property"] - 0.7))
        same = [p for p in points if abs(p["ratio"] - target["ratio"]) < 1e-9]
        names.append(method.replace(" ", "\n", 1))
        latency.append(st.mean(p["latency_ms"] for p in same))
        colors.append(COLORS.get(method, "#94a3b8"))
    order = sorted(range(len(names)), key=lambda i: latency[i])
    bars = axes[1].bar([names[i] for i in order], [latency[i] for i in order],
                       color=[colors[i] for i in order], width=0.6)
    for bar, index in zip(bars, order):
        axes[1].text(bar.get_x() + bar.get_width() / 2, latency[index],
                     f"{latency[index]:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("время вмешательства, мс/токен")
    axes[1].set_title("Стоимость repair при сопоставимом property")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].grid(alpha=0.25, axis="y")

    figure.tight_layout()
    out = Path("assets/pareto-and-cost.png")
    figure.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
