import os

os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
from pathlib import Path

# Config:
EVENLY_SPACED_X = True
LOG_Y_SCALE = True
LINES = False
MEASUREMENT_TYPE = "pixel"
RESULTS_ROOT = Path("results/mnist/")

fig, ax = plt.subplots(figsize=(10, 6))

if MEASUREMENT_TYPE == "pixel":
    sampling_rates = ["10", "25", "50", "100", "250", "500"]
    positions = [2, 5, 8, 11, 14, 17]
else:
    sampling_rates = ["2", "4", "8", "16", "24"]
    positions = [2, 5, 8, 11, 14]


entropy_data = [
    np.load(RESULTS_ROOT / f"{MEASUREMENT_TYPE}_entropy_{rate}.npy")
    for rate in sampling_rates
]
random_data = [
    np.load(RESULTS_ROOT / f"{MEASUREMENT_TYPE}_random_{rate}.npy")
    for rate in sampling_rates
]
variance_data = [
    np.load(RESULTS_ROOT / f"{MEASUREMENT_TYPE}_data_variance_{rate}.npy")
    for rate in sampling_rates
]


offset = 0.5

facecolours = {
    "random": "#FF5733",
    "data_variance": "#70cc8f",
    "entropy": "#337aff",
}
edgecolours = {
    "random": "#b32707",
    "data_variance": "#03ab41",
    "entropy": "#0941ab",
}

# Plot the violin plots
for i, rate in enumerate(sampling_rates):
    pos = positions[i]
    print(
        f"Random, rate={rate}, mean MAE={np.mean(random_data[i])}, std={np.std(random_data[i])}"
    )
    rand_vp = ax.violinplot(
        random_data[i],
        positions=[pos - offset],
        widths=0.5,
        showmeans=True,
        showmedians=False,
        showextrema=False,
    )
    for body in rand_vp["bodies"]:
        body.set_facecolor(facecolours["random"])
    rand_vp["cmeans"].set_edgecolor(edgecolours["random"])

    print(
        f"variance map, rate={rate}, mean MAE={np.mean(variance_data[i])}, std={np.std(variance_data[i])}"
    )
    var_vp = ax.violinplot(
        variance_data[i],
        positions=[pos],
        widths=0.5,
        showmeans=True,
        showmedians=False,
        showextrema=False,
    )
    for body in var_vp["bodies"]:
        body.set_facecolor(facecolours["data_variance"])
    var_vp["cmeans"].set_edgecolor(edgecolours["data_variance"])

    print(
        f"max ent, rate={rate}, mean MAE={np.mean(entropy_data[i])}, std={np.std(entropy_data[i])}"
    )
    ent_vp = ax.violinplot(
        entropy_data[i],
        positions=[pos + offset],
        widths=0.5,
        showmeans=True,
        showmedians=False,
        showextrema=False,
    )
    for body in ent_vp["bodies"]:
        body.set_facecolor(facecolours["entropy"])
    ent_vp["cmeans"].set_edgecolor(edgecolours["entropy"])

# need custom legend for violin plots unfortunately
legend_elements = [
    plt.Line2D([0], [0], color=edgecolours["random"], lw=3, label="Random"),
    plt.Line2D(
        [0], [0], color=edgecolours["data_variance"], lw=3, label="Data Variance"
    ),
    plt.Line2D([0], [0], color=edgecolours["entropy"], lw=3, label="ADS"),
]
ax.legend(
    handles=legend_elements,
    loc="upper right",
    borderpad=1,
    handlelength=1.5,
    fontsize=18,
)


# Customize the plot
ax.set_xticks(positions)
ax.set_xticklabels(sampling_rates)
ax.tick_params(axis="both", labelsize=20)
ax.set_xlabel(
    f"Number of {'pixel' if MEASUREMENT_TYPE == 'pixel' else 'line'}s sampled",
    fontsize=22,
)
ax.set_ylabel("MAE", fontsize=22)
# ax.set_title("Comparison of Subsamping Strategies", fontsize=22, pad=10)

# Show plot
plt.grid(True, axis="y", alpha=0.3)
plt.yscale("log" if LOG_Y_SCALE else "linear")
plt.tight_layout()
outfile = f"{MEASUREMENT_TYPE}_violin.pdf"
plt.savefig(outfile)
print(f"Plot saved to {outfile}")
