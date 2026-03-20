import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_phase_diagram(tau_vals, eps_vals, phase_map, save_path=None):
    """
    Plot the phase diagram with color-coded regions.

    Colors: white=none, red=incoherent, blue=sync, purple=bistable.
    Overlays the tau=1/epsilon boundary line.
    """
    cmap = ListedColormap(["white", "lightcoral", "cornflowerblue", "mediumpurple"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(tau_vals, eps_vals, phase_map, cmap=cmap, shading="nearest", vmin=0, vmax=3)

    # Boundary line: tau = 1/epsilon
    eps_line = np.linspace(0.01, 0.6, 100)
    tau_line = 1.0 / eps_line
    ax.plot(tau_line, eps_line, "k:", linewidth=2, label=r"$\tau = 1/\epsilon$")

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel(r"Time Delay $\tau$")
    ax.set_ylabel(r"Coupling Strength $\epsilon$")
    ax.set_title("Simulation-based Stability Diagram (Fig. 3b equivalent)")

    # Legend
    ax.scatter([], [], c="lightcoral", label="Incoherent")
    ax.scatter([], [], c="cornflowerblue", label="Synchronized")
    ax.scatter([], [], c="mediumpurple", label="Bistable")
    ax.legend(loc="upper right")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")

    plt.close(fig)
