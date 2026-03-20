import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.model import simulate_delayed_kuramoto


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


def plot_timeseries(save_path=None):
    """
    Plot R(t) time series for three representative (tau, eps) points:
    Synchronized, Incoherent, and Bistable.

    Each subplot shows two curves (random start vs sync start)
    with a horizontal dashed line at R=0.5.
    """
    N = 100
    omega_0 = np.pi / 2
    dt = 0.05
    t_max = 50.0

    cases = [
        {"title": "Synchronized", "tau": 0.5, "eps": 0.3},
        {"title": "Incoherent", "tau": 1.8, "eps": 0.3},
        {"title": "Bistable", "tau": 2.5, "eps": 0.3},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax, case in zip(axes, cases):
        _, t_rand, R_rand = simulate_delayed_kuramoto(
            N, omega_0, case["eps"], case["tau"], dt, t_max,
            init_state="random", return_timeseries=True,
        )
        _, t_sync, R_sync = simulate_delayed_kuramoto(
            N, omega_0, case["eps"], case["tau"], dt, t_max,
            init_state="sync", return_timeseries=True,
        )

        ax.plot(t_rand, R_rand, label="Random start", color="tab:orange")
        ax.plot(t_sync, R_sync, label="Sync start", color="tab:blue")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_title(rf"{case['title']} ($\tau$={case['tau']}, $\epsilon$={case['eps']})")
        ax.set_xlabel("Time")
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Order parameter R(t)")
    axes[0].legend(loc="center right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")

    plt.close(fig)
