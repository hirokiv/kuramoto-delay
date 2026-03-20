import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.dataset import (
    _Q_T_SETTLE,
    TRANS_SYNC_TO_INCOH, TRANS_INCOH_TO_SYNC,
    TRANS_SYNC_TO_BIST, TRANS_INCOH_TO_BIST,
    PHASE_SYNC, PHASE_INCOH, PHASE_BIST,
)

_TRANS_LABELS = {
    TRANS_SYNC_TO_INCOH: "Sync → Incoh",
    TRANS_INCOH_TO_SYNC: "Incoh → Sync",
    TRANS_SYNC_TO_BIST: "Sync → Bistable",
    TRANS_INCOH_TO_BIST: "Incoh → Bistable",
}

_PHASE_LABELS = {
    PHASE_SYNC: "Sync base",
    PHASE_INCOH: "Incoh base",
    PHASE_BIST: "Bistable base",
}


# ── Quench validation ──────────────────────────────────────────────


def plot_quench_validation(dataset_path, save_dir):
    """Generate quench validation plots."""
    d = np.load(dataset_path)
    t = d["t"]
    R = d["R"]
    trans = d["transition"]
    init = d["init_state"]

    # ── 1. Trajectory gallery: 4 rows (transition type) × 3 cols ──
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True, sharey=True)
    for row, tr_type in enumerate([TRANS_SYNC_TO_INCOH, TRANS_INCOH_TO_SYNC,
                                    TRANS_SYNC_TO_BIST, TRANS_INCOH_TO_BIST]):
        mask = trans == tr_type
        idxs = np.where(mask)[0]
        for col in range(3):
            ax = axes[row, col]
            if col < len(idxs):
                i = idxs[col]
                color = "tab:blue" if init[i] == 1 else "tab:orange"
                label = "sync" if init[i] == 1 else "random"
                ax.plot(t, R[i], color=color, linewidth=0.6, label=label)
                ax.axvline(_Q_T_SETTLE, color="red", linestyle="--",
                           linewidth=0.8, alpha=0.7)
                ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)
                ax.legend(fontsize=7, loc="upper right")
            if col == 0:
                ax.set_ylabel(_TRANS_LABELS[tr_type], fontsize=8)
            if row == 3:
                ax.set_xlabel("Time")
            ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Quench Trajectories (red dashed = quench time)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(save_dir, "quench_trajectories.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # ── 2. ΔR histogram by transition type ──
    R_before = d["R_before"]
    R_after = d["R_after"]
    dR = R_after - R_before

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, tr_type in zip(axes, [TRANS_SYNC_TO_INCOH, TRANS_INCOH_TO_SYNC,
                                   TRANS_SYNC_TO_BIST, TRANS_INCOH_TO_BIST]):
        mask = trans == tr_type
        ax.hist(dR[mask], bins=25, edgecolor="black", linewidth=0.5, alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(_TRANS_LABELS[tr_type], fontsize=9)
        ax.set_xlabel("ΔR = R_after − R_before")

    axes[0].set_ylabel("Count")
    fig.suptitle("ΔR Distribution by Transition Type", fontsize=12)
    fig.tight_layout()
    path = os.path.join(save_dir, "quench_deltaR.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Forcing validation ─────────────────────────────────────────────


def plot_forcing_validation(dataset_path, save_dir):
    """Generate periodic-forcing validation plots."""
    d = np.load(dataset_path)
    t = d["t"]
    R = d["R"]
    base_phase = d["base_phase"]
    eps_0 = d["eps_0"]
    A = d["A"]
    Omega = d["Omega"]
    R_std = d["R_std"]

    # ── 1. Trajectory gallery: 3 rows (base phase) × 3 cols ──
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True, sharey=True)
    for row, phase in enumerate([PHASE_SYNC, PHASE_INCOH, PHASE_BIST]):
        mask = base_phase == phase
        idxs = np.where(mask)[0]
        for col in range(3):
            ax = axes[row, col]
            if col < len(idxs):
                i = idxs[col]
                ax.plot(t, R[i], color="tab:blue", linewidth=0.6, label="R(t)")
                # Overlay eps(t) on secondary axis
                ax2 = ax.twinx()
                eps_t = eps_0[i] + A[i] * np.sin(Omega[i] * t)
                ax2.plot(t, eps_t, color="tab:red", linewidth=0.4, alpha=0.5)
                ax2.set_ylim(0, 0.8)
                if col == 2:
                    ax2.set_ylabel(r"$\epsilon(t)$", color="tab:red", fontsize=8)
                else:
                    ax2.set_yticklabels([])
            if col == 0:
                ax.set_ylabel(_PHASE_LABELS[phase], fontsize=8)
            if row == 2:
                ax.set_xlabel("Time")
            ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Forcing Trajectories (blue=R, red=ε(t))", fontsize=12)
    fig.tight_layout()
    path = os.path.join(save_dir, "forcing_trajectories.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # ── 2. Response heatmap: R_std on (A, Ω) plane, one per base phase ──
    A_unique = np.sort(np.unique(np.round(A, 6)))
    Omega_unique = np.sort(np.unique(np.round(Omega, 6)))

    if len(A_unique) < 2 or len(Omega_unique) < 2:
        print("  Skipping heatmap (not enough A or Omega values)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, phase in zip(axes, [PHASE_SYNC, PHASE_INCOH, PHASE_BIST]):
        mask = base_phase == phase
        grid = np.full((len(Omega_unique), len(A_unique)), np.nan)

        for i in np.where(mask)[0]:
            ai = np.argmin(np.abs(A_unique - A[i]))
            oi = np.argmin(np.abs(Omega_unique - Omega[i]))
            # Average R_std across samples at this (A, Omega)
            if np.isnan(grid[oi, ai]):
                grid[oi, ai] = R_std[i]
            else:
                grid[oi, ai] = (grid[oi, ai] + R_std[i]) / 2

        im = ax.pcolormesh(
            A_unique, Omega_unique, grid,
            shading="nearest", cmap="viridis",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Amplitude A")
        ax.set_ylabel(r"Frequency $\Omega$")
        ax.set_title(_PHASE_LABELS[phase], fontsize=9)
        fig.colorbar(im, ax=ax, label="R_std")

    fig.suptitle("Response Map: R_std on (A, Ω) plane", fontsize=12)
    fig.tight_layout()
    path = os.path.join(save_dir, "forcing_response_map.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
