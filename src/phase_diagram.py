import numpy as np
import multiprocessing

from src.model import simulate_delayed_kuramoto

# Fixed parameters matching the paper
N = 100
OMEGA_0 = np.pi / 2
DT = 0.05
T_MAX = 50.0
R_THRESHOLD = 0.5

TAU_VALS = np.linspace(0, 8, 30)
EPS_VALS = np.linspace(0, 0.6, 20)


def classify_point(params):
    """
    Worker function for multiprocessing.
    Runs two simulations (random + sync start) and classifies the point.

    Returns (i, j, classification):
        0 = None, 1 = Incoherent, 2 = Sync, 3 = Bistable
    """
    i, j, eps, tau = params

    R_from_random = simulate_delayed_kuramoto(N, OMEGA_0, eps, tau, DT, T_MAX, "random")
    R_from_sync = simulate_delayed_kuramoto(N, OMEGA_0, eps, tau, DT, T_MAX, "sync")

    is_sync_stable = R_from_sync > R_THRESHOLD
    is_incoh_stable = R_from_random < R_THRESHOLD

    if is_sync_stable and is_incoh_stable:
        classification = 3  # Bistable
    elif is_sync_stable:
        classification = 2  # Sync
    elif is_incoh_stable:
        classification = 1  # Incoherent
    else:
        classification = 0  # None

    return i, j, classification


def generate_phase_diagram(n_workers=None):
    """
    Build the (tau, epsilon) phase diagram using multiprocessing.

    Returns (tau_vals, eps_vals, phase_map).
    """
    tau_vals = TAU_VALS
    eps_vals = EPS_VALS

    phase_map = np.zeros((len(eps_vals), len(tau_vals)), dtype=int)

    # Build work items
    work_items = []
    for i, eps in enumerate(eps_vals):
        for j, tau in enumerate(tau_vals):
            work_items.append((i, j, eps, tau))

    total = len(work_items)
    print(f"Phase diagram: {total} grid points, {n_workers or multiprocessing.cpu_count()} workers")

    with multiprocessing.Pool(processes=n_workers) as pool:
        for done, (i, j, cls) in enumerate(pool.imap_unordered(classify_point, work_items), 1):
            phase_map[i, j] = cls
            if done % 50 == 0 or done == total:
                print(f"  Progress: {done}/{total} ({100 * done // total}%)")

    return tau_vals, eps_vals, phase_map
