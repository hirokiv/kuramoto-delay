import numpy as np
import multiprocessing
import os

from src.model_variable import (
    simulate_with_schedule,
    constant_schedule,
    quench_schedule,
    periodic_schedule,
)

# ── Shared constants ────────────────────────────────────────────────

N = 100
OMEGA_0 = np.pi / 2
DT = 0.05
R_THRESHOLD = 0.5

# ── Quench dataset ──────────────────────────────────────────────────

# Quench protocol timing
_Q_T_SETTLE = 50.0   # equilibration before quench
_Q_T_POST = 80.0     # observation after quench
_Q_T_MAX = _Q_T_SETTLE + _Q_T_POST  # 130s total
_Q_MAX_TAU = 10.0

# Transition type labels
TRANS_SYNC_TO_INCOH = 0
TRANS_INCOH_TO_SYNC = 1
TRANS_SYNC_TO_BIST = 2
TRANS_INCOH_TO_BIST = 3


def _sample_quench_params(n_pairs, rng):
    """Generate (eps, tau_0, tau_1, transition_type) tuples."""
    counts = {
        TRANS_SYNC_TO_INCOH: int(n_pairs * 0.40),
        TRANS_INCOH_TO_SYNC: int(n_pairs * 0.20),
        TRANS_SYNC_TO_BIST: int(n_pairs * 0.20),
    }
    counts[TRANS_INCOH_TO_BIST] = n_pairs - sum(counts.values())

    params = []
    for trans_type, count in counts.items():
        for _ in range(count):
            eps = rng.uniform(0.15, 0.5)
            boundary = 1.0 / eps  # tau = 1/eps is the phase boundary

            if trans_type == TRANS_SYNC_TO_INCOH:
                tau_0 = rng.uniform(max(0.1, boundary * 0.3), boundary * 0.9)
                tau_1 = rng.uniform(boundary * 1.2, min(boundary * 2.5, _Q_MAX_TAU))
            elif trans_type == TRANS_INCOH_TO_SYNC:
                tau_0 = rng.uniform(boundary * 1.2, min(boundary * 2.5, _Q_MAX_TAU))
                tau_1 = rng.uniform(max(0.1, boundary * 0.3), boundary * 0.9)
            elif trans_type == TRANS_SYNC_TO_BIST:
                tau_0 = rng.uniform(max(0.1, boundary * 0.3), boundary * 0.9)
                tau_1 = rng.uniform(boundary * 0.9, boundary * 1.1)
            else:  # INCOH_TO_BIST
                tau_0 = rng.uniform(boundary * 1.2, min(boundary * 2.5, _Q_MAX_TAU))
                tau_1 = rng.uniform(boundary * 0.9, boundary * 1.1)

            params.append((eps, tau_0, tau_1, trans_type))

    rng.shuffle(params)
    return params


def _quench_worker(args):
    """Worker for a single quench simulation. Receives a dict."""
    sample_id = args["sample_id"]
    eps = args["eps"]
    tau_0 = args["tau_0"]
    tau_1 = args["tau_1"]
    trans_type = args["trans_type"]
    init_state = args["init_state"]
    seed = args["seed"]

    np.random.seed(seed)

    eps_sched = constant_schedule(eps)
    tau_sched = quench_schedule(tau_0, tau_1, _Q_T_SETTLE)

    t_arr, R_arr = simulate_with_schedule(
        N, OMEGA_0, DT, _Q_T_MAX,
        eps_sched, tau_sched,
        init_state=init_state,
        max_tau=_Q_MAX_TAU,
    )

    settle_idx = int(_Q_T_SETTLE / DT)
    R_before = np.mean(R_arr[max(0, settle_idx - 20):settle_idx])
    R_after = np.mean(R_arr[-20:])

    return {
        "sample_id": sample_id,
        "R": R_arr,
        "tau_0": tau_0,
        "eps": eps,
        "tau_1": tau_1,
        "init_state": 0 if init_state == "random" else 1,
        "transition": trans_type,
        "R_before": R_before,
        "R_after": R_after,
    }


def generate_quench_dataset(n_pairs=200, master_seed=42, n_workers=None):
    """Generate quench dataset and save to output/dataset_quench.npz."""
    rng = np.random.RandomState(master_seed)
    pair_params = _sample_quench_params(n_pairs, rng)

    # Each pair → 2 samples (random + sync init)
    work_items = []
    for idx, (eps, tau_0, tau_1, trans_type) in enumerate(pair_params):
        for init_state in ("random", "sync"):
            sid = idx * 2 + (0 if init_state == "random" else 1)
            work_items.append({
                "sample_id": sid,
                "eps": eps,
                "tau_0": tau_0,
                "tau_1": tau_1,
                "trans_type": trans_type,
                "init_state": init_state,
                "seed": master_seed + sid,
            })

    total = len(work_items)
    steps = int(_Q_T_MAX / DT)
    print(f"Quench dataset: {total} samples, {n_workers or multiprocessing.cpu_count()} workers")

    # Allocate output arrays
    t_out = np.linspace(0, _Q_T_MAX, steps)
    R_out = np.empty((total, steps), dtype=np.float32)
    tau_0_out = np.empty(total)
    eps_out = np.empty(total)
    tau_1_out = np.empty(total)
    init_out = np.empty(total, dtype=np.int8)
    trans_out = np.empty(total, dtype=np.int8)
    R_before_out = np.empty(total)
    R_after_out = np.empty(total)

    with multiprocessing.Pool(processes=n_workers) as pool:
        for done, result in enumerate(pool.imap_unordered(_quench_worker, work_items), 1):
            sid = result["sample_id"]
            R_out[sid] = result["R"]
            tau_0_out[sid] = result["tau_0"]
            eps_out[sid] = result["eps"]
            tau_1_out[sid] = result["tau_1"]
            init_out[sid] = result["init_state"]
            trans_out[sid] = result["transition"]
            R_before_out[sid] = result["R_before"]
            R_after_out[sid] = result["R_after"]
            if done % 50 == 0 or done == total:
                print(f"  Progress: {done}/{total} ({100 * done // total}%)")

    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", "dataset_quench.npz")
    np.savez(
        path,
        t=t_out, R=R_out,
        tau_0=tau_0_out, eps=eps_out, tau_1=tau_1_out,
        init_state=init_out, transition=trans_out,
        R_before=R_before_out, R_after=R_after_out,
    )
    print(f"Saved {path}  ({total} samples, R shape {R_out.shape})")
    return path


# ── Periodic Forcing dataset ───────────────────────────────────────

_F_T_MAX = 200.0
_F_MAX_TAU = 10.0

# Base phase labels
PHASE_SYNC = 1
PHASE_INCOH = 2
PHASE_BIST = 3


def _sample_forcing_base_points(n_base, rng):
    """Sample base (tau, eps_0) points across 3 phase regions."""
    n_per = n_base // 3
    remainder = n_base - 3 * n_per
    counts = [n_per, n_per, n_per + remainder]  # sync, incoh, bist

    points = []
    for phase, count in zip([PHASE_SYNC, PHASE_INCOH, PHASE_BIST], counts):
        for _ in range(count):
            eps_0 = rng.uniform(0.15, 0.5)
            boundary = 1.0 / eps_0

            if phase == PHASE_SYNC:
                tau = rng.uniform(max(0.1, boundary * 0.2), boundary * 0.7)
            elif phase == PHASE_INCOH:
                tau = rng.uniform(boundary * 1.5, min(boundary * 3.0, _F_MAX_TAU))
            else:  # BIST
                tau = rng.uniform(boundary * 0.85, boundary * 1.15)

            points.append((tau, eps_0, phase))

    return points


def _forcing_worker(args):
    """Worker for a single periodic-forcing simulation."""
    sample_id = args["sample_id"]
    tau = args["tau"]
    eps_0 = args["eps_0"]
    A = args["A"]
    Omega = args["Omega"]
    init_state = args["init_state"]
    base_phase = args["base_phase"]
    seed = args["seed"]

    np.random.seed(seed)

    eps_sched = periodic_schedule(eps_0, A, Omega)
    tau_sched = constant_schedule(tau)

    t_arr, R_arr = simulate_with_schedule(
        N, OMEGA_0, DT, _F_T_MAX,
        eps_sched, tau_sched,
        init_state=init_state,
        max_tau=_F_MAX_TAU,
    )

    R_mean = float(np.mean(R_arr))
    R_std = float(np.std(R_arr))

    return {
        "sample_id": sample_id,
        "R": R_arr,
        "tau": tau,
        "eps_0": eps_0,
        "A": A,
        "Omega": Omega,
        "init_state": 0 if init_state == "random" else 1,
        "base_phase": base_phase,
        "R_mean": R_mean,
        "R_std": R_std,
    }


def generate_forcing_dataset(n_base=15, n_A=4, n_Omega=4,
                              master_seed=42, n_workers=None):
    """Generate periodic-forcing dataset and save to output/dataset_forcing.npz."""
    rng = np.random.RandomState(master_seed)
    base_points = _sample_forcing_base_points(n_base, rng)

    A_vals = np.logspace(np.log10(0.02), np.log10(0.2), n_A)
    Omega_vals = np.logspace(np.log10(0.05), np.log10(2.0), n_Omega)

    work_items = []
    sid = 0
    for tau, eps_0, phase in base_points:
        for A in A_vals:
            for Omega in Omega_vals:
                for init_state in ("random", "sync"):
                    work_items.append({
                        "sample_id": sid,
                        "tau": tau,
                        "eps_0": eps_0,
                        "A": A,
                        "Omega": Omega,
                        "init_state": init_state,
                        "base_phase": phase,
                        "seed": master_seed + sid,
                    })
                    sid += 1

    total = len(work_items)
    steps = int(_F_T_MAX / DT)
    print(f"Forcing dataset: {total} samples, {n_workers or multiprocessing.cpu_count()} workers")

    t_out = np.linspace(0, _F_T_MAX, steps)
    R_out = np.empty((total, steps), dtype=np.float32)
    tau_out = np.empty(total)
    eps_0_out = np.empty(total)
    A_out = np.empty(total)
    Omega_out = np.empty(total)
    init_out = np.empty(total, dtype=np.int8)
    phase_out = np.empty(total, dtype=np.int8)
    R_mean_out = np.empty(total)
    R_std_out = np.empty(total)

    with multiprocessing.Pool(processes=n_workers) as pool:
        for done, result in enumerate(pool.imap_unordered(_forcing_worker, work_items), 1):
            sid = result["sample_id"]
            R_out[sid] = result["R"]
            tau_out[sid] = result["tau"]
            eps_0_out[sid] = result["eps_0"]
            A_out[sid] = result["A"]
            Omega_out[sid] = result["Omega"]
            init_out[sid] = result["init_state"]
            phase_out[sid] = result["base_phase"]
            R_mean_out[sid] = result["R_mean"]
            R_std_out[sid] = result["R_std"]
            if done % 50 == 0 or done == total:
                print(f"  Progress: {done}/{total} ({100 * done // total}%)")

    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", "dataset_forcing.npz")
    np.savez(
        path,
        t=t_out, R=R_out,
        tau=tau_out, eps_0=eps_0_out, A=A_out, Omega=Omega_out,
        init_state=init_out, base_phase=phase_out,
        R_mean=R_mean_out, R_std=R_std_out,
    )
    print(f"Saved {path}  ({total} samples, R shape {R_out.shape})")
    return path
