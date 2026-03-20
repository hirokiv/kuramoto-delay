import numpy as np


# ── Schedule factories ──────────────────────────────────────────────


def constant_schedule(value):
    """Return a callable that always returns *value*."""
    def _schedule(t):
        return value
    return _schedule


def quench_schedule(v0, v1, t_q):
    """Step function: v0 for t < t_q, v1 for t >= t_q."""
    def _schedule(t):
        return v0 if t < t_q else v1
    return _schedule


def periodic_schedule(base, A, Omega):
    """base + A * sin(Omega * t)."""
    def _schedule(t):
        return base + A * np.sin(Omega * t)
    return _schedule


# ── Simulator with time-varying parameters ──────────────────────────


def simulate_with_schedule(N, omega_0, dt, t_max,
                           eps_schedule, tau_schedule,
                           init_state="random", max_tau=10.0):
    """
    Simulate the delayed Kuramoto model with time-varying eps(t) and tau(t).

    Parameters
    ----------
    N : int
        Number of oscillators.
    omega_0 : float
        Natural frequency.
    dt : float
        Euler integration time step.
    t_max : float
        Total simulation time.
    eps_schedule : callable(float) -> float
        Coupling strength as a function of time.
    tau_schedule : callable(float) -> float
        Delay as a function of time.
    init_state : str
        "random" (R ≈ 0) or "sync" (R ≈ 1).
    max_tau : float
        Maximum delay expected during the simulation.
        Determines history buffer size.

    Returns
    -------
    t_array : ndarray, shape (steps,)
    R_array : ndarray, shape (steps,)
    """
    steps = int(t_max / dt)
    max_delay_steps = int(max_tau / dt)
    hist_len = max(1, max_delay_steps + 1)

    # History buffer
    history = np.zeros((hist_len, N))
    if init_state == "random":
        history[:] = np.random.uniform(0, 2 * np.pi, N)
    elif init_state == "sync":
        history[:] = np.random.uniform(0, 0.1, N)

    t_array = np.linspace(0, t_max, steps)
    R_array = np.empty(steps)

    for step in range(steps):
        t = step * dt
        eps = eps_schedule(t)
        tau = tau_schedule(t)
        delay_steps = max(0, int(tau / dt))

        current_theta = history[-1]

        if delay_steps > 0 and delay_steps < hist_len:
            delayed_theta = history[-(delay_steps + 1)]
        else:
            delayed_theta = current_theta

        # Coupling
        diff_matrix = delayed_theta[np.newaxis, :] - current_theta[:, np.newaxis]
        coupling = np.sum(np.sin(diff_matrix), axis=1)

        dtheta = omega_0 + (eps / N) * coupling
        next_theta = current_theta + dtheta * dt

        # Shift history
        history[:-1] = history[1:]
        history[-1] = next_theta

        R_array[step] = np.abs(np.mean(np.exp(1j * next_theta)))

    return t_array, R_array
