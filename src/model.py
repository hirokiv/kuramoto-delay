import numpy as np


def simulate_delayed_kuramoto(N, omega_0, epsilon, tau, dt, t_max, init_state="random"):
    """
    Simulate the delayed Kuramoto model using Euler method with a history buffer.

    Returns the final order parameter R.
    """
    steps = int(t_max / dt)
    delay_steps = int(tau / dt)

    # History buffer: history[0] = oldest, history[-1] = current
    hist_len = max(1, delay_steps + 1)
    history = np.zeros((hist_len, N))

    # Initial conditions
    if init_state == "random":
        history[:] = np.random.uniform(0, 2 * np.pi, N)  # R ≈ 0
    elif init_state == "sync":
        history[:] = np.random.uniform(0, 0.1, N)  # R ≈ 1

    for step in range(steps):
        current_theta = history[-1]

        if delay_steps > 0:
            delayed_theta = history[0]  # tau time ago
        else:
            delayed_theta = current_theta

        # Coupling: sum(sin(theta_k(t-tau) - theta_j(t)))
        diff_matrix = delayed_theta[np.newaxis, :] - current_theta[:, np.newaxis]
        coupling = np.sum(np.sin(diff_matrix), axis=1)

        # Eq. 1
        dtheta = omega_0 + (epsilon / N) * coupling

        # Euler step
        next_theta = current_theta + dtheta * dt

        # Shift history buffer
        history[:-1] = history[1:]
        history[-1] = next_theta

    # Final order parameter R
    final_theta = history[-1]
    R = np.abs(np.mean(np.exp(1j * final_theta)))
    return R
