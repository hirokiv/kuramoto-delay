import os
import time


def main():
    from src.phase_diagram import generate_phase_diagram
    from src.plot import plot_phase_diagram, plot_timeseries

    os.makedirs("output", exist_ok=True)

    print("Starting delayed Kuramoto model simulation...")
    t0 = time.time()

    tau_vals, eps_vals, phase_map = generate_phase_diagram()

    elapsed = time.time() - t0
    print(f"Simulation complete in {elapsed:.1f}s")

    save_path = os.path.join("output", "phase_diagram.png")
    plot_phase_diagram(tau_vals, eps_vals, phase_map, save_path=save_path)

    print("Generating time series plots...")
    ts_path = os.path.join("output", "timeseries.png")
    plot_timeseries(save_path=ts_path)

    print("Done.")


if __name__ == "__main__":
    main()
