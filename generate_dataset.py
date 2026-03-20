#!/usr/bin/env python
"""Generate ML training datasets for the delayed Kuramoto model.

Usage:
    python generate_dataset.py quench   [--n-pairs 200] [--seed 42]
    python generate_dataset.py forcing  [--n-base 15]   [--seed 42]
    python generate_dataset.py both     [--seed 42]
"""
import argparse
import time
import os


def main():
    parser = argparse.ArgumentParser(
        description="Generate quench / periodic-forcing datasets."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── quench ──
    p_q = sub.add_parser("quench", help="Generate quench dataset")
    p_q.add_argument("--n-pairs", type=int, default=200,
                      help="Number of (tau_0, tau_1) pairs (default: 200)")
    p_q.add_argument("--seed", type=int, default=42)
    p_q.add_argument("--workers", type=int, default=None)
    p_q.add_argument("--plot", action="store_true",
                      help="Generate validation plots after dataset creation")

    # ── forcing ──
    p_f = sub.add_parser("forcing", help="Generate periodic-forcing dataset")
    p_f.add_argument("--n-base", type=int, default=15,
                      help="Number of base (tau, eps) points (default: 15)")
    p_f.add_argument("--n-A", type=int, default=4)
    p_f.add_argument("--n-Omega", type=int, default=4)
    p_f.add_argument("--seed", type=int, default=42)
    p_f.add_argument("--workers", type=int, default=None)
    p_f.add_argument("--plot", action="store_true",
                      help="Generate validation plots after dataset creation")

    # ── both ──
    p_b = sub.add_parser("both", help="Generate both datasets + validation plots")
    p_b.add_argument("--n-pairs", type=int, default=200)
    p_b.add_argument("--n-base", type=int, default=15)
    p_b.add_argument("--n-A", type=int, default=4)
    p_b.add_argument("--n-Omega", type=int, default=4)
    p_b.add_argument("--seed", type=int, default=42)
    p_b.add_argument("--workers", type=int, default=None)

    args = parser.parse_args()

    from src.dataset import generate_quench_dataset, generate_forcing_dataset

    os.makedirs("output", exist_ok=True)

    if args.command in ("quench", "both"):
        print("=" * 60)
        print("Generating QUENCH dataset")
        print("=" * 60)
        t0 = time.time()
        q_path = generate_quench_dataset(
            n_pairs=args.n_pairs, master_seed=args.seed, n_workers=args.workers,
        )
        print(f"Quench done in {time.time() - t0:.1f}s\n")

    if args.command in ("forcing", "both"):
        print("=" * 60)
        print("Generating FORCING dataset")
        print("=" * 60)
        t0 = time.time()
        f_path = generate_forcing_dataset(
            n_base=args.n_base, n_A=args.n_A, n_Omega=args.n_Omega,
            master_seed=args.seed, n_workers=args.workers,
        )
        print(f"Forcing done in {time.time() - t0:.1f}s\n")

    # Validation plots
    do_plot = (args.command == "both") or getattr(args, "plot", False)
    if do_plot:
        from src.plot_dataset import plot_quench_validation, plot_forcing_validation
        save_dir = os.path.join("output", "validation")
        os.makedirs(save_dir, exist_ok=True)

        if args.command in ("quench", "both"):
            plot_quench_validation(
                os.path.join("output", "dataset_quench.npz"), save_dir,
            )
        if args.command in ("forcing", "both"):
            plot_forcing_validation(
                os.path.join("output", "dataset_forcing.npz"), save_dir,
            )

    print("All done.")


if __name__ == "__main__":
    main()
