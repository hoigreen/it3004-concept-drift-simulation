from __future__ import annotations
import argparse
from pathlib import Path

from src.eda import run_eda
from src.experiment import run_experiment


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    eda = sub.add_parser("eda")
    eda.add_argument("--dataset", required=True,
                     choices=["nslkdd", "unsw", "cicids2018"])
    eda.add_argument("--data_dir", default="data")
    eda.add_argument("--out", default="outputs/eda")
    eda.add_argument("--sample", type=int, default=0)

    exp = sub.add_parser("exp")
    exp.add_argument("--dataset", required=True,
                     choices=["nslkdd", "unsw", "cicids2018"])
    exp.add_argument("--data_dir", default="data")
    exp.add_argument("--out", default="outputs/exp")
    exp.add_argument("--sample", type=int, default=0)

    exp.add_argument("--chunks", type=int, default=5)
    exp.add_argument("--window", type=int, default=2)
    exp.add_argument(
        "--detector", choices=["ks", "adwin", "ddm", "eddm"], default="ks")

    # drift knobs
    exp.add_argument("--new_start", type=int, default=0,
                     help="Chunk index (1-based) new attacks start; 0=auto mid")
    exp.add_argument("--new_frac", type=float, default=0.40,
                     help="Fraction of attacks in late chunks that are NEW")
    exp.add_argument("--benign_frac", type=float, default=0.70,
                     help="Benign fraction per chunk (approx)")
    exp.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    out = Path(args.out)

    if args.cmd == "eda":
        run_eda(dataset=args.dataset, data_dir=Path(
            args.data_dir), out_dir=out, sample=args.sample)
    else:
        run_experiment(
            dataset=args.dataset,
            data_dir=Path(args.data_dir),
            out_dir=out,
            sample=args.sample,
            n_chunks=args.chunks,
            window=args.window,
            detector=args.detector,
            new_start=args.new_start,
            new_frac=args.new_frac,
            benign_frac=args.benign_frac,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
