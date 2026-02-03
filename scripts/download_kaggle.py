import argparse
from pathlib import Path
import kagglehub

DATASETS = {
    "nslkdd": "hassan06/nslkdd",
    "unsw": "harshwardhanbhangale/unsw-complete-dataset",
    "cicids2018": "dhoogla/csecicids2018",
}


def download_one(slug: str) -> Path:
    """
    Downloads dataset to kagglehub cache and returns local directory path.
    """
    path = kagglehub.dataset_download(slug)
    p = Path(path)
    print(f"[kagglehub] Downloaded: {slug}")
    print(f"[kagglehub] Local path: {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", choices=list(DATASETS.keys()))
    args = ap.parse_args()

    if args.all:
        for name, slug in DATASETS.items():
            download_one(slug)
    else:
        if not args.only:
            raise SystemExit("Use --all or --only <dataset>")
        download_one(DATASETS[args.only])


if __name__ == "__main__":
    main()
