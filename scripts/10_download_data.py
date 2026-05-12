"""Download chess game data for SL bot training.

Default source: Lichess Elite Database (https://database.nikonoel.fr/),
pre-filtered to ~2300+ rated players. ~700MB-2GB compressed per month
(much smaller than the 25-50GB raw monthly Lichess dump).

Usage:
    python scripts/10_download_data.py
    # downloads + extracts to data/lichess_elite_2025-11.pgn

Override the URL for a different month or alternate source:
    python scripts/10_download_data.py --url https://database.nikonoel.fr/lichess_elite_2025-10.zip
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = "https://database.nikonoel.fr/lichess_elite_2025-11.zip"


def _download_with_progress(url: str, out_path: Path) -> None:
    print(f"[dl] {url}")
    print(f"     -> {out_path}")
    last_pct = -1

    def hook(blocknum, blocksize, totalsize):
        nonlocal last_pct
        if totalsize <= 0:
            return
        done = blocknum * blocksize
        pct = int(100 * done / totalsize)
        if pct != last_pct and pct % 5 == 0:
            print(f"  [{pct:>3}%] {done/1e6:>7.1f} / {totalsize/1e6:.1f} MB",
                  flush=True)
            last_pct = pct

    urllib.request.urlretrieve(url, out_path, reporthook=hook)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"download URL (default: {DEFAULT_URL})")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--no-extract", action="store_true",
                   help="skip unzipping")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_name = args.url.rstrip("/").split("/")[-1]
    zip_path = args.out_dir / zip_name

    if zip_path.exists():
        print(f"[skip-dl] {zip_path} already present "
              f"({zip_path.stat().st_size/1e6:.1f} MB)")
    else:
        _download_with_progress(args.url, zip_path)
        print(f"[done-dl] {zip_path.stat().st_size/1e6:.1f} MB")

    if args.no_extract:
        print(f"[ok] zip at {zip_path}")
        return 0

    if not zipfile.is_zipfile(zip_path):
        print(f"[warn] {zip_path} is not a zip; treating as final artifact")
        return 0

    print(f"[unzip] {zip_path}")
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            target = args.out_dir / Path(name).name
            if target.exists():
                print(f"  [skip] {target} exists")
            else:
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                print(f"  [extracted] {target} "
                      f"({target.stat().st_size/1e6:.1f} MB)")
            extracted.append(target)

    print(f"\n[ok] {len(extracted)} file(s) ready in {args.out_dir}")
    pgns = [e for e in extracted if e.suffix.lower() == ".pgn"]
    if pgns:
        print(f"[next] label with:")
        print(f"  python scripts/02_label_batch.py --source pgn \\")
        print(f"      --pgn-path {pgns[0]} --n 500000 \\")
        print(f"      --out data/labels_elite_500k.npz \\")
        print(f"      --workers 12 --depth 12 --multipv 5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
