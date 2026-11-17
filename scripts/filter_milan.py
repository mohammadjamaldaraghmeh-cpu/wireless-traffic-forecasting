"""Shrink the raw Telecom Italia Milan files to the few cells we need.

Harvard Dataverse (doi:10.7910/DVN/EGZHFV) requires a guestbook form before download, so
the daily files have to be downloaded by hand in the browser. Each one is ~350 MB and
covers all 10,000 grid cells. This script keeps only the rows of the chosen cells and
writes small files with the same name and format into data/milan, where src/data.py
reads them.

    python scripts/filter_milan.py C:\\Users\\me\\Downloads
    python scripts/filter_milan.py C:\\Users\\me\\Downloads --cells 5060 4259 4456
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw_dir", help="folder holding the downloaded sms-call-internet-mi-*.txt files")
    p.add_argument("--cells", type=int, nargs="*", default=[5060, 4259, 4456])
    p.add_argument("--out", default="data/milan")
    args = p.parse_args()

    files = sorted(Path(args.raw_dir).glob("sms-call-internet-mi-*.txt"))
    if not files:
        raise SystemExit(f"No sms-call-internet-mi-*.txt files in {args.raw_dir}")

    cells = {str(c).encode() for c in args.cells}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        kept = 0
        with open(f, "rb") as src, open(out_dir / f.name, "wb") as dst:
            for line in src:
                if line.split(b"\t", 1)[0] in cells:
                    dst.write(line)
                    kept += 1
        print(f"{f.name}: kept {kept} rows")


if __name__ == "__main__":
    main()
