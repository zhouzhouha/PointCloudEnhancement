#!/usr/bin/env python3
"""Validate immutable benchmark metric CSV files without modifying them."""

import argparse
import csv
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--key-columns", nargs="+", required=True)
    parser.add_argument("--finite-columns", nargs="+", required=True)
    parser.add_argument(
        "--all-nonfinite-columns",
        nargs="*",
        default=[],
        help="Columns whose values must all be NaN/Inf placeholders",
    )
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit(f"empty CSV: {args.csv_path}")
    columns = set(rows[0])
    required = set(args.key_columns) | set(args.finite_columns) | set(args.all_nonfinite_columns)
    missing = sorted(required - columns)
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise SystemExit(f"row count {len(rows)} != {args.expected_rows}")

    keys = [tuple(row[name] for name in args.key_columns) for row in rows]
    duplicates = len(keys) - len(set(keys))
    nonfinite = []
    for index, row in enumerate(rows, start=2):
        for name in args.finite_columns:
            try:
                value = float(row[name])
            except ValueError:
                nonfinite.append((index, name, row[name]))
                continue
            if not math.isfinite(value):
                nonfinite.append((index, name, row[name]))

    unexpected_finite = []
    for index, row in enumerate(rows, start=2):
        for name in args.all_nonfinite_columns:
            try:
                value = float(row[name])
            except ValueError:
                continue
            if math.isfinite(value):
                unexpected_finite.append((index, name, row[name]))

    print(
        f"path={args.csv_path} rows={len(rows)} unique_keys={len(set(keys))} "
        f"duplicates={duplicates} finite_values={len(rows) * len(args.finite_columns) - len(nonfinite)} "
        f"nonfinite_values={len(nonfinite)} expected_nonfinite_values="
        f"{len(rows) * len(args.all_nonfinite_columns) - len(unexpected_finite)} "
        f"unexpected_finite_values={len(unexpected_finite)}"
    )
    if duplicates:
        raise SystemExit("duplicate metric keys found")
    if nonfinite:
        for item in nonfinite[:20]:
            print(f"nonfinite row={item[0]} column={item[1]} value={item[2]!r}")
        raise SystemExit("nonfinite metric values found")
    if unexpected_finite:
        for item in unexpected_finite[:20]:
            print(f"unexpected-finite row={item[0]} column={item[1]} value={item[2]!r}")
        raise SystemExit("expected placeholder columns contain finite values")


if __name__ == "__main__":
    main()
