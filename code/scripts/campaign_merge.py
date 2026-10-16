"""
Concatenate the row lists of split campaign jobs (PLAN: Phase 2; 16 Oct 2026): b22 / b21 jobs write one JSON list of
rows each; this joins them into one file for the figures.
Usage: python scripts/campaign_merge.py <results subdir> <out name> <part name> ...   (names without .json)
"""
import json
import os
import sys

from uwofdm.paths import output_dir


def main():
    sub, out, parts = sys.argv[1], sys.argv[2], sys.argv[3:]
    d = output_dir(sub)
    rows = []
    for p in parts:
        rows += json.load(open(os.path.join(d, p + ".json")))
    with open(os.path.join(d, out + ".json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"{sub}: {len(parts)} parts, {len(rows)} rows -> {out}.json")


if __name__ == "__main__":
    main()
