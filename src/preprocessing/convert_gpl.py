"""
convert_gpl.py
--------------
Converts the raw .gpl data file (from Lattice QCD) into structured .csv files
for each correlator label. Saves the results into `data/processed/`.
"""

import os
import re
import numpy as np


def convert_gpl_to_csv(input_path, output_dir):
    """
    Reads a .gpl file where each line starts with a label followed by numeric values.
    Groups by label and saves one CSV per label.
    """
    os.makedirs(output_dir, exist_ok=True)
    datasets = {}

    print(f"[INFO] Reading file: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            label = parts[0]  # e.g. 2pt_D_Gold_fine.ll

            # Convert numeric tokens
            numbers = []
            for p in parts[1:]:
                p = p.replace("D", "E").replace("d", "e")  # Fortran -> Python notation
                try:
                    numbers.append(float(p))
                except ValueError:
                    pass  # skip bad tokens

            if not numbers:
                continue

            datasets.setdefault(label, []).append(numbers)

            # Occasional progress print
            if i % 10000 == 0:
                print(f"[INFO] Processed {i} lines...")

    # Save each correlator as CSV
    for label, rows in datasets.items():
        arr = np.array(rows)
        safe_label = re.sub(r"[^\w\-]+", "_", label)
        out_path = os.path.join(output_dir, f"{safe_label}.csv")
        np.savetxt(out_path, arr, delimiter=",")
        print(f"[SAVED] {label} → {out_path} ({arr.shape[0]} rows × {arr.shape[1]} cols)")

    print("\n✅ Conversion complete!")
    print(f"Total correlator types: {len(datasets)}")
    total_lines = sum(len(v) for v in datasets.values())
    print(f"Total lines processed: {total_lines}")


if __name__ == "__main__":
    # Get the base project directory (2 levels up from this file)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Define input/output paths
    input_file = os.path.join(base_dir, "data", "raw",
                              "alldata_plusDgoldnongold_alsoqsqmaxdata_allstreams_allqsq_unbinned_final.gpl")
    output_folder = os.path.join(base_dir, "data", "processed")

    convert_gpl_to_csv(input_file, output_folder)
