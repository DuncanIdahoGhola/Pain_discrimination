#!/usr/bin/env python3
"""
Collate all manuscript results from different analysis runs into a single file.
"""

import os
import pandas as pd
from os.path import join as opj

def main():
    derivatives_dir = opj(os.path.dirname(os.path.dirname(__file__)), "derivatives")

    # Find all analysis subdirectories with manuscript results
    analysis_dirs = ["placebo_0", "placebo_10", "placebo_all"]

    results = {}
    for analysis in analysis_dirs:
        results_file = opj(derivatives_dir, analysis, "stats", "manuscript_results_readable.csv")
        if os.path.exists(results_file):
            df = pd.read_csv(results_file, header=None, names=["variable", analysis])
            # Skip header rows (statistic/value row)
            df = df[~df["variable"].isin(["statistic"])]
            df = df.set_index("variable")
            results[analysis] = df[analysis]
        else:
            print(f"Warning: {results_file} not found")

    if not results:
        print("No manuscript results found to collate.")
        return

    # Combine all results into a single DataFrame
    combined = pd.DataFrame(results)
    combined.index.name = "variable"

    # Save to derivatives root
    output_file = opj(derivatives_dir, "manuscript_results_all.csv")
    combined.to_csv(output_file)
    print(f"Collated manuscript results saved to {output_file}")

if __name__ == "__main__":
    main()
