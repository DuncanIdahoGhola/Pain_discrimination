# Pain Discrimination During Placebo Analgesia

**Pain Learning Lab** - MP Coll lab, Laval University

## Overview

This project investigates whether placebo analgesia affects the ability to discriminate painful stimuli. Using placebo conditioning, we examine if participants who experience placebo-induced pain reduction show altered pain discrimination accuracy compared to control conditions.

### Research Questions

1. Does placebo analgesia reduce subjective pain ratings during placebo-active trials?
2. Does placebo analgesia impair the ability to discriminate the presence of painful stimuli?
3. Are individual differences in anxiety (STAI) and pain catastrophizing (PCS) related to placebo response or discrimination accuracy?

## Project Structure

```
Pain_discrimination/
├── sourcedata/           # Raw participant data (BIDS-like format)
│   └── sub-XXX_YYYY-MM-DD/
│       ├── QUEST staircase calibration data
│       ├── Main task data
│       └── Questionnaires (STAI-Y1, STAI-Y2, PCS, sociodemo)
├── derivatives/          # Analysis outputs
│   ├── placebo_0/        # Primary analysis (include all placebo responders)
│   ├── placebo_10/       # Sensitivity analysis (exclude <10% placebo effect)
│   ├── placebo_all/      # All participants regardless of placebo effect
│   ├── no_exclusions/    # No exclusion criteria applied
│   └── manuscript_results_all.csv  # Collated results across analyses
├── scripts/
│   ├── analyze.py        # Main analysis script
│   └── collate_results.py # Combines results across analyses
└── run_analysis.sh       # Convenience script to run all analyses
```

## Running the Analysis

```bash
# Run all analyses (recommended)
./run_analysis.sh

# Run individual analyses
./run_analysis.sh 0      # Primary: exclude_placebo = 0
./run_analysis.sh 10     # Sensitivity: exclude_placebo = 10
./run_analysis.sh all    # Include all placebo levels
./run_analysis.sh none   # No exclusions

# Or run directly with custom parameters
python3 scripts/analyze.py --exclude-placebo 0 --output-dir derivatives/placebo_0
```

### Exclusion Criteria

The `exclude_placebo` parameter controls participant inclusion based on reported pain reduction:
- `exclude_placebo = 0` (primary): Include participants with any placebo effect (>0%)
- `exclude_placebo = 10` (sensitivity): Exclude participants with <10% placebo effect

Additional exclusions applied:
- Perfect discrimination accuracy (ceiling effect)
- Below-chance discrimination accuracy (≤50%)
- Custom exclusions for data quality issues

## Dependencies

```bash
pip install numpy pandas scipy matplotlib seaborn pingouin statsmodels tqdm
```

## Analysis Pipeline

1. **Calibration**: Process QUEST staircase data to determine individual pain thresholds
2. **Evaluation Task**: Extract pain ratings with/without TENS across trial blocks
3. **Discrimination Task**: Analyze ability to detect pain stimulus presence by condition
4. **Questionnaires**: Compute STAI-Y1, STAI-Y2, and PCS scores
5. **Placebo Effect**: Calculate percentage pain reduction (active vs inactive TENS)
6. **Exclusions**: Apply criteria and generate non-overlapping exclusion lists
7. **Statistics**: Run t-tests, TOST equivalence tests, repeated measures ANOVAs, correlations
8. **Output**: Generate figures, summary CSVs, and manuscript-ready statistics

## Key Outputs

Each analysis directory contains:
- `stats/manuscript_results.csv` - All statistics for manuscript (single row)
- `stats/manuscript_results_readable.csv` - Same data, one statistic per row
- `figures/` - Publication-ready figures
- Individual statistical test outputs (t-tests, ANOVAs, correlations)

The collated `derivatives/manuscript_results_all.csv` provides side-by-side comparison across all analysis variants.

## Authors

Antoine Cyr-Bouchard, MP Coll
Laval University - Pain Learning Lab