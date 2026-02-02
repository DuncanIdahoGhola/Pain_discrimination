#!/bin/bash
#
# Run pain discrimination analysis with different placebo exclusion thresholds
#
# Usage:
#   ./run_analysis.sh           # Run all four analyses
#   ./run_analysis.sh 0         # Run only exclude_placebo=0
#   ./run_analysis.sh 10        # Run only exclude_placebo=10
#   ./run_analysis.sh all       # Run only exclude_placebo=-100 (include everyone)
#   ./run_analysis.sh none      # Run with no exclusions at all

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_analysis() {
    local threshold=$1
    local output_dir=$2
    local extra_args=$3

    echo "=========================================="
    echo "Running analysis with exclude_placebo = ${threshold}"
    echo "Output directory: ${output_dir}"
    echo "=========================================="

    python3 "${SCRIPT_DIR}/scripts/analyze.py" \
        --exclude-placebo "${threshold}" \
        --output-dir "${output_dir}" \
        ${extra_args}

    echo ""
}

run_no_exclusions() {
    local output_dir=$1

    echo "=========================================="
    echo "Running analysis with NO EXCLUSIONS"
    echo "Output directory: ${output_dir}"
    echo "=========================================="

    python3 "${SCRIPT_DIR}/scripts/analyze.py" \
        --no-exclusions \
        --output-dir "${output_dir}"

    echo ""
}

collate_results() {
    echo "=========================================="
    echo "Collating manuscript results"
    echo "=========================================="
    python3 "${SCRIPT_DIR}/scripts/collate_results.py"
    echo ""
}

if [ $# -eq 0 ]; then
    # No arguments: run all four analyses
    run_analysis 0 "derivatives/placebo_0"
    run_analysis 10 "derivatives/placebo_10"
    run_analysis -100 "derivatives/placebo_all"
    run_no_exclusions "derivatives/no_exclusions"
    collate_results
    echo "All four analyses finished!"
elif [ "$1" = "0" ]; then
    run_analysis 0 "derivatives/placebo_0"
    echo "Analysis finished!"
elif [ "$1" = "10" ]; then
    run_analysis 10 "derivatives/placebo_10"
    echo "Analysis finished!"
elif [ "$1" = "all" ] || [ "$1" = "-100" ]; then
    run_analysis -100 "derivatives/placebo_all"
    echo "Analysis finished!"
elif [ "$1" = "none" ]; then
    run_no_exclusions "derivatives/no_exclusions"
    echo "Analysis finished!"
else
    echo "Usage: $0 [0|10|all|none]"
    echo "  No arguments: run all four analyses"
    echo "  0: run only exclude_placebo=0 (primary analysis)"
    echo "  10: run only exclude_placebo=10 (sensitivity analysis)"
    echo "  all: run only exclude_placebo=-100 (include all placebo levels)"
    echo "  none: run with no exclusions at all"
    exit 1
fi
