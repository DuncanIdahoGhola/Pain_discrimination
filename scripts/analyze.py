#!/usr/bin/env python3
"""
Pain Discrimination Analysis Script

Analyzes pain discrimination during placebo conditioning. This study investigates
whether placebo analgesia affects the ability to discriminate painful stimuli.

The analysis pipeline:
1. Processes QUEST staircase data (pain threshold calibration)
2. Extracts evaluation task trials (pain ratings with/without TENS)
3. Extracts discrimination task trials (detect pain stimulus presence)
4. Computes questionnaire scores (STAI-Y1, STAI-Y2, PCS)
5. Calculates placebo effect metrics
6. Applies exclusion criteria
7. Generates figures and statistical analyses
8. Compiles manuscript-ready results

Usage:
    python scripts/analyze.py --exclude-placebo 0 --output-dir derivatives/placebo_0
    python scripts/analyze.py --exclude-placebo 10 --output-dir derivatives/placebo_10

Authors: Antoine Cyr-Bouchard, MP Coll - Pain Learning Lab, Laval University
"""

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import math
import os
from os.path import join as opj

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy
from scipy import stats
from scipy.stats import norm
import seaborn as sns
from matplotlib.patches import Patch
from statsmodels.stats.weightstats import ttost_paired
from tqdm import tqdm
import urllib.request


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run pain discrimination analysis with configurable parameters."
    )
    parser.add_argument(
        "--exclude-placebo",
        type=int,
        default=0,
        help="Exclude participants with less than X%% placebo effect (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="derivatives",
        help="Directory for output files (default: derivatives)",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Source data directory (default: sourcedata/ in project root)",
    )
    parser.add_argument(
        "--low-acc-threshold",
        type=float,
        default=0.50,
        help="Exclude participants with accuracy <= this threshold (default: 0.50)",
    )
    parser.add_argument(
        "--no-exclusions",
        action="store_true",
        help="Disable all exclusions (include all participants)",
    )
    return parser.parse_args()


def setup_output_dirs(output_dir):
    """Create output directory structure."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(opj(output_dir, "figures"), exist_ok=True)
    os.makedirs(opj(output_dir, "stats"), exist_ok=True)


def setup_fonts(output_dir):
    """Download and configure fonts for plots."""
    font_path = opj(output_dir, "figures", "arialnarrow.ttf")
    if not os.path.exists(font_path):
        urllib.request.urlretrieve(
            "https://github.com/gbif/analytics/raw/master/fonts/Arial%20Narrow.ttf",
            font_path,
        )
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = prop.get_name()


def compute_questionnaire_scores(bidsroot):
    """
    Load questionnaire data from sourcedata and compute total scores.

    Returns a DataFrame with computed totals (iastay1_total, iastay2_total, pcs_total)
    without modifying the original sourcedata files.
    """
    iastay1 = pd.read_csv(opj(bidsroot, "iasta_y1.csv"))
    iastay2 = pd.read_csv(opj(bidsroot, "iasta_y2.csv"))
    pcs = pd.read_csv(opj(bidsroot, "pcs.csv"))

    results_df = pd.DataFrame(index=iastay1.index)

    for p in iastay1.index:
        # IASTA Y2
        all_iasta2 = []
        for c in range(2, len(iastay2.columns)):
            try:
                results_df.loc[p, "qiastay2_" + list(iastay2.columns)[c]] = int(
                    str(iastay2.loc[p, list(iastay2.columns)[c]])[0]
                )
            except:
                results_df.loc[p, "qiastay2_" + list(iastay2.columns)[c]] = "nan"
            try:
                all_iasta2.append(int(str(iastay2.loc[p, list(iastay2.columns)[c]])[0]))
            except:
                all_iasta2.append(np.nan)
        assert len(all_iasta2) == 20
        # Invert scores for some columns [0, 2, 5, 6, 9, 12, 13, 15, 18]
        all_iasta2 = np.asarray(all_iasta2)
        all_iasta2[[0, 2, 5, 6, 9, 12, 13, 15, 18]] = (
            5 - all_iasta2[[0, 2, 5, 6, 9, 12, 13, 15, 18]]
        )
        results_df.loc[p, "iastay2_total"] = np.nansum(all_iasta2)

        # IASTA Y1
        all_iasta1 = []
        for c in range(2, len(iastay1.columns)):
            try:
                results_df.loc[p, "qiastay1_" + list(iastay1.columns)[c]] = int(
                    str(iastay1.loc[p, list(iastay1.columns)[c]])[0]
                )
            except:
                results_df.loc[p, "qiastay1_" + list(iastay1.columns)[c]] = "nan"
            try:
                all_iasta1.append(int(str(iastay1.loc[p, list(iastay1.columns)[c]])[0]))
            except:
                all_iasta1.append(np.nan)
        assert len(all_iasta1) == 20
        # Invert scores for some columns [0, 1, 4, 7, 9, 10, 14, 15, 18, 19]
        all_iasta1 = np.asarray(all_iasta1)
        all_iasta1[[0, 1, 4, 7, 9, 10, 14, 15, 18, 19]] = (
            5 - all_iasta1[[0, 1, 4, 7, 9, 10, 14, 15, 18, 19]]
        )
        results_df.loc[p, "iastay1_total"] = np.nansum(all_iasta1)

        # PCS
        all_pcs = []
        for c in range(2, len(pcs.columns)):
            try:
                results_df.loc[p, "qpcs_" + list(pcs.columns)[c]] = int(
                    str(pcs.loc[p, list(pcs.columns)[c]])[0]
                )
                all_pcs.append(int(str(pcs.loc[p, list(pcs.columns)[c]])[0]))
            except:
                results_df.loc[p, "qpcs_" + list(pcs.columns)[c]] = "nan"
        if len(all_pcs) == 13:
            results_df.loc[p, "pcs_total"] = np.nansum(all_pcs)
        else:
            results_df.loc[p, "pcs_total"] = "nan"

    return results_df


def SDT(hits, misses, fas, crs):
    """
    Calculate signal detection theory measures.

    Returns a dict with d-prime measures given hits, misses, false alarms,
    and correct rejections.
    """
    Z = norm.ppf
    # Floors and ceilings are replaced by half hits and half FA's
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)

    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    if hit_rate == 1:
        hit_rate = 1 - half_hit
    if hit_rate == 0:
        hit_rate = half_hit

    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    if fa_rate == 1:
        fa_rate = 1 - half_fa
    if fa_rate == 0:
        fa_rate = half_fa

    # Return d', beta, c and Ad'
    out = {}
    out["d"] = Z(hit_rate) - Z(fa_rate)
    out["beta"] = math.exp((Z(fa_rate) ** 2 - Z(hit_rate) ** 2) / 2)
    out["c"] = -(Z(hit_rate) + Z(fa_rate)) / 2
    out["Ad"] = norm.cdf(out["d"] / math.sqrt(2))

    return out


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================
def main():
    """
    Main analysis pipeline for pain discrimination study.

    This function orchestrates the entire analysis:
    1. Setup: Parse arguments, configure paths and output directories
    2. Individual processing: Loop through participants to extract data
    3. Aggregation: Combine individual data into group-level dataframes
    4. Exclusions: Apply exclusion criteria and track excluded participants
    5. Visualization: Generate publication-ready figures
    6. Statistics: Run inferential tests (ANOVAs, t-tests, correlations)
    7. Output: Save results and compile manuscript statistics
    """
    args = parse_args()

    # -------------------------------------------------------------------------
    # SETUP: Configure paths and directories
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    if args.source_dir:
        bidsroot = args.source_dir
    else:
        bidsroot = opj(project_dir, "sourcedata")

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = opj(project_dir, output_dir)

    # Initialize output directories and fonts for plotting
    setup_output_dirs(output_dir)
    setup_fonts(output_dir)

    # -------------------------------------------------------------------------
    # PARAMETERS: Configure exclusion criteria
    # -------------------------------------------------------------------------
    if args.no_exclusions:
        exclude_perfect = False
        exclude_placebo = None
        exclude_low_acc = False
    else:
        exclude_perfect = True
        exclude_placebo = args.exclude_placebo
        exclude_low_acc = True
    exclude_custom = ["sub-052", "sub-057", "sub-070"]
    low_acc_threshold = args.low_acc_threshold

    print(f"Running analysis with exclude_placebo = {exclude_placebo}")
    print(f"Output directory: {output_dir}")

    # -------------------------------------------------------------------------
    # PARTICIPANT DISCOVERY: Find all participant folders in sourcedata
    # -------------------------------------------------------------------------
    participants = [p.split("_")[0] for p in os.listdir(bidsroot) if "sub-" in p]
    participants.sort()

    # Create empty lists to store data
    all_active, all_thresholds, all_plateaus = [], [], []
    all_eval_frames, all_discrim_task = [], []
    all_discrim_task_long = []

    # Create empty dataframe to store wide data
    wide_dat = pd.DataFrame(
        index=participants,
        columns=[
            "acc_all",
            "active_acc_all",
            "inactive_acc_all",
            "active_acc_b1",
            "inactive_acc_b1",
            "active_acc_b2",
            "inactive_acc_b2",
            "temp_pic",
            "temp_plateau",
            "temp_placebo",
            "average_placebo_all",
            "average_placebo_b1",
            "average_placebo_b2",
            "perc_placebo_b1",
            "perc_placebo_b2",
            "perc_placebo_all",
            "average_eval_inactive",
            "average_eval_active",
            "average_eval_active_b1",
            "average_eval_active_b2",
            "average_eval_inactive_b1",
            "average_eval_inactive_b2",
        ],
    )

    # =========================================================================
    # INDIVIDUAL PARTICIPANT PROCESSING
    # Loop through each participant to extract and process their data
    # =========================================================================
    for p in tqdm(participants, desc="Processing individual participants"):
        # Get participant folder and create derivatives folder
        par_fold = [c for c in os.listdir(bidsroot) if p in c]
        assert len(par_fold) == 1
        par_fold = opj(bidsroot, par_fold[0])
        deriv_path = opj(output_dir, p)
        if not os.path.exists(deriv_path):
            os.mkdir(deriv_path)

        # ---------------------------------------------------------------------
        # QUEST STAIRCASE: Process pain threshold calibration data
        # The QUEST procedure determines each participant's pain detection
        # threshold using an adaptive staircase method
        # ---------------------------------------------------------------------
        quest_dir = opj(par_fold, "QUEST")
        quest_file = [f for f in os.listdir(quest_dir) if ".csv" in f and "trial" not in f]

        if len(quest_file) > 1:
            quest_file = [quest_file[-1]]

        for idx, f in enumerate(quest_file):
            quest = pd.read_csv(os.path.join(par_fold, "QUEST", quest_file[idx]))
            plateau = quest["temp_plateau"].values[0]
            quest = quest[quest["pic_response"] != "None"]
            trials_num = quest["trials.thisRepN"].dropna().values
            quest.loc[quest["pic_response"].isna(), "pic_sent"] = np.nan
            quest_intensities = quest["pic_sent"].dropna().values
            quest_detections = quest["pic_response"].dropna().values

            if "sub-012" in p:
                thresh = 0.5 + plateau
            elif int(p[4:7]) < 7:
                thresh = quest["threshold"].dropna().values[0]
            else:
                thresh = quest["mean_6_reversals"].dropna().values[0] + plateau

            combinedInten = quest_intensities.astype(float)
            combinedResp = quest_detections.astype(float)

            thresh = float(thresh)
            plateau = float(plateau)
            plt.figure()
            plt.title("threshold = %0.3f" % (thresh) + " plateau = %0.1f" % (plateau))
            plt.axvline(x=thresh * 10, color="k", linestyle="--")
            plt.plot(quest_intensities, quest_detections, "o")
            plt.ylim([-0.5, 1.5])
            plt.savefig(opj(deriv_path, p + "_quest.png"))

            # Plot temp - all trials
            temp_files = [
                s
                for s in os.listdir(quest_dir)
                if "temp" in s and s.split("_")[5][:11] in f and ".csv" in s
            ]
            plt.figure(figsize=(5, 5))
            for trial in temp_files:
                temp = pd.read_csv(opj(quest_dir, trial))
                avg_temp = np.average(temp[["z1", "z2", "z3", "z4", "z5"]], axis=1)
                plt.plot(np.arange(len(temp)), avg_temp)
                plt.xlabel("Sample")
                plt.ylabel("Temperature")
            plt.axhline(y=thresh, color="k", linestyle="--", label="threshold")
            plt.axhline(y=plateau, color="r", linestyle="--", label="plateau")
            plt.legend()
            plt.savefig(opj(deriv_path, p + "_quest_" + str(idx).zfill(2) + "_temps.png"))

            # Plot temp - individual trials
            fig, axes = plt.subplots(6, 6, figsize=(10, 10))
            axes = axes.flatten()
            for trial in temp_files:
                trial_num = int(trial.split("_")[-1].replace(".csv", ""))
                if trial_num in trials_num:
                    temp = pd.read_csv(opj(quest_dir, trial))
                    axes[trial_num].set_title(trial_num)
                    axes[trial_num].set_ylim(42, plateau + 2)
                    pic = quest.loc[trial_num, "pic_sent"] / 10
                    axes[trial_num].plot(
                        np.arange(len(temp)),
                        np.average(temp[["z1", "z2", "z3", "z4", "z5"]], 1),
                        label=trial_num,
                    )
                    axes[trial_num].axhline(y=pic, color="k", linestyle="--", label="pic")

            fig.suptitle(p + " all quest trials")
            plt.tight_layout()
            plt.xlabel("Sample")
            plt.ylabel("Temperature")
            plt.savefig(opj(deriv_path, p + "_quest_" + str(idx).zfill(2) + "_temps_all.png"))

        # ---------------------------------------------------------------------
        # MAIN TASK DATA: Load and process evaluation and discrimination tasks
        # ---------------------------------------------------------------------
        main_file = [
            f
            for f in os.listdir(par_fold)
            if "maintask" in f and "trial" not in f and ".csv" in f
        ]
        assert len(main_file) == 1

        main = pd.read_csv(os.path.join(par_fold, main_file[0]))
        if len(main.columns) == 1:
            main = pd.read_csv(os.path.join(par_fold, main_file[0]), sep=";")

        wide_dat.loc[p, "temp_pic"] = np.round(
            main["temp_pic_set"].values[0] - main["temp_flat"].values[0], 2
        )
        wide_dat.loc[p, "temp_plateau"] = np.round(main["temp_flat"].values[0], 2)
        wide_dat.loc[p, "temp_placebo"] = np.round(main["temp_active"].values[0], 2)

        # EVALUATION TASK: Extract pain ratings for TENS-active vs TENS-inactive
        # Participants rate their pain on each trial, allowing us to measure
        # the placebo effect (reduced pain ratings during TENS-active trials)
        eval_task = main[
            [
                "ratingScale.response",
                "condition",
                "loop_eval_discri.thisRepN",
                "trials.thisN",
            ]
        ]
        eval_task = eval_task[eval_task["ratingScale.response"].isna() == False]

        eval_task_active_noplacebo = eval_task[
            (eval_task["condition"] == "active")
            & (eval_task["loop_eval_discri.thisRepN"].isna())
        ].reset_index(drop=True)
        eval_task_inactive_noplacebo = eval_task[
            (eval_task["condition"] == "inactive")
            & (eval_task["loop_eval_discri.thisRepN"].isna())
        ].reset_index(drop=True)

        eval_task_active_placebo = eval_task[
            (eval_task["condition"] == "active")
            & (~eval_task["loop_eval_discri.thisRepN"].isna())
        ].reset_index(drop=True)
        eval_task_inactive_placebo = eval_task[
            (eval_task["condition"] == "inactive")
            & ~(eval_task["loop_eval_discri.thisRepN"].isna())
        ].reset_index(drop=True)

        eval_task_active = eval_task[(eval_task["condition"] == "active")].reset_index(
            drop=True
        )
        eval_task_inactive = eval_task[(eval_task["condition"] == "inactive")].reset_index(
            drop=True
        )

        wide_dat.loc[p, "average_eval_active"] = np.mean(
            eval_task_active_noplacebo["ratingScale.response"].values
        )
        wide_dat.loc[p, "average_eval_inactive"] = np.mean(
            eval_task_inactive_noplacebo["ratingScale.response"].values
        )

        wide_dat.loc[p, "average_placebo_eval_active"] = np.mean(
            eval_task_active_placebo["ratingScale.response"].values
        )
        wide_dat.loc[p, "average_placebo_eval_inactive"] = np.mean(
            eval_task_inactive_placebo["ratingScale.response"].values
        )

        placebo_diff = np.mean(
            eval_task_inactive_placebo["ratingScale.response"].values
            - eval_task_active_placebo["ratingScale.response"].values
        )
        placebo_diff_perc = (
            placebo_diff
            / np.nanmean(eval_task_inactive_placebo["ratingScale.response"].values)
            * 100
        )

        wide_dat.loc[p, "average_placebo_all"] = placebo_diff
        wide_dat.loc[p, "perc_placebo_all"] = placebo_diff_perc

        # Block-level analysis
        eval_task_b1 = eval_task[eval_task["trials.thisN"] == 0]
        eval_task_b2 = eval_task[eval_task["trials.thisN"] == 1]

        eval_task_active_noplacebo_b1 = eval_task_b1[
            (eval_task_b1["condition"] == "active")
            & (eval_task_b1["loop_eval_discri.thisRepN"].isna())
        ].reset_index(drop=True)
        eval_task_active_noplacebo_b2 = eval_task_b2[
            (eval_task_b2["condition"] == "active")
            & (eval_task_b2["loop_eval_discri.thisRepN"].isna())
        ].reset_index(drop=True)

        wide_dat.loc[p, "average_eval_active_b1"] = np.mean(
            eval_task_active_noplacebo_b1["ratingScale.response"].values
        )
        wide_dat.loc[p, "average_eval_active_b2"] = np.mean(
            eval_task_active_noplacebo_b2["ratingScale.response"].values
        )

        eval_task_inactive_noplacebo_b1 = eval_task_b1[
            (eval_task_b1["condition"] == "inactive")
            & (eval_task_b1["loop_eval_discri.thisRepN"].isna())
        ].reset_index(drop=True)
        eval_task_inactive_noplacebo_b2 = eval_task_b2[
            (eval_task_b2["condition"] == "inactive")
            & (eval_task_b2["loop_eval_discri.thisRepN"].isna())
        ].reset_index(drop=True)

        wide_dat.loc[p, "average_eval_inactive"] = np.mean(
            eval_task_inactive_noplacebo["ratingScale.response"].values
        )
        wide_dat.loc[p, "average_eval_inactive"] = np.mean(
            eval_task_inactive_noplacebo["ratingScale.response"].values
        )

        placebo_diff_b1 = np.mean(
            eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 0][
                "ratingScale.response"
            ].values
            - eval_task_active_placebo[eval_task_active_placebo["trials.thisN"] == 0][
                "ratingScale.response"
            ].values
        )
        placebo_diff_b2 = np.mean(
            eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 1][
                "ratingScale.response"
            ].values
            - eval_task_active_placebo[eval_task_active_placebo["trials.thisN"] == 1][
                "ratingScale.response"
            ].values
        )

        placebo_diff_all_b1 = (
            eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 0][
                "ratingScale.response"
            ].values
            - eval_task_active_placebo[eval_task_active_placebo["trials.thisN"] == 0][
                "ratingScale.response"
            ].values
        )
        placebo_diff_all_b2 = (
            eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 1][
                "ratingScale.response"
            ].values
            - eval_task_active_placebo[eval_task_active_placebo["trials.thisN"] == 1][
                "ratingScale.response"
            ].values
        )

        for i in range(len(placebo_diff_all_b1)):
            wide_dat.loc[p, "placebo_b1_" + str(i + 1)] = placebo_diff_all_b1[i]
        for i in range(len(placebo_diff_all_b2)):
            wide_dat.loc[p, "placebo_b2_" + str(i + 1)] = placebo_diff_all_b2[i]

        wide_dat.loc[p, "average_placebo_eval_active_b1"] = np.mean(
            eval_task_active_placebo[eval_task_active_placebo["trials.thisN"] == 0][
                "ratingScale.response"
            ].values
        )
        wide_dat.loc[p, "average_placebo_eval_inactive_b1"] = np.mean(
            eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 0][
                "ratingScale.response"
            ].values
        )

        wide_dat.loc[p, "average_placebo_eval_inactive_b2"] = np.mean(
            eval_task_active_placebo[eval_task_active_placebo["trials.thisN"] == 1][
                "ratingScale.response"
            ].values
        )
        wide_dat.loc[p, "average_placebo_eval_inactive_b2"] = np.mean(
            eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 1][
                "ratingScale.response"
            ].values
        )

        wide_dat.loc[p, "average_placebo_b1"] = placebo_diff_b1
        wide_dat.loc[p, "average_placebo_b2"] = placebo_diff_b2

        placebo_diff_perc_b1 = (
            placebo_diff_b1
            / np.mean(
                eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 0][
                    "ratingScale.response"
                ].values
            )
            * 100
        )
        placebo_diff_perc_b2 = (
            placebo_diff_b2
            / np.mean(
                eval_task_inactive_placebo[eval_task_inactive_placebo["trials.thisN"] == 1][
                    "ratingScale.response"
                ].values
            )
            * 100
        )

        wide_dat.loc[p, "perc_placebo_b1"] = placebo_diff_perc_b1
        wide_dat.loc[p, "perc_placebo_b2"] = placebo_diff_perc_b2

        thresh = main["temp_pic_set"].values[0]

        # Plot evaluation task for this participant
        plt.figure()
        plt.plot(eval_task_active["ratingScale.response"].values, color="g")
        plt.scatter(
            x=np.arange(len(eval_task_active)),
            y=eval_task_active["ratingScale.response"].values,
            color="g",
        )
        plt.plot(eval_task_inactive["ratingScale.response"].values, color="r")
        plt.scatter(
            x=np.arange(len(eval_task_inactive)),
            y=eval_task_inactive["ratingScale.response"].values,
            color="r",
        )
        plt.axvline(x=7.5, color="k", linestyle="--")
        plt.axvline(x=11.5, color="k", linestyle="--")
        plt.axvline(x=19.5, color="k", linestyle="--")
        plt.ylim([0, 100])
        plt.xlabel("Trial")
        plt.ylabel("Pain intensity rating")
        plt.savefig(opj(deriv_path, p + "_eval_task.png"))

        out_frame = pd.DataFrame(
            dict(
                ratings=list(eval_task_active["ratingScale.response"].values)
                + list(eval_task_inactive["ratingScale.response"].values),
                condition=["active"] * len(eval_task_active)
                + ["inactive"] * len(eval_task_inactive),
                trial=list(np.arange(len(eval_task_active))) * 2,
                participant=p[:7],
                block=np.where(
                    np.asarray(list(np.arange(len(eval_task_active))) * 2) < 12, 1, 2
                ),
            )
        )
        all_eval_frames.append(out_frame)

        # Plot temperature in each trial
        temp_files = [
            f
            for f in os.listdir(par_fold)
            if "maintask" in f and "temp_trial_eval" in f and ".csv" in f
        ]
        plt.figure(figsize=(5, 5))
        for trial in temp_files:
            temp = pd.read_csv(opj(par_fold, trial))
            avg_temp = np.average(temp[["z1", "z2", "z3", "z4", "z5"]], axis=1)
            plt.plot(np.arange(len(temp)), avg_temp, color="grey", alpha=0.5)
            plt.xlabel("Sample")
            plt.ylim([38, plateau + 2])
            plt.ylabel("Temperature")
        plt.axhline(
            y=main["temp_flat"].values[0], color="k", linestyle="--", label="plateau"
        )
        plt.axhline(
            y=main["temp_active"].values[0], color="r", linestyle="--", label="active"
        )
        plt.legend()
        plt.savefig(opj(deriv_path, p + "_task_temp_eval.png"))

        # Main task - discrimination
        discrim_task = main[
            [
                "participant",
                "condition",
                "pic_presence",
                "pic_response",
                "trials.thisN",
                "loop_eval_discri.thisN",
                "loop_discri.thisRepN",
            ]
        ]
        discrim_task = discrim_task[discrim_task["loop_discri.thisRepN"].isna() == False]

        discrim_task["actual_trial"] = (
            [9] * 4
            + [10] * 4
            + [11] * 4
            + [12] * 4
            + [21] * 4
            + [22] * 4
            + [23] * 4
            + [24] * 4
        )

        discrim_task["pic_response"] = discrim_task["pic_response"].fillna("None")

        discrim_task_missed = len(discrim_task[discrim_task["pic_response"] == "None"])
        wide_dat.loc[p, "discrim_missed_responses"] = discrim_task_missed

        discrim_task = discrim_task[discrim_task["pic_response"] != "None"]
        discrim_task["pic_response"] = discrim_task["pic_response"].astype(int)
        discrim_task.reset_index(drop=True, inplace=True)

        accurate = []
        detection_type = []
        for presence, response in zip(
            discrim_task["pic_presence"].values, discrim_task["pic_response"].values
        ):
            if presence == "pic-present" and response == 1:
                accurate.append(1)
                detection_type.append("hit")
            elif presence == "pic-absent" and response == 0:
                accurate.append(1)
                detection_type.append("correct rejection")
            elif presence == "pic-present" and response == 0:
                accurate.append(0)
                detection_type.append("miss")
            elif presence == "pic-absent" and response == 1:
                accurate.append(0)
                detection_type.append("false alarm")

        discrim_task["accuracy"] = accurate
        discrim_task["detection_type"] = detection_type

        # Signal detection theory measures
        hits = len(discrim_task[discrim_task["detection_type"] == "hit"])
        misses = len(discrim_task[discrim_task["detection_type"] == "miss"])
        fas = len(discrim_task[discrim_task["detection_type"] == "false alarm"])
        crs = len(discrim_task[discrim_task["detection_type"] == "correct rejection"])

        out = SDT(hits, misses, fas, crs)
        wide_dat.loc[p, "d_prime_all"] = out["d"]
        wide_dat.loc[p, "beta_all"] = out["beta"]
        wide_dat.loc[p, "c_all"] = out["c"]

        # Signal detection in active and inactive conditions
        hits_active = np.sum(
            discrim_task[discrim_task["condition"] == "active"]["detection_type"] == "hit"
        )
        misses_active = np.sum(
            discrim_task[discrim_task["condition"] == "active"]["detection_type"] == "miss"
        )
        fas_active = np.sum(
            discrim_task[discrim_task["condition"] == "active"]["detection_type"]
            == "false alarm"
        )
        crs_active = np.sum(
            discrim_task[discrim_task["condition"] == "active"]["detection_type"]
            == "correct rejection"
        )
        hits_inactive = np.sum(
            discrim_task[discrim_task["condition"] == "inactive"]["detection_type"] == "hit"
        )
        misses_inactive = np.sum(
            discrim_task[discrim_task["condition"] == "inactive"]["detection_type"] == "miss"
        )
        fas_inactive = np.sum(
            discrim_task[discrim_task["condition"] == "inactive"]["detection_type"]
            == "false alarm"
        )
        crs_inactive = np.sum(
            discrim_task[discrim_task["condition"] == "inactive"]["detection_type"]
            == "correct rejection"
        )

        out_active = SDT(hits_active, misses_active, fas_active, crs_active)
        out_inactive = SDT(hits_inactive, misses_inactive, fas_inactive, crs_inactive)

        wide_dat.loc[p, "d_prime_active"] = out_active["d"]
        wide_dat.loc[p, "d_prime_inactive"] = out_inactive["d"]
        wide_dat.loc[p, "beta_active"] = out_active["beta"]
        wide_dat.loc[p, "beta_inactive"] = out_inactive["beta"]
        wide_dat.loc[p, "c_active"] = out_active["c"]
        wide_dat.loc[p, "c_inactive"] = out_inactive["c"]

        wide_dat.loc[p, "acc_all"] = np.mean(discrim_task["accuracy"].values)

        wide_dat.loc[p, "active_acc_all"] = np.mean(
            discrim_task[discrim_task["condition"] == "active"]["accuracy"].values
        )
        wide_dat.loc[p, "inactive_acc_all"] = np.mean(
            discrim_task[discrim_task["condition"] == "inactive"]["accuracy"].values
        )

        discrim_task_b1 = discrim_task[discrim_task["trials.thisN"] == 0]
        wide_dat.loc[p, "active_acc_b1"] = np.mean(
            discrim_task_b1[discrim_task_b1["condition"] == "active"]["accuracy"].values
        )
        wide_dat.loc[p, "inactive_acc_b1"] = np.mean(
            discrim_task_b1[discrim_task_b1["condition"] == "inactive"]["accuracy"].values
        )

        discrim_task_b2 = discrim_task[discrim_task["trials.thisN"] == 1]
        wide_dat.loc[p, "active_acc_b2"] = np.mean(
            discrim_task_b2[discrim_task_b2["condition"] == "active"]["accuracy"].values
        )
        wide_dat.loc[p, "inactive_acc_b2"] = np.mean(
            discrim_task_b2[discrim_task_b2["condition"] == "inactive"]["accuracy"].values
        )

        # Calculate average response time
        reaction_time = []
        reaction_time_good_a = []
        reaction_time_bad_a = []
        for _, row in main.iterrows():
            if pd.notnull(row["discrimin_resp.rt"]):
                rt = float(row["discrimin_resp.rt"])
                reaction_time.append(rt)
                if row["pic_response"] == 1 and row["pic_presence"] == "pic-present":
                    reaction_time_good_a.append(rt)
                elif row["pic_response"] == 0 and row["pic_presence"] == "pic-absent":
                    reaction_time_good_a.append(rt)
                else:
                    reaction_time_bad_a.append(rt)

        discrim_task.loc[p, "reaction_time"] = (
            np.mean(reaction_time) if reaction_time else np.nan
        )
        discrim_task.loc[p, "reaction_time_good_a"] = (
            np.mean(reaction_time_good_a) if reaction_time_good_a else np.nan
        )
        discrim_task.loc[p, "reaction_time_bad_a"] = (
            np.mean(reaction_time_bad_a) if reaction_time_bad_a else np.nan
        )

        discrim_task_avg = (
            discrim_task.groupby(["condition"]).mean(numeric_only=True).reset_index()
        )
        discrim_task_avg["participant"] = p

        all_discrim_task.append(discrim_task_avg)
        all_discrim_task_long.append(discrim_task)
        plt.figure()
        sns.catplot(x="condition", y="accuracy", data=discrim_task_avg, kind="point")
        plt.ylim([0, 1.2])
        plt.savefig(opj(deriv_path, p + "_discrim_task.png"))

        # Plot temperature pic
        temp_files = [
            f
            for f in os.listdir(par_fold)
            if "maintask" in f and "temp_trial_pic" in f and ".csv" in f
        ]
        plt.figure(figsize=(5, 5))
        for trial in temp_files:
            temp = pd.read_csv(opj(par_fold, trial))
            avg_temp = np.average(temp[["z1", "z2", "z3", "z4", "z5"]], axis=1)
            plt.plot(np.arange(len(temp)), avg_temp, color="grey", alpha=0.5)
            plt.xlabel("Sample")
            plt.ylabel("Temperature")
        plt.axhline(
            y=main["temp_flat"].values[0], color="k", linestyle="--", label="active"
        )
        plt.axhline(y=thresh, color="r", linestyle="--", label="plateau")
        plt.legend()
        plt.savefig(opj(deriv_path, p + "_task_temp_pic.png"))
        plt.close("all")

    # =========================================================================
    # DATA AGGREGATION
    # Combine individual participant data into group-level dataframes
    # =========================================================================
    all_eval_frame = pd.concat(all_eval_frames)
    all_discrim_task = pd.concat(all_discrim_task)
    all_discrim_task_long = pd.concat(all_discrim_task_long)
    wide_dat["participant"] = list(wide_dat.index)

    # -------------------------------------------------------------------------
    # QUESTIONNAIRE PROCESSING
    # Compute STAI-Y1, STAI-Y2, and PCS scores (in memory only)
    # Note: Scores are computed fresh each run; sourcedata is never modified
    # -------------------------------------------------------------------------
    questionnaire_scores = compute_questionnaire_scores(bidsroot)

    # Load sociodemographic data and merge with questionnaire scores
    socio = pd.read_csv(opj(bidsroot, "sociodemo.csv"))
    socio["pcs_total"] = questionnaire_scores["pcs_total"].reset_index(drop=True)
    socio["iastay1_total"] = questionnaire_scores["iastay1_total"].reset_index(drop=True)
    socio["iastay2_total"] = questionnaire_scores["iastay2_total"].reset_index(drop=True)
    # Note: We do NOT save back to sourcedata - scores are computed fresh each run

    # Add sociodemo to wide_dat
    socio.index = socio[socio.columns[1]]

    wide_dat["age"] = np.nan
    for row in socio.iterrows():
        if row[0] in wide_dat.index:
            wide_dat.loc[row[0], "age"] = int(row[1]["2. Quel est votre âge en années? "])
            wide_dat.loc[row[0], "ismale"] = (
                row[1]["4. Quel est votre genre? "] == "Masculin"
            )
            wide_dat.loc[row[0], "isfemale"] = (
                row[1]["4. Quel est votre genre? "] == "Féminin"
            )
            wide_dat.loc[row[0], "Autres"] = (
                row[1]["4. Quel est votre genre? "] == "Autres"
            )
            wide_dat.loc[row[0], "pcs_total"] = row[1]["pcs_total"]
            wide_dat.loc[row[0], "iastay1_total"] = row[1]["iastay1_total"]
            wide_dat.loc[row[0], "iastay2_total"] = row[1]["iastay2_total"]

    wide_dat["ismale"] = wide_dat["ismale"].astype(int)
    wide_dat["isfemale"] = wide_dat["isfemale"].astype(int)
    wide_dat["Autres"] = wide_dat["Autres"].astype(int)

    # =========================================================================
    # EXCLUSION CRITERIA
    # Apply exclusion criteria in order of priority:
    # 1. Perfect discrimination (ceiling effect - cannot detect differences)
    # 2. Low placebo effect (below threshold - no placebo response)
    # 3. Custom exclusions (data quality issues)
    # 4. Below-chance accuracy (<=50% - task not performed correctly)
    # =========================================================================
    wide_dat["exclude"] = 0
    all_eval_frame["exclude"] = 0
    all_discrim_task["exclude"] = 0
    all_discrim_task_long["exclude"] = 0

    # Initialize exclusion lists for manuscript reporting
    excluded_perfect = []
    excluded_placebo = []
    excluded_custom = []
    excluded_low_acc = []

    if exclude_perfect:
        wide_dat_perf = list(wide_dat[wide_dat["acc_all"] == 1]["participant"])
        excluded_perfect = wide_dat_perf
        wide_dat.loc[wide_dat["participant"].isin(wide_dat_perf), "exclude"] = 1
        all_eval_frame.loc[
            all_eval_frame["participant"].isin(wide_dat_perf), "exclude"
        ] = 1
        all_discrim_task.loc[
            all_discrim_task["participant"].isin(wide_dat_perf), "exclude"
        ] = 1
        all_discrim_task_long.loc[
            all_discrim_task_long["participant"].isin(wide_dat_perf), "exclude"
        ] = 1
        print(
            f"{len(wide_dat_perf)} participants with perfect discrimination excluded, "
            f"leaving {len(wide_dat) - len(wide_dat_perf)} participants"
        )

    if exclude_placebo is not None:
        wide_dat_placebo = list(
            wide_dat[wide_dat["perc_placebo_all"] < exclude_placebo]["participant"]
        )
        excluded_placebo = wide_dat_placebo
        wide_dat.loc[wide_dat["participant"].isin(wide_dat_placebo), "exclude"] = 1
        all_eval_frame.loc[
            all_eval_frame["participant"].isin(wide_dat_placebo), "exclude"
        ] = 1
        all_discrim_task.loc[
            all_discrim_task["participant"].isin(wide_dat_placebo), "exclude"
        ] = 1
        all_discrim_task_long.loc[
            all_discrim_task_long["participant"].isin(wide_dat_placebo), "exclude"
        ] = 1
        print(
            f"{len(wide_dat_placebo)} participants with low placebo effect excluded, "
            f"leaving {len(wide_dat) - np.sum(wide_dat['exclude'])} participants"
        )

    if exclude_custom:
        excluded_custom = exclude_custom
        wide_dat.loc[wide_dat["participant"].isin(exclude_custom), "exclude"] = 1
        all_eval_frame.loc[
            all_eval_frame["participant"].isin(exclude_custom), "exclude"
        ] = 1
        all_discrim_task.loc[
            all_discrim_task["participant"].isin(exclude_custom), "exclude"
        ] = 1
        all_discrim_task_long.loc[
            all_discrim_task_long["participant"].isin(exclude_custom), "exclude"
        ] = 1
        print(
            f"{len(exclude_custom)} participants excluded for other reasons, "
            f"leaving {len(wide_dat) - np.sum(wide_dat['exclude'])} participants"
        )

    if exclude_low_acc:
        wide_dat_lowacc = list(
            wide_dat[wide_dat["acc_all"] <= low_acc_threshold]["participant"]
        )
        excluded_low_acc = wide_dat_lowacc
        wide_dat.loc[wide_dat["participant"].isin(wide_dat_lowacc), "exclude"] = 1
        all_eval_frame.loc[
            all_eval_frame["participant"].isin(wide_dat_lowacc), "exclude"
        ] = 1
        all_discrim_task.loc[
            all_discrim_task["participant"].isin(wide_dat_lowacc), "exclude"
        ] = 1
        all_discrim_task_long.loc[
            all_discrim_task_long["participant"].isin(wide_dat_lowacc), "exclude"
        ] = 1
        print(
            f"{len(wide_dat_lowacc)} participants with low discrimination "
            f"(<= {low_acc_threshold}) excluded, "
            f"leaving {len(wide_dat) - np.sum(wide_dat['exclude'])} participants"
        )

    # Save full dataframes
    wide_dat.to_csv(opj(output_dir, "data_wide_dat_full.csv"), index=False)
    all_eval_frame.to_csv(opj(output_dir, "data_all_eval_frame_full.csv"), index=False)
    all_discrim_task.to_csv(opj(output_dir, "data_all_discrim_task_full.csv"), index=False)
    all_discrim_task_long.to_csv(
        opj(output_dir, "data_all_discrim_task_long_full.csv"), index=False
    )

    # Remove excluded participants
    wide_dat = wide_dat[wide_dat["exclude"] == 0]
    all_eval_frame = all_eval_frame[all_eval_frame["exclude"] == 0]
    all_discrim_task = all_discrim_task[all_discrim_task["exclude"] == 0]
    all_discrim_task_long = all_discrim_task_long[all_discrim_task_long["exclude"] == 0]

    # Save data with exclusions
    wide_dat.to_csv(opj(output_dir, "data_wide_dat_withexcl.csv"), index=False)
    all_eval_frame.to_csv(opj(output_dir, "data_all_eval_frame_withexcl.csv"), index=False)
    all_discrim_task.to_csv(
        opj(output_dir, "data_all_discrim_task_withexcl.csv"), index=False
    )
    all_discrim_task_long.to_csv(
        opj(output_dir, "data_all_discrim_task_long_withexcl.csv"), index=False
    )

    # =========================================================================
    # VISUALIZATION: Generate publication-ready figures
    # =========================================================================

    # -------------------------------------------------------------------------
    # Figure: Evaluation task - Pain ratings by trial and condition
    # Shows pain ratings across all 24 trials, with TENS-active vs inactive
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.pointplot(
        x="trial",
        y="ratings",
        hue="condition",
        data=all_eval_frame,
        errorbar="se",
        alpha=0.9,
        ax=ax,
    )
    plt.ylim([0, 80])
    ax.fill_between(
        np.arange(7.5, 11.5, 0.1), y1=80, facecolor="#90ee90", alpha=0.2, zorder=0
    )
    ax.fill_between(
        np.arange(19.5, 24, 0.1),
        y1=80,
        facecolor="#90ee90",
        alpha=0.2,
        label="Conditoning",
        zorder=0,
    )
    ax.fill_between(
        np.arange(-1, 7.5, 0.1),
        y1=80,
        facecolor="#8a8a8a",
        alpha=0.2,
        label="Placebo",
        zorder=0,
    )
    ax.fill_between(
        np.arange(11.5, 19.5, 0.1),
        y1=80,
        facecolor="#8a8a8a",
        alpha=0.2,
        label="Placebo",
        zorder=0,
    )

    l, h = ax.get_legend_handles_labels()
    l.append(
        Patch(
            facecolor="#90ee90",
            edgecolor="#90ee90",
            label="Conditioning",
            alpha=1,
            zorder=999,
        )
    )
    l.append(
        Patch(
            facecolor="#8a8a8a", edgecolor="#8a8a8a", label="Placebo", alpha=1, zorder=999
        )
    )
    ax.legend(
        l,
        ["TENS on", "TENS off", "Placebo", "Conditioning"],
        fontsize=12,
        title="",
        loc="upper right",
        ncol=4,
        columnspacing=0.5,
    )

    plt.ylabel("Pain rating", fontsize=18)
    plt.xlabel("Trials", fontsize=18)
    plt.xticks(labels=np.arange(1, 25), ticks=np.arange(24))
    plt.tick_params(labelsize=14)
    plt.setp(ax.collections[0:2], alpha=0.4)
    plt.setp(ax.lines, alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        opj(output_dir, "figures", "group_eval_task.png"), dpi=800, bbox_inches="tight"
    )

    # Same but with discrimination interspersed
    all_discrim_task_long_avg = (
        all_discrim_task_long.groupby(["participant", "condition", "actual_trial"])
        .mean(numeric_only=True)
        .reset_index()
    )

    all_discrim_task_long_avg["actual_trial"] = (
        all_discrim_task_long_avg["actual_trial"].astype(int) - 0.8
    )
    all_eval_frame["trial"].astype(int)

    all_eval_frame_avg = (
        all_eval_frame.groupby(["participant", "condition", "trial"])
        .mean(numeric_only=True)
        .reset_index()
    )

    color_placebo_back = "#9fc8c8"
    color_conditioning_back = "#BBBBBB"
    TENS_on = "#a00000"
    TENS_off = "#1a80bb"
    pal = [TENS_off, TENS_on]
    all_discrim_task_long_avg_p1 = all_discrim_task_long_avg[
        all_discrim_task_long_avg["actual_trial"] < 12
    ]
    all_discrim_task_long_avg_p2 = all_discrim_task_long_avg[
        all_discrim_task_long_avg["actual_trial"] > 12
    ]
    fig, (host2, host) = plt.subplots(
        2, 1, figsize=(7, 6), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )
    plt.sca(host)
    sns.pointplot(
        x="actual_trial",
        y="accuracy",
        hue="condition",
        errorbar="se",
        data=all_discrim_task_long_avg_p1,
        alpha=0.9,
        native_scale=True,
        palette=pal,
    )
    sns.pointplot(
        x="actual_trial",
        y="accuracy",
        hue="condition",
        errorbar="se",
        data=all_discrim_task_long_avg_p2,
        alpha=0.9,
        native_scale=True,
        palette=pal,
    )
    plt.sca(host2)
    sns.pointplot(
        x="trial",
        y="ratings",
        hue="condition",
        data=all_eval_frame_avg,
        errorbar="se",
        alpha=0.9,
        palette=pal,
    )

    host.set_ylim([0.5, 1])
    host2.set_ylim([0, 70])
    plt.ylabel("Pain rating", fontsize=18)
    host.set_xlim([-1, 23.6])
    host.set_xlabel("Trials", fontsize=18)
    plt.tick_params(labelsize=14)
    host.fill_between(
        np.arange(7.5, 11.5, 0.1), y1=1, facecolor=color_placebo_back, alpha=0.4, zorder=0
    )
    host.fill_between(
        np.arange(19.5, 24, 0.1),
        y1=1,
        facecolor=color_placebo_back,
        alpha=0.4,
        label="Conditoning",
        zorder=0,
    )
    host.fill_between(
        np.arange(-1, 7.5, 0.1),
        y1=1,
        facecolor=color_conditioning_back,
        alpha=0.2,
        label="Placebo",
        zorder=0,
    )
    host.fill_between(
        np.arange(11.5, 19.5, 0.1),
        y1=1,
        facecolor=color_conditioning_back,
        alpha=0.2,
        label="Placebo",
        zorder=0,
    )

    host2.fill_between(
        np.arange(7.5, 11.5, 0.1), y1=80, facecolor=color_placebo_back, alpha=0.4, zorder=0
    )
    host2.fill_between(
        np.arange(19.5, 24, 0.1),
        y1=80,
        facecolor=color_placebo_back,
        alpha=0.4,
        label="Conditoning",
        zorder=0,
    )
    host2.fill_between(
        np.arange(-1, 7.5, 0.1),
        y1=80,
        facecolor=color_conditioning_back,
        alpha=0.2,
        label="Placebo",
        zorder=0,
    )
    host2.fill_between(
        np.arange(11.5, 19.5, 0.1),
        y1=80,
        facecolor=color_conditioning_back,
        alpha=0.2,
        label="Placebo",
        zorder=0,
    )
    plt.xticks(labels=np.arange(1, 25), ticks=np.arange(24))
    host2.axvline(7.5, color="gray", linestyle="--")
    host.axvline(7.5, color="gray", linestyle="--")
    host2.axvline(11.5, color="gray", linestyle="--")
    host.axvline(11.5, color="gray", linestyle="--")

    host2.axvline(19.5, color="gray", linestyle="--")
    host.axvline(19.5, color="gray", linestyle="--")
    host2.axvline(23.5, color="gray", linestyle="--")
    host.axvline(23.5, color="gray", linestyle="--")

    host.set_ylabel("Proportion correct", fontsize=18)
    l, h = host.get_legend_handles_labels()
    l = l[0:2]

    l.append(
        Patch(
            facecolor=color_placebo_back,
            edgecolor=color_placebo_back,
            label="Placebo",
            alpha=0.4,
            zorder=999,
        )
    )
    l.append(
        Patch(
            facecolor=color_conditioning_back,
            edgecolor=color_conditioning_back,
            label="Conditioning",
            alpha=0.2,
            zorder=999,
        )
    )
    host2.legend(
        l,
        ["TENS on", "TENS off", "Placebo", "Conditioning"],
        fontsize=12,
        title="",
        loc="lower left",
        ncol=4,
        columnspacing=0.5,
    )
    host.tick_params(labelsize=14)
    host2.tick_params(labelsize=14)
    host.legend().remove()

    plt.setp(ax.collections, alpha=0.6)
    plt.setp(ax.lines, alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        opj(output_dir, "figures", "group_discrim_task.svg"),
        transparent=True,
        dpi=800,
        bbox_inches="tight",
    )

    plt.ylim([0, 80])
    ax.fill_between(
        np.arange(7.5, 11.5, 0.1), y1=80, facecolor="#d3d3d3", alpha=0.2, zorder=0
    )
    ax.fill_between(
        np.arange(19.5, 23.5, 0.1),
        y1=80,
        facecolor="#d3d3d3",
        alpha=0.2,
        label="Placebo",
        zorder=0,
    )
    l, h = ax.get_legend_handles_labels()
    l.append(
        Patch(
            facecolor="#d3d3d3", edgecolor="#d3d3d3", label="Placebo", alpha=1, zorder=999
        )
    )
    ax.legend(
        l, ["TENS on", "TENS off", "Placebo"], fontsize=12, title="", loc="upper right"
    )

    plt.ylabel("Pain rating", fontsize=18)
    plt.xlabel("Trials", fontsize=18)
    plt.xticks(labels=np.arange(1, 25), ticks=np.arange(24))

    plt.tight_layout()
    plt.savefig(opj(output_dir, "figures", "group_eval_task.png"), dpi=800)

    fig, ax = plt.subplots(figsize=(4, 4))
    all_eval_frame_placeb = all_eval_frame[
        all_eval_frame.trial.isin([8, 9, 10, 11, 20, 21, 22, 23])
    ]
    all_eval_frame_placeb = (
        all_eval_frame_placeb.groupby(
            [
                "participant",
                "condition",
            ]
        )
        .mean(numeric_only=True)
        .reset_index()
    )
    all_eval_frame_placeb.index = all_eval_frame_placeb["participant"]
    sns.boxplot(
        x="condition",
        y="ratings",
        hue="condition",
        data=all_eval_frame_placeb,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "15",
            "zorder": 999,
        },
        showcaps=False,
        palette=pal,
    )
    sns.stripplot(
        x="condition",
        y="ratings",
        hue="condition",
        data=all_eval_frame_placeb,
        alpha=0.5,
        size=12,
        jitter=False,
        edgecolor="black",
        linewidth=1,
        palette=pal,
    )
    for p in list(all_eval_frame_placeb.index):
        plt.plot(
            [0, 1],
            [
                all_eval_frame_placeb[all_eval_frame_placeb["condition"] == "active"].loc[
                    p, "ratings"
                ],
                all_eval_frame_placeb[
                    all_eval_frame_placeb["condition"] == "inactive"
                ].loc[p, "ratings"],
            ],
            color="gray",
            alpha=0.5,
        )
    plt.xticks([0, 1], ["TENS on", "TENS off"], fontsize=12)
    plt.ylabel("Pain rating", fontsize=18)
    plt.xlabel("", fontsize=18)
    plt.tick_params(labelsize=14)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.18, top=0.98)
    plt.savefig(
        opj(output_dir, "figures", "mean_placebo_effect.png"),
        transparent=True,
        bbox_inches="tight",
    )

    # Plot placebo effect distribution
    plt.figure()
    ax = sns.stripplot(y=wide_dat["perc_placebo_all"], jitter=True)
    ax.axhline(10, color="k", linestyle="--")
    plt.ylabel("Placebo effect (%)")
    plt.xlabel("Participants")
    plt.savefig(
        opj(output_dir, "figures", "placebo_effect_distribution.png"),
        dpi=800,
        bbox_inches="tight",
    )

    ##########################################################
    # Statistics
    ##########################################################

    wide_dat["active_acc_all"] = wide_dat["active_acc_all"].astype(float)
    wide_dat["inactive_acc_all"] = wide_dat["inactive_acc_all"].astype(float)

    # Descriptives
    desc_stats = wide_dat[["age", "temp_pic", "temp_plateau", "temp_placebo"]].describe()
    desc_stats["n_male"] = wide_dat["ismale"].sum()
    desc_stats["n_female"] = wide_dat["isfemale"].sum()
    desc_stats.to_csv(opj(output_dir, "stats", "descriptives.csv"))

    # Evaluation
    # T-test for placebo effect eval across all participants
    out = pg.ttest(
        all_eval_frame_placeb[all_eval_frame_placeb["condition"] == "active"]["ratings"],
        all_eval_frame_placeb[all_eval_frame_placeb["condition"] == "inactive"]["ratings"],
        paired=True,
    )
    out.to_csv(opj(output_dir, "stats", "t_test_placebo_eval.csv"))

    # =========================================================================
    # STATISTICAL ANALYSES
    # =========================================================================

    # -------------------------------------------------------------------------
    # RM ANOVA: Condition (TENS active/inactive) x Block on pain ratings
    # Tests whether placebo effect differs between conditions and across blocks
    # -------------------------------------------------------------------------
    all_eval_frame_placeb = all_eval_frame[
        all_eval_frame.trial.isin([8, 9, 10, 11, 20, 21, 22, 23])
    ]
    all_eval_frame_placeb = (
        all_eval_frame_placeb.groupby(["participant", "condition", "block"])
        .mean(numeric_only=True)
        .reset_index()
    )

    out = pg.rm_anova(
        data=all_eval_frame_placeb,
        dv="ratings",
        within=["condition", "block"],
        subject="participant",
        correction=False,
    )
    out.to_csv(opj(output_dir, "stats", "rm_anova_cond-block_eval.csv"))

    # -------------------------------------------------------------------------
    # DISCRIMINATION TASK STATISTICS
    # Paired t-tests comparing TENS-active vs TENS-inactive conditions
    # -------------------------------------------------------------------------
    # T-test for placebo effect across all participants on accuracy
    t_test_paired = pg.ttest(
        wide_dat["inactive_acc_all"].astype(float),
        wide_dat["active_acc_all"].astype(float),
        paired=True,
    )
    t_test_paired.to_csv(opj(output_dir, "stats", "t_test_accuracy.csv"))

    # T-test for placebo effect across all participants on d prime
    t_test_paired_d = pg.ttest(
        wide_dat["d_prime_inactive"].astype(float),
        wide_dat["d_prime_active"].astype(float),
        paired=True,
    )
    t_test_paired_d.to_csv(opj(output_dir, "stats", "t_test_dprime.csv"))

    # T-test for placebo effect across all participants on beta
    t_test_paired_beta = pg.ttest(
        wide_dat["beta_inactive"].astype(float),
        wide_dat["beta_active"].astype(float),
        paired=True,
    )
    t_test_paired_beta.to_csv(opj(output_dir, "stats", "t_test_beta.csv"))

    # T-test for placebo effect across all participants on c
    t_test_paired_c = pg.ttest(
        wide_dat["c_inactive"].astype(float),
        wide_dat["c_active"].astype(float),
        paired=True,
    )
    t_test_paired_c.to_csv(opj(output_dir, "stats", "t_test_c.csv"))

    # -------------------------------------------------------------------------
    # TOST EQUIVALENCE TEST
    # Two One-Sided Tests to establish equivalence between conditions
    # Tests whether the difference is practically equivalent to zero
    # -------------------------------------------------------------------------
    cohen_dz_bounds = 0.45  # Equivalence bounds in Cohen's dz units
    sd_diff_acc = np.std(
        wide_dat["active_acc_all"] - wide_dat["inactive_acc_all"], ddof=1
    )
    raw_diff = cohen_dz_bounds * sd_diff_acc
    diff = wide_dat["inactive_acc_all"] - wide_dat["active_acc_all"]
    mean_diff = np.mean(wide_dat["inactive_acc_all"] - wide_dat["active_acc_all"])

    p, tlowbound, thighbound = ttost_paired(
        wide_dat["inactive_acc_all"].values.astype(float),
        wide_dat["active_acc_all"].values.astype(float),
        low=-raw_diff.round(4),
        upp=raw_diff.round(4),
    )

    tost_df = pd.DataFrame(
        {
            "p_val": [p],
            "t_low_bound": [tlowbound[0]],
            "p_low_bound": [tlowbound[1]],
            "df_low_bound": [tlowbound[2]],
            "t_high_bound": [thighbound[0]],
            "p_high_bound": [thighbound[1]],
            "df_high_bound": [thighbound[2]],
            "mean_diff": [mean_diff],
            "cohen_dz_bounds": [cohen_dz_bounds],
            "sd_diff": [sd_diff_acc],
            "raw_diff": [raw_diff],
            "n_participants": [len(wide_dat)],
        }
    )
    tost_df.to_csv(opj(output_dir, "stats", "tost_accuracy.csv"))

    # -------------------------------------------------------------------------
    # CORRELATION ANALYSES
    # Test relationships between placebo effect, discrimination accuracy,
    # and psychological measures (STAI-Y1, STAI-Y2, PCS)
    # -------------------------------------------------------------------------
    placebo = pd.read_csv(opj(output_dir, "data_wide_dat_full.csv"))
    placebo_effect_exclude = []
    accuracy_all = []
    for _, row in placebo.iterrows():
        if row["exclude"] == 0:
            placebo_effect_exclude.append(row["perc_placebo_all"])
            accuracy_all.append(row["acc_all"])

    accuracy = pd.read_csv(opj(output_dir, "data_all_discrim_task_full.csv"))
    accuracy = accuracy[accuracy["exclude"] != 1]
    accuracy_active = []
    accuracy_inactive = []
    for _, row in accuracy.iterrows():
        if row["condition"] == "active":
            accuracy_active.append(row["accuracy"])
        elif row["condition"] == "inactive":
            accuracy_inactive.append(row["accuracy"])

    correlation_df = pd.DataFrame(
        {
            "placebo_effect": placebo_effect_exclude,
            "accuracy_active": accuracy_active,
            "accuracy_inactive": accuracy_inactive,
            "accuracy_all": accuracy_all,
        }
    )

    correlation_1 = pg.corr(
        x=correlation_df["placebo_effect"],
        y=correlation_df["accuracy_all"],
    )
    correlation_1.to_csv(opj(output_dir, "stats", "corr_placebo_accuracy.csv"))

    plt.figure(figsize=(6, 6))
    plt.scatter(
        correlation_df["placebo_effect"],
        correlation_df["accuracy_all"],
        s=90,
        color="mediumblue",
    )
    plt.tick_params(labelsize=14)
    plt.title(
        "Correlation between placebo effect and discrimination accuracy",
        fontdict={"fontsize": 18},
    )

    sns.regplot(
        x=correlation_df["placebo_effect"],
        y=correlation_df["accuracy_all"],
        scatter=False,
        color="black",
        line_kws={"color": "black", "alpha": 1, "lw": 2},
        ci=None,
        marker="o",
    )
    plt.xlabel("Placebo effect (%)", fontsize=18)
    plt.ylabel("Discrimination accuracy", fontsize=18)
    plt.savefig(opj(output_dir, "figures", "correlation_placebo_accuracy.png"))

    # Get scores for questionnaires, placebo effect and discrim performance
    score_pcs = []
    score_iasta1 = []
    score_iasta2 = []
    for _, row in placebo.iterrows():
        if row["exclude"] == 0:
            score_pcs.append(row["pcs_total"])
            score_iasta1.append(row["iastay1_total"])
            score_iasta2.append(row["iastay2_total"])

    correlation_2 = pd.DataFrame(
        {
            "placebo_effect": placebo_effect_exclude,
            "accuracy_active": accuracy_active,
            "accuracy_inactive": accuracy_inactive,
            "pcs_total": score_pcs,
            "iastay1_total": score_iasta1,
            "iastay2_total": score_iasta2,
            "accuracy_all": accuracy_all,
        }
    )

    # Correlation between placebo effect and pcs
    correlation_2_pcs = pg.corr(
        x=correlation_2["placebo_effect"],
        y=correlation_2["pcs_total"],
    )
    correlation_2_pcs.to_csv(opj(output_dir, "stats", "corr_2_pcs.csv"))

    plt.figure(figsize=(6, 6))
    plt.scatter(
        correlation_2["placebo_effect"],
        correlation_2["pcs_total"],
        s=90,
        color="mediumblue",
    )
    plt.xlabel("Placebo effect (%)", fontsize=18)
    plt.ylabel("PCS score", fontsize=18)
    plt.tick_params(labelsize=14)
    plt.title(
        "Correlation between placebo effect and PCS score",
        fontdict={"fontsize": 18},
    )
    sns.regplot(
        x=correlation_2["placebo_effect"],
        y=correlation_2["pcs_total"],
        scatter=False,
        color="black",
        line_kws={"color": "black", "alpha": 1, "lw": 2},
        ci=None,
        marker="o",
    )
    plt.xlabel("Placebo effect (%)", fontsize=18)
    plt.ylabel("PCS score", fontsize=18)
    plt.savefig(opj(output_dir, "figures", "correlation_placebo_pcs.png"))

    # Correlation between placebo effect and iasta1
    correlation_2_iasta1 = pg.corr(
        x=correlation_2["placebo_effect"],
        y=correlation_2["iastay1_total"],
    )
    correlation_2_iasta1.to_csv(opj(output_dir, "stats", "corr_2_iasta1.csv"))

    plt.figure(figsize=(6, 6))
    plt.scatter(
        correlation_2["placebo_effect"],
        correlation_2["iastay1_total"],
        s=90,
        color="mediumblue",
    )
    plt.xlabel("Placebo effect (%)", fontsize=18)
    plt.ylabel("IASTA1 score", fontsize=18)
    plt.tick_params(labelsize=14)
    plt.title(
        "Correlation between placebo effect and IASTA1 score",
        fontdict={"fontsize": 18},
    )
    sns.regplot(
        x=correlation_2["placebo_effect"],
        y=correlation_2["iastay1_total"],
        scatter=False,
        color="black",
        line_kws={"color": "black", "alpha": 1, "lw": 2},
        ci=None,
        marker="o",
    )
    plt.xlabel("Placebo effect (%)", fontsize=18)
    plt.ylabel("IASTA1 score", fontsize=18)
    plt.savefig(opj(output_dir, "figures", "correlation_placebo_iasta1.png"))

    # Correlation between placebo effect and iasta2
    correlation_2_iasta2 = pg.corr(
        x=correlation_2["placebo_effect"],
        y=correlation_2["iastay2_total"],
    )
    correlation_2_iasta2.to_csv(opj(output_dir, "stats", "corr_2_iasta2.csv"))

    plt.figure(figsize=(6, 6))
    plt.scatter(
        correlation_2["placebo_effect"],
        correlation_2["iastay2_total"],
        s=90,
        color="mediumblue",
    )
    plt.xlabel("Placebo effect (%)", fontsize=18)
    plt.ylabel("IASTA2 score", fontsize=18)
    plt.tick_params(labelsize=14)
    plt.title(
        "Correlation between placebo effect and IASTA2 score",
        fontdict={"fontsize": 18},
    )
    sns.regplot(
        x=correlation_2["placebo_effect"],
        y=correlation_2["iastay2_total"],
        scatter=False,
        color="black",
        line_kws={"color": "black", "alpha": 1, "lw": 2},
        ci=None,
        marker="o",
    )
    plt.xlabel("Placebo effect (%)", fontsize=18)
    plt.ylabel("IASTA2 score", fontsize=18)
    plt.savefig(opj(output_dir, "figures", "correlation_placebo_iasta2.png"))

    # Calculate the 90% confidence interval
    m, se = np.mean(diff + raw_diff), scipy.stats.sem(diff + raw_diff)
    h = se * scipy.stats.t.ppf((1 + 0.90) / 2.0, len(wide_dat) - 1)

    plt.figure(figsize=(4, 4))
    plt.scatter(diff.mean() / sd_diff_acc, 1, s=90)
    plt.plot(
        [
            diff.mean() / sd_diff_acc - h / sd_diff_acc,
            diff.mean() / sd_diff_acc + h / sd_diff_acc,
        ],
        [1, 1],
        linewidth=2,
    )
    plt.axvline(-0.5, color="k", linestyle="--")
    plt.axvline(0.5, color="k", linestyle="--")
    plt.yticks([])
    plt.tick_params(labelsize=14)
    plt.xticks([-0.5, 0, 0.5], ["-0.5", "0", "0.5"])
    plt.xlabel("Cohen's dz", fontsize=18)
    plt.savefig(
        opj(output_dir, "figures", "tost_discrim.svg"),
        transparent=True,
        bbox_inches="tight",
    )

    # Correlation between accuracy and pcs scores
    correlation_3_pcs = pg.corr(
        x=correlation_2["accuracy_all"],
        y=correlation_2["pcs_total"],
    )
    correlation_3_pcs.to_csv(opj(output_dir, "stats", "corr_3_pcs.csv"))

    plt.figure(figsize=(6, 6))
    plt.scatter(
        correlation_2["accuracy_all"],
        correlation_2["pcs_total"],
        s=90,
        color="mediumblue",
    )
    plt.xlabel("Discrimination accuracy (active - inactive)", fontsize=18)
    plt.ylabel("PCS score", fontsize=18)
    plt.tick_params(labelsize=14)
    plt.title(
        "Correlation between discrimination accuracy and PCS score",
        fontdict={"fontsize": 18},
    )
    sns.regplot(
        x=correlation_2["accuracy_all"],
        y=correlation_2["pcs_total"],
        scatter=False,
        color="black",
        line_kws={"color": "black", "alpha": 1, "lw": 2},
        ci=None,
        marker="o",
    )
    plt.xlabel("Discrimination accuracy", fontsize=18)
    plt.ylabel("PCS score", fontsize=18)
    plt.savefig(opj(output_dir, "figures", "correlation_accuracy_pcs.png"))

    # Correlation between accuracy and iasta1 scores
    correlation_3_iasta1 = pg.corr(
        x=correlation_2["accuracy_all"],
        y=correlation_2["iastay1_total"],
    )
    correlation_3_iasta1.to_csv(opj(output_dir, "stats", "corr_3_iasta1.csv"))

    plt.figure(figsize=(6, 6))
    plt.scatter(
        correlation_2["accuracy_all"],
        correlation_2["iastay1_total"],
        s=90,
        color="mediumblue",
    )
    plt.xlabel("Discrimination accuracy (active - inactive)", fontsize=18)
    plt.ylabel("IASTA1 score", fontsize=18)
    plt.tick_params(labelsize=14)
    plt.title(
        "Correlation between discrimination accuracy and IASTA1 score",
        fontdict={"fontsize": 18},
    )
    sns.regplot(
        x=correlation_2["accuracy_all"],
        y=correlation_2["iastay1_total"],
        scatter=False,
        color="black",
        line_kws={"color": "black", "alpha": 1, "lw": 2},
        ci=None,
        marker="o",
    )
    plt.xlabel("Discrimination accuracy", fontsize=18)
    plt.ylabel("IASTA1 score", fontsize=18)
    plt.savefig(opj(output_dir, "figures", "correlation_accuracy_iasta1.png"))

    # Correlation between accuracy and iasta2 scores
    correlation_3_iasta2 = pg.corr(
        x=correlation_2["accuracy_all"],
        y=correlation_2["iastay2_total"],
    )
    correlation_3_iasta2.to_csv(opj(output_dir, "stats", "corr_3_iasta2.csv"))

    plt.figure(figsize=(6, 6))
    plt.scatter(
        correlation_2["accuracy_all"],
        correlation_2["iastay2_total"],
        s=90,
        color="mediumblue",
    )
    plt.xlabel("Discrimination accuracy (active - inactive)", fontsize=18)
    plt.ylabel("IASTA2 score", fontsize=18)
    plt.tick_params(labelsize=14)
    plt.title(
        "Correlation between discrimination accuracy and IASTA2 score",
        fontdict={"fontsize": 18},
    )
    sns.regplot(
        x=correlation_2["accuracy_all"],
        y=correlation_2["iastay2_total"],
        scatter=False,
        color="black",
        line_kws={"color": "black", "alpha": 1, "lw": 2},
        ci=None,
        marker="o",
    )
    plt.xlabel("Discrimination accuracy", fontsize=18)
    plt.ylabel("IASTA2 score", fontsize=18)
    plt.savefig(opj(output_dir, "figures", "correlation_accuracy_iasta2.png"))

    # Check if difference between active and inactive x block
    anova_dat = wide_dat.melt(
        id_vars="participant",
        value_vars=["active_acc_b1", "inactive_acc_b1", "active_acc_b2", "inactive_acc_b2"],
    )
    anova_dat["block"] = np.where(
        anova_dat["variable"].isin(["active_acc_b1", "inactive_acc_b1"]), 1, 2
    )
    anova_dat["condition"] = np.where(
        anova_dat["variable"].isin(["active_acc_b1", "active_acc_b2"]), "active", "inactive"
    )

    anova_dat["value"] = anova_dat["value"].astype(float)
    out = pg.rm_anova(
        data=anova_dat, dv="value", within=["condition", "block"], subject="participant"
    )
    out.to_csv(opj(output_dir, "stats", "rm_anova_cond-block_acc.csv"))

    # Plot accuracy by condition
    anova_dat = wide_dat.melt(
        id_vars="participant", value_vars=["active_acc_all", "inactive_acc_all"]
    )
    color = sns.color_palette("Set2")[2:]
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(
        x="variable",
        y="value",
        hue="variable",
        data=anova_dat,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "15",
            "zorder": 999,
        },
        showcaps=False,
        palette=pal,
    )

    anova_dat["jitter"] = np.random.normal(0, 0.05, size=len(anova_dat))
    anova_dat["condition_jitter"] = np.where(
        anova_dat["variable"] == "active_acc_all",
        0 + anova_dat["jitter"],
        1 + anova_dat["jitter"],
    )

    sns.stripplot(
        x="condition_jitter",
        y="value",
        data=anova_dat,
        alpha=0.5,
        size=12,
        jitter=False,
        edgecolor="black",
        linewidth=1,
        palette=pal,
        native_scale=True,
        hue="variable",
    )
    anova_dat.index = anova_dat["participant"]
    ax.set_xlim([-1, 2])

    anova_jitter_active = anova_dat[anova_dat["variable"] == "active_acc_all"]
    anova_jitter_inactive = anova_dat[anova_dat["variable"] == "inactive_acc_all"]

    for p in list(anova_dat.index):
        plt.plot(
            [
                0 + anova_jitter_active.loc[p, "jitter"],
                1 + anova_jitter_inactive.loc[p, "jitter"],
            ],
            [
                anova_dat[anova_dat["variable"] == "active_acc_all"].loc[p, "value"],
                anova_dat[anova_dat["variable"] == "inactive_acc_all"].loc[p, "value"],
            ],
            color="gray",
            alpha=0.5,
        )
    plt.xticks([0, 1], ["TENS on", "TENS off"], fontsize=12)
    plt.ylabel("Proportion correct", fontsize=18)
    plt.xlabel("", fontsize=18)
    plt.tick_params(labelsize=14)
    ax.legend().remove()
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.18, top=0.98)
    plt.savefig(
        opj(output_dir, "figures", "discrim_acc_cond.png"),
        transparent=True,
        bbox_inches="tight",
    )

    # Plot d prime by condition
    anova_dat = wide_dat.melt(
        id_vars="participant", value_vars=["d_prime_active", "d_prime_inactive"]
    )
    color = sns.color_palette("Set2")[2:]
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(
        x="variable",
        y="value",
        hue="variable",
        data=anova_dat,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "15",
            "zorder": 999,
        },
        showcaps=False,
        palette=pal,
    )

    anova_dat["jitter"] = np.random.normal(0, 0.05, size=len(anova_dat))
    anova_dat["condition_jitter"] = np.where(
        anova_dat["variable"] == "d_prime_active",
        0 + anova_dat["jitter"],
        1 + anova_dat["jitter"],
    )

    sns.stripplot(
        x="condition_jitter",
        y="value",
        data=anova_dat,
        alpha=0.5,
        size=12,
        jitter=False,
        edgecolor="black",
        linewidth=1,
        palette=pal,
        native_scale=True,
        hue="variable",
    )
    anova_dat.index = anova_dat["participant"]
    ax.set_xlim([-1, 2])

    anova_jitter_active = anova_dat[anova_dat["variable"] == "d_prime_active"]
    anova_jitter_inactive = anova_dat[anova_dat["variable"] == "d_prime_inactive"]

    for p in list(anova_dat.index):
        plt.plot(
            [
                0 + anova_jitter_active.loc[p, "jitter"],
                1 + anova_jitter_inactive.loc[p, "jitter"],
            ],
            [
                anova_dat[anova_dat["variable"] == "d_prime_active"].loc[p, "value"],
                anova_dat[anova_dat["variable"] == "d_prime_inactive"].loc[p, "value"],
            ],
            color="gray",
            alpha=0.5,
        )
    plt.xticks([0, 1], ["Placebo on", "Placebo off"], fontsize=12)
    plt.ylabel("d-prime", fontsize=18)
    plt.xlabel("", fontsize=18)
    plt.tick_params(labelsize=14)
    plt.title("Sensitivity\nduring the placebo phase", fontdict={"fontsize": 18})
    ax.legend().remove()
    plt.tight_layout()
    plt.savefig(
        opj(output_dir, "figures", "discrim_dprime_cond.png"), dpi=800, bbox_inches="tight"
    )

    # Plot bias by condition
    anova_dat = wide_dat.melt(
        id_vars="participant", value_vars=["beta_active", "beta_inactive"]
    )
    color = sns.color_palette("Set2")[2:]
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(
        x="variable",
        y="value",
        hue="variable",
        data=anova_dat,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "15",
            "zorder": 999,
        },
        showcaps=False,
        palette=pal,
    )

    anova_dat["jitter"] = np.random.normal(0, 0.05, size=len(anova_dat))
    anova_dat["condition_jitter"] = np.where(
        anova_dat["variable"] == "beta_active",
        0 + anova_dat["jitter"],
        1 + anova_dat["jitter"],
    )

    sns.stripplot(
        x="condition_jitter",
        y="value",
        data=anova_dat,
        alpha=0.5,
        size=12,
        jitter=False,
        edgecolor="black",
        linewidth=1,
        palette=pal,
        native_scale=True,
        hue="variable",
    )
    anova_dat.index = anova_dat["participant"]
    ax.set_xlim([-1, 2])

    anova_jitter_active = anova_dat[anova_dat["variable"] == "beta_active"]
    anova_jitter_inactive = anova_dat[anova_dat["variable"] == "beta_inactive"]

    for p in list(anova_dat.index):
        plt.plot(
            [
                0 + anova_jitter_active.loc[p, "jitter"],
                1 + anova_jitter_inactive.loc[p, "jitter"],
            ],
            [
                anova_dat[anova_dat["variable"] == "beta_active"].loc[p, "value"],
                anova_dat[anova_dat["variable"] == "beta_inactive"].loc[p, "value"],
            ],
            color="gray",
            alpha=0.5,
        )
    plt.xticks([0, 1], ["Placebo on", "Placebo off"], fontsize=12)
    plt.ylabel("Bias", fontsize=18)
    plt.xlabel("", fontsize=18)
    plt.tick_params(labelsize=14)
    plt.title("Bias\nduring the placebo phase", fontdict={"fontsize": 18})
    ax.legend().remove()
    plt.tight_layout()
    plt.savefig(
        opj(output_dir, "figures", "discrim_beta_cond.png"), dpi=800, bbox_inches="tight"
    )

    # Plot c by condition
    anova_dat = wide_dat.melt(
        id_vars="participant", value_vars=["c_active", "c_inactive"]
    )
    color = sns.color_palette("Set2")[2:]
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(
        x="variable",
        y="value",
        hue="variable",
        data=anova_dat,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "15",
            "zorder": 999,
        },
        showcaps=False,
        palette=pal,
    )

    anova_dat["jitter"] = np.random.normal(0, 0.05, size=len(anova_dat))
    anova_dat["condition_jitter"] = np.where(
        anova_dat["variable"] == "c_active",
        0 + anova_dat["jitter"],
        1 + anova_dat["jitter"],
    )

    sns.stripplot(
        x="condition_jitter",
        y="value",
        data=anova_dat,
        alpha=0.5,
        size=12,
        jitter=False,
        edgecolor="black",
        linewidth=1,
        palette=pal,
        native_scale=True,
        hue="variable",
    )
    anova_dat.index = anova_dat["participant"]
    ax.set_xlim([-1, 2])

    anova_jitter_active = anova_dat[anova_dat["variable"] == "c_active"]
    anova_jitter_inactive = anova_dat[anova_dat["variable"] == "c_inactive"]

    for p in list(anova_dat.index):
        plt.plot(
            [
                0 + anova_jitter_active.loc[p, "jitter"],
                1 + anova_jitter_inactive.loc[p, "jitter"],
            ],
            [
                anova_dat[anova_dat["variable"] == "c_active"].loc[p, "value"],
                anova_dat[anova_dat["variable"] == "c_inactive"].loc[p, "value"],
            ],
            color="gray",
            alpha=0.5,
        )
    plt.xticks([0, 1], ["Placebo on", "Placebo off"], fontsize=12)
    plt.ylabel("Criterion", fontsize=18)
    plt.xlabel("", fontsize=18)
    plt.tick_params(labelsize=14)
    plt.title(
        "Discrimination performance\nduring the placebo phase", fontdict={"fontsize": 18}
    )
    ax.legend().remove()
    plt.tight_layout()
    plt.savefig(
        opj(output_dir, "figures", "discrim_c_cond.png"), dpi=800, bbox_inches="tight"
    )

    # Plot extinction placebo effect
    anova_dat = wide_dat.melt(
        id_vars="participant",
        value_vars=[
            "placebo_b1_1",
            "placebo_b1_2",
            "placebo_b1_3",
            "placebo_b1_4",
            "placebo_b2_1",
            "placebo_b2_2",
            "placebo_b2_3",
            "placebo_b2_4",
        ],
    )

    anova_dat[["condition", "block", "trial"]] = anova_dat["variable"].str.split(
        "_", expand=True
    )

    anova_dat["value"] = anova_dat["value"].astype(float)
    anova_dat["trial"] = anova_dat["trial"].astype(float)

    out = pg.rm_anova(
        data=anova_dat, dv="value", within=["block", "trial"], subject="participant"
    )

    plt.figure()
    sns.lmplot(
        x="trial",
        y="value",
        data=anova_dat,
        hue="block",
        legend=False,
        scatter_kws={"s": 20, "alpha": 0.6},
        palette=color,
    )
    plt.xlim([0, 5])
    plt.xlabel("Essais", fontsize=18)
    plt.xticks([1, 2, 3, 4])
    L = plt.legend()
    L.get_texts()[0].set_text("Bloc 1")
    L.get_texts()[0].set_fontsize(14)
    L.get_texts()[1].set_text("Bloc 2")
    L.get_texts()[1].set_fontsize(14)
    plt.tick_params(labelsize=14)
    plt.ylabel("Placebo effect (TENS on - TENS off)", fontsize=18)
    plt.savefig(opj(output_dir, "figures", "extinction_placebo.png"))

    # =========================================================================
    # DESCRIPTIVE STATISTICS
    # Compute summary statistics for temperatures, placebo effect, and accuracy
    # =========================================================================
    last_values = pd.read_csv(opj(output_dir, "data_wide_dat_withexcl.csv"))

    temp_pic_mean = last_values["temp_pic"].mean()
    temp_pic_max = last_values["temp_pic"].max()
    temp_pic_min = last_values["temp_pic"].min()
    temp_pic_sd = last_values["temp_pic"].std()

    temp_plateau_mean = last_values["temp_plateau"].mean()
    temp_plateau_max = last_values["temp_plateau"].max()
    temp_plateau_min = last_values["temp_plateau"].min()
    temp_plateau_sd = last_values["temp_plateau"].std()

    temp_placebo_mean = last_values["temp_placebo"].mean()
    temp_placebo_max = last_values["temp_placebo"].max()
    temp_placebo_min = last_values["temp_placebo"].min()
    temp_placebo_sd = last_values["temp_placebo"].std()

    placebo_effect_mean = last_values["perc_placebo_all"].mean()
    placebo_effect_max = last_values["perc_placebo_all"].max()
    placebo_effect_min = last_values["perc_placebo_all"].min()
    placebo_effect_sd = last_values["perc_placebo_all"].std()

    accuracy_all_mean = last_values["acc_all"].mean()
    accuracy_all_max = last_values["acc_all"].max()
    accuracy_all_min = last_values["acc_all"].min()
    accuracy_all_sd = last_values["acc_all"].std()

    ci_all = stats.t.interval(
        0.95,
        len(last_values["acc_all"]) - 1,
        loc=np.mean(last_values["acc_all"]),
        scale=stats.sem(last_values["acc_all"]),
    )

    accuracy_active_mean = last_values["active_acc_all"].mean()
    accuracy_active_max = last_values["active_acc_all"].max()
    accuracy_active_min = last_values["active_acc_all"].min()
    accuracy_active_sd = last_values["active_acc_all"].std()

    ci_active = stats.t.interval(
        0.95,
        len(last_values["active_acc_all"]) - 1,
        loc=np.mean(last_values["active_acc_all"]),
        scale=stats.sem(last_values["active_acc_all"]),
    )

    accuracy_inactive_mean = last_values["inactive_acc_all"].mean()
    accuracy_inactive_max = last_values["inactive_acc_all"].max()
    accuracy_inactive_min = last_values["inactive_acc_all"].min()
    accuracy_inactive_sd = last_values["inactive_acc_all"].std()

    ci_inactive = stats.t.interval(
        0.95,
        len(last_values["inactive_acc_all"]) - 1,
        loc=np.mean(last_values["inactive_acc_all"]),
        scale=stats.sem(last_values["inactive_acc_all"]),
    )

    temp_values = pd.DataFrame(
        {
            "temp_pic_mean": [temp_pic_mean],
            "temp_pic_max": [temp_pic_max],
            "temp_pic_min": [temp_pic_min],
            "temp_plateau_mean": [temp_plateau_mean],
            "temp_plateau_max": [temp_plateau_max],
            "temp_plateau_min": [temp_plateau_min],
            "temp_placebo_mean": [temp_placebo_mean],
            "temp_placebo_max": [temp_placebo_max],
            "temp_placebo_min": [temp_placebo_min],
            "placebo_effect_mean": [placebo_effect_mean],
            "placebo_effect_max": [placebo_effect_max],
            "placebo_effect_min": [placebo_effect_min],
            "placebo_effect_sd": [placebo_effect_sd],
            "accuracy_all_mean": [accuracy_all_mean],
            "accuracy_all_max": [accuracy_all_max],
            "accuracy_all_min": [accuracy_all_min],
            "accuracy_all_ci_lower": [ci_all[0]],
            "accuracy_all_ci_upper": [ci_all[1]],
            "accuracy_active_ci_lower": [ci_active[0]],
            "accuracy_active_ci_upper": [ci_active[1]],
            "accuracy_active_mean": [accuracy_active_mean],
            "accuracy_active_max": [accuracy_active_max],
            "accuracy_active_min": [accuracy_active_min],
            "accuracy_inactive_mean": [accuracy_inactive_mean],
            "accuracy_inactive_max": [accuracy_inactive_max],
            "accuracy_inactive_min": [accuracy_inactive_min],
            "accuracy_inactive_ci_lower": [ci_inactive[0]],
            "accuracy_inactive_ci_upper": [ci_inactive[1]],
            "accuracy_all_sd": [accuracy_all_sd],
            "accuracy_active_sd": [accuracy_active_sd],
            "accuracy_inactive_sd": [accuracy_inactive_sd],
        }
    )
    temp_values.to_csv(opj(output_dir, "stats", "temp_values_filtered.csv"))

    # Get same values for all participants (including excluded ones)
    last_values_all = pd.read_csv(opj(output_dir, "data_wide_dat_withexcl.csv"))

    temp_pic_mean_all = last_values_all["temp_pic"].mean()
    temp_pic_max_all = last_values_all["temp_pic"].max()
    temp_pic_min_all = last_values_all["temp_pic"].min()
    temp_pic_sd = last_values_all["temp_pic"].std()

    temp_plateau_mean_all = last_values_all["temp_plateau"].mean()
    temp_plateau_max_all = last_values_all["temp_plateau"].max()
    temp_plateau_min_all = last_values_all["temp_plateau"].min()
    temp_plateau_sd = last_values_all["temp_plateau"].std()

    temp_placebo_mean_all = last_values_all["temp_placebo"].mean()
    temp_placebo_max_all = last_values_all["temp_placebo"].max()
    temp_placebo_min_all = last_values_all["temp_placebo"].min()
    temp_placebo_sd = last_values_all["temp_placebo"].std()

    temp_values_all = pd.DataFrame(
        {
            "temp_pic_mean_all": [temp_pic_mean_all],
            "temp_pic_max_all": [temp_pic_max_all],
            "temp_pic_min_all": [temp_pic_min_all],
            "temp_plateau_mean_all": [temp_plateau_mean_all],
            "temp_plateau_max_all": [temp_plateau_max_all],
            "temp_plateau_min_all": [temp_plateau_min_all],
            "temp_placebo_mean_all": [temp_placebo_mean_all],
            "temp_placebo_max_all": [temp_placebo_max_all],
            "temp_placebo_min_all": [temp_placebo_min_all],
            "temp_pic_sd": [temp_pic_sd],
            "temp_plateau_sd": [temp_plateau_sd],
            "temp_placebo_sd": [temp_placebo_sd],
        }
    )
    temp_values_all.to_csv(opj(output_dir, "stats", "temp_values_all.csv"))

    # =========================================================================
    # SOCIODEMOGRAPHIC STATISTICS
    # Compute sample characteristics (age, gender distribution)
    # =========================================================================
    socio_demo_stats = pd.read_csv(opj(output_dir, "data_wide_dat_withexcl.csv"))

    total_participants = len(socio_demo_stats)
    total_males = sum(socio_demo_stats["ismale"])
    total_females = sum(socio_demo_stats["isfemale"])
    total_others = sum(socio_demo_stats["Autres"])
    mean_age = socio_demo_stats["age"].mean()
    sd_age = socio_demo_stats["age"].std()
    min_age = socio_demo_stats["age"].min()
    max_age = socio_demo_stats["age"].max()

    age_males_mean = socio_demo_stats[socio_demo_stats["ismale"] == 1]["age"].mean()
    age_females_mean = socio_demo_stats[socio_demo_stats["isfemale"] == 1]["age"].mean()
    age_others_mean = socio_demo_stats[socio_demo_stats["Autres"] == 1]["age"].mean()
    age_males_min = socio_demo_stats[socio_demo_stats["ismale"] == 1]["age"].min()
    age_females_min = socio_demo_stats[socio_demo_stats["isfemale"] == 1]["age"].min()
    age_others_min = socio_demo_stats[socio_demo_stats["Autres"] == 1]["age"].min()
    age_males_max = socio_demo_stats[socio_demo_stats["ismale"] == 1]["age"].max()
    age_females_max = socio_demo_stats[socio_demo_stats["isfemale"] == 1]["age"].max()
    age_others_max = socio_demo_stats[socio_demo_stats["Autres"] == 1]["age"].max()

    socio_demo_summary = pd.DataFrame(
        {
            "total_participants": [total_participants],
            "total_males": [total_males],
            "total_females": [total_females],
            "total_others": [total_others],
            "mean_age": [mean_age],
            "sd_age": [sd_age],
            "min_age": [min_age],
            "max_age": [max_age],
            "age_males_mean": [age_males_mean],
            "age_females_mean": [age_females_mean],
            "age_others_mean": [age_others_mean],
            "age_males_min": [age_males_min],
            "age_females_min": [age_females_min],
            "age_others min": [age_others_min],
            "age_males_max": [age_males_max],
            "age_females_max": [age_females_max],
            "age_others_max": [age_others_max],
        }
    )
    socio_demo_summary.to_csv(opj(output_dir, "stats", "socio_demo_summary.csv"))

    # Missed responses
    mean_missed_responses = wide_dat["discrim_missed_responses"].mean()
    std_missed_responses = wide_dat["discrim_missed_responses"].std()
    min_missed_responses = wide_dat["discrim_missed_responses"].min()
    max_missed_responses = wide_dat["discrim_missed_responses"].max()

    discrim_missed_responses = pd.DataFrame(
        {
            "mean_missed_responses": [mean_missed_responses],
            "std_missed_responses": [std_missed_responses],
            "min_missed_responses": [min_missed_responses],
            "max_missed_responses": [max_missed_responses],
        }
    )
    discrim_missed_responses.to_csv(
        opj(output_dir, "stats", "discrim_missed_responses_stats.csv")
    )

    wide_dat[["participant", "discrim_missed_responses"]].to_csv(
        opj(output_dir, "stats", "discrim_missed_responses.csv")
    )

    # =========================================================================
    # MANUSCRIPT RESULTS COMPILATION
    # Compile all key statistics into a single CSV for easy manuscript writing
    # Each variable corresponds to a statistic reported in the paper
    # =========================================================================
    print("\nCompiling manuscript results...")

    manuscript_results = {}

    # --- Sample characteristics ---
    manuscript_results["n_final_sample"] = total_participants
    manuscript_results["n_males"] = int(total_males)
    manuscript_results["n_females"] = int(total_females)
    manuscript_results["n_nonbinary"] = int(total_others)
    manuscript_results["age_mean"] = round(mean_age, 2)
    manuscript_results["age_sd"] = round(sd_age, 2)
    manuscript_results["age_min"] = int(min_age)
    manuscript_results["age_max"] = int(max_age)

    # --- Exclusion lists (non-overlapping, hierarchy: custom > perfect/below-chance > placebo) ---
    # Custom exclusions (highest priority)
    report_custom = set(excluded_custom)

    # Perfect discrimination and below-chance accuracy combined (excluding those already in custom)
    report_perfect_belowchance = (set(excluded_perfect) | set(excluded_low_acc)) - report_custom

    # Low placebo (excluding those already reported above)
    report_placebo = set(excluded_placebo) - report_custom - report_perfect_belowchance

    manuscript_results["n_excluded_custom"] = len(report_custom)
    manuscript_results["excluded_custom"] = "; ".join(sorted(report_custom)) if report_custom else ""
    manuscript_results["n_excluded_perfect_belowchance"] = len(report_perfect_belowchance)
    manuscript_results["excluded_perfect_belowchance"] = "; ".join(sorted(report_perfect_belowchance)) if report_perfect_belowchance else ""
    manuscript_results["n_excluded_low_placebo"] = len(report_placebo)
    manuscript_results["excluded_low_placebo"] = "; ".join(sorted(report_placebo)) if report_placebo else ""

    # --- Placebo effect descriptives ---
    manuscript_results["placebo_effect_mean_pct"] = round(placebo_effect_mean, 2)
    manuscript_results["placebo_effect_min_pct"] = round(placebo_effect_min, 2)
    manuscript_results["placebo_effect_max_pct"] = round(placebo_effect_max, 2)
    manuscript_results["placebo_effect_sd_pct"] = round(placebo_effect_sd, 2)

    # --- RM ANOVA: Condition x Block on pain ratings (placebo trials) ---
    # Recompute with full output
    all_eval_frame_placeb_anova = all_eval_frame[
        all_eval_frame.trial.isin([8, 9, 10, 11, 20, 21, 22, 23])
    ]
    all_eval_frame_placeb_anova = (
        all_eval_frame_placeb_anova.groupby(["participant", "condition", "block"])
        .mean(numeric_only=True)
        .reset_index()
    )
    anova_eval = pg.rm_anova(
        data=all_eval_frame_placeb_anova,
        dv="ratings",
        within=["condition", "block"],
        subject="participant",
        correction=False,
    )
    # Extract condition effect
    cond_row = anova_eval[anova_eval["Source"] == "condition"].iloc[0]
    manuscript_results["anova_condition_F"] = round(cond_row["F"], 2)
    manuscript_results["anova_condition_p"] = cond_row["p-unc"]
    manuscript_results["anova_condition_df1"] = int(cond_row["ddof1"])
    manuscript_results["anova_condition_df2"] = int(cond_row["ddof2"])
    manuscript_results["anova_condition_eta2"] = round(cond_row["ng2"], 3)

    # Extract block effect
    block_row = anova_eval[anova_eval["Source"] == "block"].iloc[0]
    manuscript_results["anova_block_F"] = round(block_row["F"], 2)
    manuscript_results["anova_block_p"] = round(block_row["p-unc"], 4)
    manuscript_results["anova_block_df1"] = int(block_row["ddof1"])
    manuscript_results["anova_block_df2"] = int(block_row["ddof2"])
    manuscript_results["anova_block_eta2"] = round(block_row["ng2"], 3)

    # Extract interaction
    inter_row = anova_eval[anova_eval["Source"] == "condition * block"].iloc[0]
    manuscript_results["anova_interaction_F"] = round(inter_row["F"], 2)
    manuscript_results["anova_interaction_p"] = round(inter_row["p-unc"], 4)
    manuscript_results["anova_interaction_df1"] = int(inter_row["ddof1"])
    manuscript_results["anova_interaction_df2"] = int(inter_row["ddof2"])
    manuscript_results["anova_interaction_eta2"] = round(inter_row["ng2"], 3)

    # --- Pain rating t-test (TENS active vs inactive during placebo trials) ---
    all_eval_frame_placeb_ttest = all_eval_frame[
        all_eval_frame.trial.isin([8, 9, 10, 11, 20, 21, 22, 23])
    ]
    all_eval_frame_placeb_ttest = (
        all_eval_frame_placeb_ttest.groupby(["participant", "condition"])
        .mean(numeric_only=True)
        .reset_index()
    )
    active_ratings = all_eval_frame_placeb_ttest[
        all_eval_frame_placeb_ttest["condition"] == "active"
    ]["ratings"].values
    inactive_ratings = all_eval_frame_placeb_ttest[
        all_eval_frame_placeb_ttest["condition"] == "inactive"
    ]["ratings"].values

    pain_ttest = pg.ttest(active_ratings, inactive_ratings, paired=True)
    manuscript_results["pain_ttest_t"] = round(pain_ttest["T"].values[0], 2)
    manuscript_results["pain_ttest_p"] = pain_ttest["p-val"].values[0]
    manuscript_results["pain_ttest_df"] = int(pain_ttest["dof"].values[0])
    manuscript_results["pain_ttest_d"] = round(pain_ttest["cohen-d"].values[0], 2)

    # --- Discrimination accuracy descriptives ---
    manuscript_results["acc_overall_mean"] = round(accuracy_all_mean * 100, 2)
    manuscript_results["acc_overall_min"] = round(accuracy_all_min * 100, 2)
    manuscript_results["acc_overall_max"] = round(accuracy_all_max * 100, 2)
    manuscript_results["acc_overall_sd"] = round(accuracy_all_sd * 100, 2)
    manuscript_results["acc_overall_ci_lower"] = round(ci_all[0] * 100, 2)
    manuscript_results["acc_overall_ci_upper"] = round(ci_all[1] * 100, 2)

    manuscript_results["acc_active_mean"] = round(accuracy_active_mean * 100, 2)
    manuscript_results["acc_active_min"] = round(accuracy_active_min * 100, 2)
    manuscript_results["acc_active_max"] = round(accuracy_active_max * 100, 2)
    manuscript_results["acc_active_sd"] = round(accuracy_active_sd * 100, 2)
    manuscript_results["acc_active_ci_lower"] = round(ci_active[0] * 100, 2)
    manuscript_results["acc_active_ci_upper"] = round(ci_active[1] * 100, 2)

    manuscript_results["acc_inactive_mean"] = round(accuracy_inactive_mean * 100, 2)
    manuscript_results["acc_inactive_min"] = round(accuracy_inactive_min * 100, 2)
    manuscript_results["acc_inactive_max"] = round(accuracy_inactive_max * 100, 2)
    manuscript_results["acc_inactive_sd"] = round(accuracy_inactive_sd * 100, 2)
    manuscript_results["acc_inactive_ci_lower"] = round(ci_inactive[0] * 100, 2)
    manuscript_results["acc_inactive_ci_upper"] = round(ci_inactive[1] * 100, 2)

    # --- Discrimination t-test (TENS active vs inactive) ---
    discrim_ttest = pg.ttest(
        last_values["inactive_acc_all"].astype(float),
        last_values["active_acc_all"].astype(float),
        paired=True,
    )
    manuscript_results["discrim_ttest_t"] = round(discrim_ttest["T"].values[0], 2)
    manuscript_results["discrim_ttest_p"] = round(discrim_ttest["p-val"].values[0], 4)
    manuscript_results["discrim_ttest_df"] = int(discrim_ttest["dof"].values[0])
    manuscript_results["discrim_ttest_dz"] = round(discrim_ttest["cohen-d"].values[0], 2)

    # --- Bayes Factor for discrimination t-test ---
    bf_result = pg.ttest(
        last_values["inactive_acc_all"].astype(float),
        last_values["active_acc_all"].astype(float),
        paired=True,
    )
    bf_value = bf_result["BF10"].values[0]
    # BF10 can be a string like 'inf' or a float
    try:
        manuscript_results["discrim_BF10"] = round(float(bf_value), 3)
    except (ValueError, TypeError):
        manuscript_results["discrim_BF10"] = str(bf_value)

    # --- TOST equivalence test ---
    tost_mean_diff_val = np.mean(
        last_values["inactive_acc_all"] - last_values["active_acc_all"]
    )
    tost_sd_diff_val = np.std(
        last_values["inactive_acc_all"] - last_values["active_acc_all"], ddof=1
    )
    tost_bounds_cohen = 0.45
    tost_bounds_raw = tost_bounds_cohen * tost_sd_diff_val

    p_tost, t_low, t_high = ttost_paired(
        last_values["inactive_acc_all"].values.astype(float),
        last_values["active_acc_all"].values.astype(float),
        low=-tost_bounds_raw,
        upp=tost_bounds_raw,
    )
    manuscript_results["tost_mean_diff"] = round(tost_mean_diff_val, 3)
    manuscript_results["tost_sd_diff"] = round(tost_sd_diff_val, 3)
    manuscript_results["tost_bounds_cohen_d"] = tost_bounds_cohen
    manuscript_results["tost_lower_t"] = round(t_low[0], 2)
    manuscript_results["tost_lower_p"] = t_low[1]
    manuscript_results["tost_lower_df"] = int(t_low[2])
    manuscript_results["tost_upper_t"] = round(t_high[0], 2)
    manuscript_results["tost_upper_p"] = round(t_high[1], 4)
    manuscript_results["tost_upper_df"] = int(t_high[2])
    manuscript_results["tost_p"] = p_tost

    # --- Correlation: Placebo effect vs Discrimination accuracy ---
    corr_placebo_acc = pg.corr(
        x=correlation_df["placebo_effect"], y=correlation_df["accuracy_all"]
    )
    manuscript_results["corr_placebo_acc_r"] = round(corr_placebo_acc["r"].values[0], 2)
    manuscript_results["corr_placebo_acc_p"] = round(corr_placebo_acc["p-val"].values[0], 4)
    manuscript_results["corr_placebo_acc_n"] = int(corr_placebo_acc["n"].values[0])

    # --- Questionnaire correlations ---
    # STAI-Y1 vs placebo effect
    corr_stai1_placebo = pg.corr(
        x=correlation_2["placebo_effect"], y=correlation_2["iastay1_total"]
    )
    manuscript_results["corr_staiy1_placebo_r"] = round(
        corr_stai1_placebo["r"].values[0], 2
    )
    manuscript_results["corr_staiy1_placebo_p"] = round(
        corr_stai1_placebo["p-val"].values[0], 4
    )

    # STAI-Y1 vs accuracy
    corr_stai1_acc = pg.corr(
        x=correlation_2["accuracy_all"], y=correlation_2["iastay1_total"]
    )
    manuscript_results["corr_staiy1_acc_r"] = round(corr_stai1_acc["r"].values[0], 2)
    manuscript_results["corr_staiy1_acc_p"] = round(corr_stai1_acc["p-val"].values[0], 4)

    # STAI-Y2 vs placebo effect
    corr_stai2_placebo = pg.corr(
        x=correlation_2["placebo_effect"], y=correlation_2["iastay2_total"]
    )
    manuscript_results["corr_staiy2_placebo_r"] = round(
        corr_stai2_placebo["r"].values[0], 2
    )
    manuscript_results["corr_staiy2_placebo_p"] = round(
        corr_stai2_placebo["p-val"].values[0], 4
    )

    # STAI-Y2 vs accuracy
    corr_stai2_acc = pg.corr(
        x=correlation_2["accuracy_all"], y=correlation_2["iastay2_total"]
    )
    manuscript_results["corr_staiy2_acc_r"] = round(corr_stai2_acc["r"].values[0], 2)
    manuscript_results["corr_staiy2_acc_p"] = round(corr_stai2_acc["p-val"].values[0], 4)

    # PCS vs placebo effect
    corr_pcs_placebo = pg.corr(
        x=correlation_2["placebo_effect"], y=correlation_2["pcs_total"]
    )
    manuscript_results["corr_pcs_placebo_r"] = round(corr_pcs_placebo["r"].values[0], 2)
    manuscript_results["corr_pcs_placebo_p"] = round(
        corr_pcs_placebo["p-val"].values[0], 4
    )

    # PCS vs accuracy
    corr_pcs_acc = pg.corr(x=correlation_2["accuracy_all"], y=correlation_2["pcs_total"])
    manuscript_results["corr_pcs_acc_r"] = round(corr_pcs_acc["r"].values[0], 2)
    manuscript_results["corr_pcs_acc_p"] = round(corr_pcs_acc["p-val"].values[0], 4)

    # --- Temperature calibration values ---
    manuscript_results["temp_plateau_mean"] = round(temp_plateau_mean, 2)
    manuscript_results["temp_plateau_min"] = round(temp_plateau_min, 2)
    manuscript_results["temp_plateau_max"] = round(temp_plateau_max, 2)
    manuscript_results["temp_plateau_sd"] = round(temp_plateau_sd, 2)

    manuscript_results["temp_placebo_mean"] = round(temp_placebo_mean, 2)
    manuscript_results["temp_placebo_min"] = round(temp_placebo_min, 2)
    manuscript_results["temp_placebo_max"] = round(temp_placebo_max, 2)
    manuscript_results["temp_placebo_sd"] = round(temp_placebo_sd, 2)

    manuscript_results["temp_pic_mean"] = round(temp_pic_mean, 2)
    manuscript_results["temp_pic_min"] = round(temp_pic_min, 2)
    manuscript_results["temp_pic_max"] = round(temp_pic_max, 2)
    manuscript_results["temp_pic_sd"] = round(temp_pic_sd, 2)

    # Save manuscript results to CSV
    manuscript_df = pd.DataFrame([manuscript_results])
    manuscript_df.to_csv(opj(output_dir, "stats", "manuscript_results.csv"), index=False)

    # Also save a transposed version for easier reading
    manuscript_df_transposed = manuscript_df.T
    manuscript_df_transposed.columns = ["value"]
    manuscript_df_transposed.index.name = "statistic"
    manuscript_df_transposed.to_csv(
        opj(output_dir, "stats", "manuscript_results_readable.csv")
    )

    print(f"Manuscript results saved to {opj(output_dir, 'stats', 'manuscript_results.csv')}")
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
