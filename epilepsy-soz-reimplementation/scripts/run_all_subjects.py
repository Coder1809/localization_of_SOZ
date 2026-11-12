"""
scripts/run_all_subjects.py - Full Cohort Evaluation Pipeline
Evaluates the 22 paper subjects across both ENCS (Table 3) and ENCS+PE (Table 4) models,
generating per-subject electrode reports, clinical JSON summaries, diagnostic plots,
and exporting master cohort CSVs.
"""

import sys
import argparse
import time
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_DATASET_DIR, RESULTS_DIR, PAPER_COHORT
from src.dataset import BIDSDataSetLoader
from src.preprocessing import preprocess_recording
from src.features import extract_recording_dataset
from src.evaluation import (
    evaluate_5fold_cv,
    generate_channel_classification_report,
    plot_subject_results,
    generate_clinical_localization_summary
)


def run_cohort_evaluation(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    task: str = "interictal",
    max_subjects: Optional[int] = None,
    save_csv: bool = True,
    generate_plots: bool = True
):
    print("=" * 90)
    print(" REPLICATION OF PAPER EXPERIMENTS: FULL COHORT EVALUATION (TABLES 3 & 4)")
    print("=" * 90)
    print(f"Dataset: {dataset_dir}")
    print(f"Task: {task}")
    print(f"Target Cohort: {len(PAPER_COHORT)} SEEG Patients\n")

    loader = BIDSDataSetLoader(dataset_dir)
    subjects = loader.get_seeg_subjects(filter_paper_cohort=True)

    if not subjects:
        print("Warning: No paper cohort subjects found by exact match. Scanning all SEEG subjects...")
        subjects = loader.get_seeg_subjects(filter_paper_cohort=False)

    if max_subjects:
        subjects = subjects[:max_subjects]

    print(f"Evaluating {len(subjects)} subjects: {subjects}\n")

    table3_rows = []
    table4_rows = []
    comparison_rows = []

    start_total_time = time.time()

    for idx, sub_id in enumerate(subjects):
        t0 = time.time()
        print(f"[{idx + 1}/{len(subjects)}] Processing {sub_id} ...")

        recordings = loader.find_recordings(sub_id, task=task)
        if not recordings:
            recordings = loader.find_recordings(sub_id, task=None)

        if not recordings:
            print(f"  Skipping {sub_id}: No valid recordings found.")
            continue

        rec_info = recordings[0]
        rec_data = loader.load_recording(rec_info)

        if rec_data["n_soz"] == 0:
            print(f"  Skipping {sub_id}: No SOZ channels found in channels.tsv.")
            continue

        # Preprocessing
        preprocessed = preprocess_recording(
            rec_data["raw_data"],
            orig_srate=rec_data["sampling_rate"],
            target_srate=200.0
        )

        # -------------------------------------------------------------
        # Mode 1: ENCS only (Table 3)
        # -------------------------------------------------------------
        feat_encs = extract_recording_dataset(
            rec_data,
            preprocessed,
            feature_mode="encs",
            verbose=False
        )
        res_encs = evaluate_5fold_cv(feat_encs["X"], feat_encs["y"])

        table3_rows.append({
            "Patient ID": sub_id,
            "Epoch": feat_encs["total_epochs"],
            "Positives": feat_encs["n_positives"],
            "Negatives": feat_encs["n_negatives"],
            "AC (%)": round(res_encs["accuracy"], 2),
            "P (%)": round(res_encs["precision"], 2),
            "R (%)": round(res_encs["recall"], 2),
            "F1 (%)": round(res_encs["f1"], 2),
            "AUC (%)": round(res_encs["auc"], 2)
        })

        # -------------------------------------------------------------
        # Mode 2: ENCS + PE (Table 4)
        # -------------------------------------------------------------
        feat_encs_pe = extract_recording_dataset(
            rec_data,
            preprocessed,
            feature_mode="encs_pe",
            verbose=False
        )
        res_encs_pe = evaluate_5fold_cv(feat_encs_pe["X"], feat_encs_pe["y"])

        table4_rows.append({
            "Patient ID": sub_id,
            "Epoch": feat_encs_pe["total_epochs"],
            "Positives": feat_encs_pe["n_positives"],
            "Negatives": feat_encs_pe["n_negatives"],
            "AC (%)": round(res_encs_pe["accuracy"], 2),
            "P (%)": round(res_encs_pe["precision"], 2),
            "R (%)": round(res_encs_pe["recall"], 2),
            "F1 (%)": round(res_encs_pe["f1"], 2),
            "AUC (%)": round(res_encs_pe["auc"], 2)
        })

        # Side-by-side comparison row
        comparison_rows.append({
            "Patient ID": sub_id,
            "Surgical Target": rec_data.get("clinical_metadata", {}).get("target", "N/A"),
            "Good Channels": rec_data["n_channels"],
            "SOZ Channels": rec_data["n_soz"],
            "Table 3 AC (%)": round(res_encs["accuracy"], 2),
            "Table 4 AC (%)": round(res_encs_pe["accuracy"], 2),
            "Delta AC (%)": round(res_encs_pe["accuracy"] - res_encs["accuracy"], 2),
            "Table 3 AUC (%)": round(res_encs["auc"], 2),
            "Table 4 AUC (%)": round(res_encs_pe["auc"], 2),
            "Delta AUC (%)": round(res_encs_pe["auc"] - res_encs["auc"], 2)
        })

        # -------------------------------------------------------------
        # Per-Subject Detailed Artifacts Generation
        # -------------------------------------------------------------
        if save_csv:
            patient_dir = RESULTS_DIR / sub_id
            patient_dir.mkdir(parents=True, exist_ok=True)

            ch_report_df = generate_channel_classification_report(rec_data, feat_encs_pe, res_encs_pe)
            pred_csv_path = patient_dir / f"{sub_id}_encs_pe_electrode_predictions.csv"
            ch_report_df.to_csv(pred_csv_path, index=False)

            clinical_summary = generate_clinical_localization_summary(rec_data, ch_report_df)
            json_report_path = patient_dir / f"{sub_id}_encs_pe_clinical_report.json"
            with open(json_report_path, "w") as f:
                json.dump(clinical_summary, f, indent=2)

            if generate_plots:
                plot_subject_results(f"{sub_id}_encs_pe", res_encs_pe, ch_report_df, output_dir=str(patient_dir))

        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s | ENCS AC: {res_encs['accuracy']:.1f}% | ENCS+PE AC: {res_encs_pe['accuracy']:.1f}%, AUC: {res_encs_pe['auc']:.1f}% | Saved to results/{sub_id}/")

    df_t3 = pd.DataFrame(table3_rows)
    df_t4 = pd.DataFrame(table4_rows)
    df_comp = pd.DataFrame(comparison_rows)

    # Compute Averages
    if not df_t3.empty:
        avg_t3 = {
            "Patient ID": "Avg",
            "Epoch": "—",
            "Positives": "—",
            "Negatives": "—",
            "AC (%)": round(df_t3["AC (%)"].mean(), 2),
            "P (%)": round(df_t3["P (%)"].mean(), 2),
            "R (%)": round(df_t3["R (%)"].mean(), 2),
            "F1 (%)": round(df_t3["F1 (%)"].mean(), 2),
            "AUC (%)": round(df_t3["AUC (%)"].mean(), 2)
        }
        df_t3_full = pd.concat([df_t3, pd.DataFrame([avg_t3])], ignore_index=True)
    else:
        df_t3_full = df_t3

    if not df_t4.empty:
        avg_t4 = {
            "Patient ID": "Avg",
            "Epoch": "—",
            "Positives": "—",
            "Negatives": "—",
            "AC (%)": round(df_t4["AC (%)"].mean(), 2),
            "P (%)": round(df_t4["P (%)"].mean(), 2),
            "R (%)": round(df_t4["R (%)"].mean(), 2),
            "F1 (%)": round(df_t4["F1 (%)"].mean(), 2),
            "AUC (%)": round(df_t4["AUC (%)"].mean(), 2)
        }
        df_t4_full = pd.concat([df_t4, pd.DataFrame([avg_t4])], ignore_index=True)
    else:
        df_t4_full = df_t4

    # Print Results
    print("\n" + "=" * 90)
    print(" TABLE 3 REPLICATION: Performance evaluation of localization model using ENCS (%)")
    print("=" * 90)
    print(df_t3_full.to_string(index=False))

    print("\n" + "=" * 90)
    print(" TABLE 4 REPLICATION: Performance evaluation of localization model using ENCS and PE (%)")
    print("=" * 90)
    print(df_t4_full.to_string(index=False))

    if save_csv:
        path_t3 = RESULTS_DIR / "table3_encs_localization.csv"
        path_t4 = RESULTS_DIR / "table4_encs_pe_localization.csv"
        path_comp = RESULTS_DIR / "cohort_comparison_master.csv"

        path_t3_md = RESULTS_DIR / "table3_encs_localization.md"
        path_t4_md = RESULTS_DIR / "table4_encs_pe_localization.md"
        path_comp_md = RESULTS_DIR / "cohort_comparison_master.md"

        df_t3_full.to_csv(path_t3, index=False)
        df_t4_full.to_csv(path_t4, index=False)
        df_comp.to_csv(path_comp, index=False)

        try:
            with open(path_t3_md, "w") as f:
                f.write(f"# Table 3: Localization Performance Using ENCS (%)\n\n{df_t3_full.to_markdown(index=False)}\n")
            with open(path_t4_md, "w") as f:
                f.write(f"# Table 4: Localization Performance Using ENCS and PE (%)\n\n{df_t4_full.to_markdown(index=False)}\n")
            with open(path_comp_md, "w") as f:
                f.write(f"# Master Cohort Comparison Table (Table 3 vs. Table 4)\n\n{df_comp.to_markdown(index=False)}\n")
        except Exception:
            pass

        print(f"\n[Saved] Table 3 CSV & MD -> {path_t3.name}, {path_t3_md.name}")
        print(f"[Saved] Table 4 CSV & MD -> {path_t4.name}, {path_t4_md.name}")
        print(f"[Saved] Master Cohort Comparison CSV & MD -> {path_comp.name}, {path_comp_md.name}")
        print(f"[Saved] All 22 Electrode Predictions CSVs & Clinical JSONs -> {RESULTS_DIR}")
        print(f"[Saved] All 22 ROC & Confusion Matrix Plots -> {RESULTS_DIR}")

    total_duration = time.time() - start_total_time
    print(f"\nCompleted evaluation of {len(subjects)} subjects in {total_duration:.1f} seconds ({total_duration/60:.1f} minutes).")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full cohort SOZ localization evaluation")
    parser.add_argument("--task", type=str, default="interictal", help="Task name (default: interictal)")
    parser.add_argument("--max-subjects", type=int, default=None, help="Limit number of subjects to run")
    parser.add_argument("--no-save", action="store_true", help="Do not save CSVs")
    parser.add_argument("--no-plot", action="store_true", help="Do not save PNG plots")

    args = parser.parse_args()
    run_cohort_evaluation(
        task=args.task,
        max_subjects=args.max_subjects,
        save_csv=not args.no_save,
        generate_plots=not args.no_plot
    )
