"""
scripts/test_single_subject.py - Single-Subject End-to-End Pipeline
Executes preprocessing, WPLI network extraction, Takens/PE topological analysis,
5x focal balancing, PCA, SVM classification, and 5-fold cross-validation on sub-HUP142.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_DATASET_DIR, RESULTS_DIR
from src.dataset import BIDSDataSetLoader
from src.preprocessing import preprocess_recording
from src.features import extract_recording_dataset
import json
from src.evaluation import (
    evaluate_5fold_cv,
    generate_channel_classification_report,
    plot_subject_results,
    generate_clinical_localization_summary,
    print_clinical_report
)


def run_single_subject_pipeline(
    subject_id: str = "sub-HUP142",
    task: str = "interictal",
    run: str = "01",
    mode: str = "both",
    generate_plot: bool = True
):
    print("=" * 80)
    print(f" PIPELINE EXECUTION: {subject_id} (Task: {task}, Run: {run})")
    print("=" * 80)

    loader = BIDSDataSetLoader(DEFAULT_DATASET_DIR)
    recordings = loader.find_recordings(subject_id, task=task, run=run)
    if not recordings:
        print(f"Error: No recordings found for {subject_id} with task '{task}' and run '{run}'")
        return

    rec_info = recordings[0]
    print(f"Loading EDF: {rec_info['filename']} ...")
    rec_data = loader.load_recording(rec_info)

    print(f"\n[1] Dataset Statistics:")
    print(f"  Total Good SEEG Channels: {rec_data['n_channels']}")
    print(f"  Ground-Truth SOZ Channels: {rec_data['n_soz']} -> {rec_data['soz_channel_names']}")
    print(f"  Non-SOZ Channels: {rec_data['n_non_soz']}")
    print(f"  Bad Channels Excluded: {len(rec_data['bad_channel_names'])} -> {rec_data['bad_channel_names']}")
    print(f"  Original Sampling Rate: {rec_data['sampling_rate']} Hz")
    print(f"  Recording Duration: {rec_data['duration_sec']:.1f} s")

    # 2. Preprocessing
    print(f"\n[2] Preprocessing Signal:")
    print(f"  - Resampling to 200 Hz")
    print(f"  - 60 Hz Notch Filter")
    print(f"  - 0.5 - 80 Hz Butterworth Bandpass Filter")
    print(f"  - Common Average Reference (CAR)")
    preprocessed_data = preprocess_recording(
        rec_data["raw_data"],
        orig_srate=rec_data["sampling_rate"],
        target_srate=200.0
    )

    modes_to_test = ["encs", "encs_pe"] if mode == "both" else [mode]
    results_summary = {}

    for m in modes_to_test:
        mode_label = "ENCS only (Table 3)" if m == "encs" else "ENCS + PE (Table 4)"
        print(f"\n[3] Feature Extraction & 5x Focal Balancing [{mode_label}]:")
        
        feat_data = extract_recording_dataset(
            rec_data,
            preprocessed_data,
            feature_mode=m,
            verbose=True
        )

        print(f"  Total Balanced Epochs: {feat_data['total_epochs']}")
        print(f"  Positive (SOZ) Samples: {feat_data['n_positives']}")
        print(f"  Negative (Non-SOZ) Samples: {feat_data['n_negatives']}")
        print(f"  Feature Matrix X shape: {feat_data['X'].shape}")
        print(f"  Features: {feat_data['feature_names']}")

        # 4. 5-Fold Cross Validation
        print(f"\n[4] 5-Fold Stratified Cross-Validation (PCA + SVM RBF C=0.5):")
        eval_res = evaluate_5fold_cv(feat_data["X"], feat_data["y"])
        results_summary[m] = eval_res

        print(f"  >>> Accuracy (AC) : {eval_res['accuracy']:.2f} %")
        print(f"  >>> Precision (P) : {eval_res['precision']:.2f} %")
        print(f"  >>> Recall (R)    : {eval_res['recall']:.2f} %")
        print(f"  >>> F1-Score (F1) : {eval_res['f1']:.2f} %")
        print(f"  >>> ROC-AUC (AUC) : {eval_res['auc']:.2f} %")

        # 5. Physical Electrode Level Classification Report
        if m == "encs_pe" or len(modes_to_test) == 1:
            patient_dir = RESULTS_DIR / subject_id
            patient_dir.mkdir(parents=True, exist_ok=True)

            ch_report_df = generate_channel_classification_report(rec_data, feat_data, eval_res)
            report_csv_path = patient_dir / f"{subject_id}_{m}_electrode_predictions.csv"
            ch_report_df.to_csv(report_csv_path, index=False)
            
            # Generate structured clinical SOZ & EZ report
            clinical_summary = generate_clinical_localization_summary(rec_data, ch_report_df)
            json_report_path = patient_dir / f"{subject_id}_{m}_clinical_report.json"
            with open(json_report_path, "w") as f:
                json.dump(clinical_summary, f, indent=2)
            print(f"\n[5] Patient Results Directory: {patient_dir}")
            print(f"  • Electrode Predictions CSV : {report_csv_path.name}")
            print(f"  • Clinical SOZ/EZ Report JSON: {json_report_path.name}")

            # Print clinical report to terminal
            print_clinical_report(clinical_summary)

            if generate_plot:
                plot_path = plot_subject_results(f"{subject_id}_{m}", eval_res, ch_report_df, output_dir=str(patient_dir))
                print(f"  • Diagnostic Plot PNG        : {Path(plot_path).name}")

    # Comparison summary if both modes ran
    if len(modes_to_test) > 1:
        print("\n" + "=" * 80)
        print(f" COMPARISON SUMMARY FOR {subject_id}:")
        print("=" * 80)
        encs_res = results_summary["encs"]
        encs_pe_res = results_summary["encs_pe"]
        
        comp_df = pd.DataFrame([
            {
                "Method": "ENCS (Table 3)",
                "Accuracy (%)": f"{encs_res['accuracy']:.2f}",
                "Precision (%)": f"{encs_res['precision']:.2f}",
                "Recall (%)": f"{encs_res['recall']:.2f}",
                "F1-Score (%)": f"{encs_res['f1']:.2f}",
                "AUC (%)": f"{encs_res['auc']:.2f}"
            },
            {
                "Method": "ENCS + PE (Table 4)",
                "Accuracy (%)": f"{encs_pe_res['accuracy']:.2f}",
                "Precision (%)": f"{encs_pe_res['precision']:.2f}",
                "Recall (%)": f"{encs_pe_res['recall']:.2f}",
                "F1-Score (%)": f"{encs_pe_res['f1']:.2f}",
                "AUC (%)": f"{encs_pe_res['auc']:.2f}"
            }
        ])
        print(comp_df.to_string(index=False))
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test single subject SOZ localization pipeline")
    parser.add_argument("--subject", type=str, default="sub-HUP142", help="Subject ID (default: sub-HUP142)")
    parser.add_argument("--task", type=str, default="interictal", help="Task name (default: interictal)")
    parser.add_argument("--run", type=str, default="01", help="Run ID (default: 01)")
    parser.add_argument("--mode", type=str, choices=["encs", "encs_pe", "both"], default="both", help="Feature mode")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")

    args = parser.parse_args()
    run_single_subject_pipeline(
        subject_id=args.subject,
        task=args.task,
        run=args.run,
        mode=args.mode,
        generate_plot=not args.no_plot
    )
