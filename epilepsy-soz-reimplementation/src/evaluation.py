"""
src/evaluation.py - 5-Fold Stratified Cross-Validation and Metrics Computation
Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, generates classification reports,
and exports results matching Tables 3 & 4 of the paper.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
import matplotlib.pyplot as plt

from config import CV_FOLDS, RANDOM_STATE, SVM_C, RESULTS_DIR
from src.classifier import SOZClassifierPipeline


def evaluate_5fold_cv(
    X: np.ndarray,
    y: np.ndarray,
    c: float = SVM_C,
    use_pca: bool = True,
    n_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE
) -> Dict[str, Any]:
    """
    Performs individualized 5-fold Stratified Cross-Validation for a subject.
    Strictly fits scalers and PCA inside each train fold to avoid data leakage.
    
    Returns:
        dict containing:
            - 'accuracy': float (%)
            - 'precision': float (%)
            - 'recall': float (%)
            - 'f1': float (%)
            - 'auc': float (%)
            - 'y_true_all': np.ndarray
            - 'y_pred_all': np.ndarray
            - 'y_scores_all': np.ndarray
            - 'confusion_matrix': np.ndarray [[TN, FP], [FN, TP]]
            - 'fold_metrics': List[Dict]
    """
    # Check class presence
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(f"Cross-validation requires both SOZ and non-SOZ classes, but found: {unique_classes}")

    # Determine valid number of splits based on smallest class count
    min_class_count = min(np.sum(y == 0), np.sum(y == 1))
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < 2:
        actual_folds = 2

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=random_state)

    y_true_all = []
    y_pred_all = []
    y_scores_all = []
    fold_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Fit individualized pipeline on train fold
        pipeline = SOZClassifierPipeline(
            c=c,
            use_pca=use_pca,
            random_state=random_state + fold_idx
        )
        pipeline.fit(X_train, y_train)

        # Predict on test fold with calibrated threshold learned from train fold (zero data leakage)
        try:
            tr_scores = pipeline.decision_function(X_train)
            fpr_tr, tpr_tr, thresh_tr = roc_curve(y_train, tr_scores)
            optimal_idx = np.argmax(tpr_tr - fpr_tr) if len(thresh_tr) > 0 else 0
            best_thresh = thresh_tr[optimal_idx] if len(thresh_tr) > 0 else 0.0

            te_scores = pipeline.decision_function(X_test)
            y_pred = (te_scores >= best_thresh).astype(int)
            y_score = te_scores
        except Exception:
            y_pred = pipeline.predict(X_test)
            try:
                y_score = pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                y_score = y_pred.astype(float)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_scores_all.extend(y_score)

        # Per fold metrics
        fold_ac = accuracy_score(y_test, y_pred) * 100.0
        fold_p = precision_score(y_test, y_pred, zero_division=0) * 100.0
        fold_r = recall_score(y_test, y_pred, zero_division=0) * 100.0
        fold_f1 = f1_score(y_test, y_pred, zero_division=0) * 100.0
        try:
            fold_auc = roc_auc_score(y_test, y_score) * 100.0
        except ValueError:
            fold_auc = 50.0

        fold_records.append({
            "fold": fold_idx + 1,
            "accuracy": fold_ac,
            "precision": fold_p,
            "recall": fold_r,
            "f1": fold_f1,
            "auc": fold_auc
        })

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
    y_scores_arr = np.array(y_scores_all)

    # Overall cross-validated metrics
    overall_ac = accuracy_score(y_true_arr, y_pred_arr) * 100.0
    overall_p = precision_score(y_true_arr, y_pred_arr, zero_division=0) * 100.0
    overall_r = recall_score(y_true_arr, y_pred_arr, zero_division=0) * 100.0
    overall_f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0) * 100.0

    try:
        overall_auc = roc_auc_score(y_true_arr, y_scores_arr) * 100.0
    except ValueError:
        overall_auc = 50.0

    cm = confusion_matrix(y_true_arr, y_pred_arr)

    return {
        "accuracy": overall_ac,
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "auc": overall_auc,
        "y_true_all": y_true_arr,
        "y_pred_all": y_pred_arr,
        "y_scores_all": y_scores_arr,
        "confusion_matrix": cm,
        "fold_metrics": fold_records
    }


def generate_channel_classification_report(
    recording_dict: Dict[str, Any],
    dataset_features: Dict[str, Any],
    eval_results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Maps cross-validation predictions back to physical electrodes.
    Classifies each electrode as:
    - True Positive (Correct SOZ)
    - True Negative (Correct Non-SOZ)
    - False Positive (Misclassified Non-SOZ)
    - False Negative (Missed SOZ)
    """
    channel_names = recording_dict["channel_names"]
    labels = recording_dict["labels"]
    status_descs = recording_dict.get("status_descriptions", ["n/a"] * len(labels))

    sample_channels = dataset_features["sample_channel_names"]
    y_pred = eval_results["y_pred_all"]
    y_scores = eval_results["y_scores_all"]

    # For SOZ channels with 5 slices, average the predicted probabilities/scores
    ch_predictions: Dict[str, List[int]] = {ch: [] for ch in channel_names}
    ch_scores: Dict[str, List[float]] = {ch: [] for ch in channel_names}

    for sample_name, pred_val, score_val in zip(sample_channels, y_pred, y_scores):
        base_name = sample_name.split("_w")[0]
        if base_name in ch_predictions:
            ch_predictions[base_name].append(pred_val)
            ch_scores[base_name].append(score_val)

    rows = []
    for i, ch in enumerate(channel_names):
        true_label = labels[i]
        preds = ch_predictions[ch]
        scores = ch_scores[ch]
        
        # Majority vote across windows for electrode level classification
        pred_label = int(np.round(np.mean(preds))) if preds else 0
        avg_score = float(np.mean(scores)) if scores else 0.0

        if true_label == 1 and pred_label == 1:
            status = "TP (Correct SOZ)"
        elif true_label == 0 and pred_label == 0:
            status = "TN (Correct Non-SOZ)"
        elif true_label == 0 and pred_label == 1:
            status = "FP (False Alarm)"
        else:
            status = "FN (Missed SOZ)"

        rows.append({
            "Channel": ch,
            "Ground_Truth": "SOZ" if true_label == 1 else "Non-SOZ",
            "Predicted": "SOZ" if pred_label == 1 else "Non-SOZ",
            "Score": round(avg_score, 4),
            "Status": status,
            "BIDS_Status_Description": status_descs[i]
        })

    return pd.DataFrame(rows)


def plot_subject_results(
    subject_id: str,
    eval_results: Dict[str, Any],
    channel_df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> str:
    """
    Generates ROC curve, Confusion Matrix, and localization summary plots.
    """
    out_dir = output_dir or RESULTS_DIR
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. ROC Curve
    y_true = eval_results["y_true_all"]
    y_scores = eval_results["y_scores_all"]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_val = eval_results["auc"]

    axes[0].plot(fpr, tpr, color="#0C4DED", lw=2.5, label=f"ROC curve (AUC = {auc_val:.2f}%)")
    axes[0].plot([0, 1], [0, 1], color="#999999", lw=1.5, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[0].set_title(f"ROC Curve - {subject_id}", fontsize=13, fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Confusion Matrix
    cm = eval_results["confusion_matrix"]
    im = axes[1].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1].set_title(f"Confusion Matrix - {subject_id}", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=axes[1])
    tick_marks = [0, 1]
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(["Non-SOZ", "SOZ"])
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(["Non-SOZ", "SOZ"])
    axes[1].set_ylabel("Actual Label", fontsize=11)
    axes[1].set_xlabel("Predicted Label", fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold"
            )

    plt.tight_layout()
    plot_path = f"{out_dir}/{subject_id}_localization_evaluation.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return plot_path


def generate_clinical_localization_summary(
    recording_dict: Dict[str, Any],
    ch_report_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generates a structured clinical report containing:
    1. Seizure Onset Zone (SOZ) localization & electrode breakdown
    2. Epileptogenic Zone (EZ) / Resection zone analysis
    3. Seizure onset timing & duration events
    """
    subject_id = recording_dict["subject_id"]
    task = recording_dict["task"]
    meta = recording_dict.get("clinical_metadata", {})
    ev_info = recording_dict.get("events_info", {})

    # SOZ metrics
    gt_soz = recording_dict["soz_channel_names"]
    pred_soz = ch_report_df[ch_report_df["Predicted"] == "SOZ"]["Channel"].tolist()
    tp_contacts = ch_report_df[ch_report_df["Status"].str.startswith("TP")]["Channel"].tolist()
    fp_contacts = ch_report_df[ch_report_df["Status"].str.startswith("FP")]["Channel"].tolist()
    fn_contacts = ch_report_df[ch_report_df["Status"].str.startswith("FN")]["Channel"].tolist()
    tn_contacts = ch_report_df[ch_report_df["Status"].str.startswith("TN")]["Channel"].tolist()

    # Electrode shaft breakdown (e.g. LDA, LDB, LDC)
    shaft_groups: Dict[str, Dict[str, Any]] = {}
    for _, row in ch_report_df.iterrows():
        ch = str(row["Channel"])
        # Extract prefix letters (e.g. 'LDA' from 'LDA1')
        shaft = "".join([c for c in ch if c.isalpha()])
        if shaft not in shaft_groups:
            shaft_groups[shaft] = {"total": 0, "gt_soz": [], "pred_soz": [], "resected": []}
        shaft_groups[shaft]["total"] += 1
        if row["Ground_Truth"] == "SOZ":
            shaft_groups[shaft]["gt_soz"].append(ch)
        if row["Predicted"] == "SOZ":
            shaft_groups[shaft]["pred_soz"].append(ch)
        if "resect" in str(row["BIDS_Status_Description"]).lower():
            shaft_groups[shaft]["resected"].append(ch)

    # Epileptogenic Zone (EZ) / Resection Area
    resected_contacts = recording_dict.get("ez_resect_channel_names", [])
    pred_soz_in_resect = [ch for ch in pred_soz if ch in resected_contacts]
    resect_coverage = (len(pred_soz_in_resect) / max(1, len(pred_soz))) * 100.0 if pred_soz else 0.0

    return {
        "subject_id": subject_id,
        "task": task,
        "filename": recording_dict["filename"],
        "seizure_onset_zone": {
            "ground_truth_count": len(gt_soz),
            "ground_truth_contacts": gt_soz,
            "predicted_count": len(pred_soz),
            "predicted_contacts": pred_soz,
            "true_positives (correct SOZ)": tp_contacts,
            "false_positives (false alarms)": fp_contacts,
            "false_negatives (missed SOZ)": fn_contacts,
            "shaft_breakdown": shaft_groups
        },
        "epileptogenic_zone_and_resection": {
            "anatomical_target": meta.get("target", "N/A"),
            "therapy_type": meta.get("therapy", "N/A"),
            "surgical_outcome": "Seizure-Free (Success)" if str(meta.get("outcome", "")).upper() == "S" else meta.get("outcome", "N/A"),
            "resected_contacts_count": len(resected_contacts),
            "resected_contacts": resected_contacts,
            "predicted_soz_within_resection": pred_soz_in_resect,
            "resection_overlap_percentage": round(resect_coverage, 2)
        },
        "seizure_onset_timing": ev_info
    }


def print_clinical_report(summary: Dict[str, Any]):
    """Prints a formatted clinical summary to terminal."""
    soz = summary["seizure_onset_zone"]
    ez = summary["epileptogenic_zone_and_resection"]
    ev = summary["seizure_onset_timing"]

    print("\n" + "═" * 85)
    print(f" CLINICAL SOZ & EPILEPTOGENIC ZONE (EZ) LOCALIZATION REPORT")
    print("═" * 85)
    print(f"  Patient ID       : {summary['subject_id']}")
    print(f"  Recording File   : {summary['filename']} (Task: {summary['task']})")
    print(f"  Surgical Target  : {ez['anatomical_target']}")
    print(f"  Therapy Administered: {ez['therapy_type']}")
    print(f"  Surgical Outcome : {ez['surgical_outcome']}")

    # 1. Seizure Timing
    if ev.get("has_seizure_events"):
        print(f"\n  [SEIZURE ONSET TIMING EVENTS]")
        print(f"   • Seizure Onset Time   : {ev['seizure_onset_sec']:.2f} s")
        print(f"   • Seizure Offset Time  : {ev['seizure_offset_sec']:.2f} s")
        print(f"   • Seizure Duration     : {ev['seizure_duration_sec']:.2f} s")

    # 2. Seizure Onset Zone
    print(f"\n  [SEIZURE ONSET ZONE (SOZ) LOCALIZATION]")
    print(f"   • Ground-Truth SOZ Contacts ({soz['ground_truth_count']}): {soz['ground_truth_contacts']}")
    print(f"   • Predicted SOZ Contacts    ({soz['predicted_count']}): {soz['predicted_contacts']}")
    print(f"   • Correctly Localized (TP)  ({len(soz['true_positives (correct SOZ)'])}): {soz['true_positives (correct SOZ)']}")
    if soz["false_positives (false alarms)"]:
        print(f"   • False Alarms (FP)         ({len(soz['false_positives (false alarms)'])}): {soz['false_positives (false alarms)']}")
    if soz["false_negatives (missed SOZ)"]:
        print(f"   • Missed SOZ (FN)           ({len(soz['false_negatives (missed SOZ)'])}): {soz['false_negatives (missed SOZ)']}")

    # Shaft breakdown table
    print(f"\n   Electrode Shaft Involvement:")
    for shaft, info in sorted(soz["shaft_breakdown"].items()):
        gt_cnt = len(info['gt_soz'])
        pred_cnt = len(info['pred_soz'])
        resect_cnt = len(info['resected'])
        if gt_cnt > 0 or pred_cnt > 0 or resect_cnt > 0:
            status_tag = "★ SOZ Focus" if gt_cnt > 0 else "Non-SOZ"
            print(f"     - Shaft {shaft:<5}: {gt_cnt}/{info['total']} SOZ contacts | {pred_cnt}/{info['total']} Predicted | {resect_cnt}/{info['total']} Resected  [{status_tag}]")

    # 3. Epileptogenic Zone / Resection
    print(f"\n  [EPILEPTOGENIC ZONE (EZ) & SURGICAL RESECTION]")
    print(f"   • Clinically Resected Contacts ({ez['resected_contacts_count']}): {ez['resected_contacts']}")
    print(f"   • Predicted SOZ in Resected Zone ({len(ez['predicted_soz_within_resection'])}): {ez['predicted_soz_within_resection']}")
    print(f"   • Resection Zone Overlap Coverage: {ez['resection_overlap_percentage']:.1f}%")
    print("═" * 85 + "\n")
