# Epilepsy Lesion Localization Based on Brain Function Network

Official Python Reimplementation of the research paper:

> **"Epilepsy lesion localization method based on brain function network"**  
> Chunying Fang, Xingyu Li, Meng Na, Wenhao Jiang, Yuankun He, Aowei Wei, Jie Huang, Ming Zhou  
> _Frontiers in Human Neuroscience_, Volume 18, Article 1431153 (2024).  
> DOI: [10.3389/fnhum.2024.1431153](https://doi.org/10.3389/fnhum.2024.1431153) | PMCID: [PMC11266299](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11266299/)

---

## 1. Methodology Overview

The algorithm integrates dynamic functional brain network connectivity with topological data analysis (TDA) to localize Seizure Onset Zones (SOZ) from Stereo-EEG (SEEG) recordings:

```
                            ┌────────────────────────────────────────┐
                            │          Raw SEEG Recordings           │
                            └───────────────────┬────────────────────┘
                                                │
                              Preprocessing & Frequency Filtering
                              (Bad channels removed, 200 Hz, CAR,
                               60 Hz notch, 0.5–80 Hz bandpass,
                               Gamma band: 30–60 Hz)
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        ▼                                               ▼
           [Branch 1: Network Dynamics]                   [Branch 2: Nonlinear Topology]
        WPLI Functional Connectivity Matrix               Takens Delay Embedding (τ=28, m=4)
                        │                                               │
          Proportional Threshold (50%)                      Vietoris-Rips Complex Filtration
                        │                                               │
            Binarized Brain Network                     Persistent Homology (H0, H1, H2)
                        │                                               │
            Epileptic Network Connectivity Strength       Persistence Entropy (PE)
                        │                                               │
                        └───────────────────────┬───────────────────────┘
                                                │
                                          Feature Fusion
                                                │
                                               PCA
                                                │
                                  SVM (RBF Kernel, C = 0.5)
                                   (5-Fold Stratified CV)
                                                │
                                                ▼
                                     SOZ vs Non-SOZ Prediction
```

### Core Parameters from the Paper:

- **Dataset:** HUP-iEEG / OpenNeuro `ds004100` (Version 1.1.3).
- **Sampling Rate:** Downsampled to $200\text{ Hz}$.
- **Filtering:** $60\text{ Hz}$ trap / notch filter, $0.5\text{–}80\text{ Hz}$ 4th-order zero-phase Butterworth bandpass.
- **Referencing:** Common Average Reference (CAR) across good SEEG electrodes.
- **Target Frequency Band:** Gamma ($30\text{–}60\text{ Hz}$).
- **Slicing & Augmentation:** Window length $30\text{ s}$, shift $25\text{ s}$ ($5\text{ s}$ overlap). 5 slices for SOZ channels (5x augmentation), 1 slice for non-SOZ channels.
- **Brain Network Connectivity:** Weighted Phase Lag Index (WPLI).
- **Network Binarization:** $50\%$ Proportional Thresholding.
- **Network Feature:** Epileptic Network Connectivity Strength (ENCS = node degree in binarized graph).
- **Phase-Space Delay Embedding:** Delay $\tau = 28$ (AMI-selected), Embedding dimension $m = 4$ (FNN-selected).
- **Persistent Homology:** Vietoris-Rips simplicial complex filtration ($H_0, H_1, H_2$).
- **Topological Feature:** Persistence Entropy (PE).
- **Classifier:** Support Vector Machine (SVM) with RBF kernel and penalty $C = 0.5$.
- **Validation:** 5-Fold Stratified Cross-Validation (individualized per subject).

---

## 2. Project Directory Structure

```
epilepsy-soz-reimplementation/
├── README.md                             # This documentation
├── requirements.txt                      # Python dependencies
├── config.py                             # Centralized paper hyperparameters
├── main.py                               # Unified CLI entrypoint
├── src/
│   ├── __init__.py
│   ├── dataset.py                        # BIDS dataset loader, bad channel rejection & label parser
│   ├── preprocessing.py                  # Downsampling, notch, bandpass, CAR, window segmentation
│   ├── wpli.py                           # Hilbert transform, cross-spectrum, Weighted Phase Lag Index
│   ├── brain_network.py                  # 50% Proportional thresholding, ENCS feature extraction
│   ├── topology.py                       # Takens embedding, Vietoris-Rips complexes, Persistence Entropy
│   ├── features.py                       # Feature fusion, 5x focal balancing, matrix formatting
│   ├── classifier.py                     # StandardScaler, PCA, SVM (RBF, C=0.5)
│   └── evaluation.py                     # 5-Fold Stratified CV, AC, Precision, Recall, F1, AUC, plots
├── scripts/
│   ├── inspect_dataset.py                # Dataset inspection and channel statistics
│   ├── test_single_subject.py            # Single-subject end-to-end evaluation (sub-HUP142)
│   └── run_all_subjects.py               # Full cohort evaluation reproducing Tables 3 & 4
└── results/                              # Output CSV metrics, reports, and visualization figures
```

---

## 3. Installation & Setup

### Prerequisites

- Python 3.8+
- Local copy of the HUP-iEEG dataset `ds004100-1.1.3/` placed in the parent workspace directory.

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Usage Guide

### 1. Inspect the Dataset

To inspect all discovered SEEG subjects, check channel counts, bad channels, and ground-truth SOZ distribution:

```bash
python main.py --inspect
```

### 2. Run Single-Subject Pipeline (Default: sub-HUP142)

To run the full end-to-end pipeline (preprocessing, WPLI, TDA, 5x focal balancing, PCA, SVM, 5-fold CV) on `sub-HUP142`:

```bash
python main.py --subject sub-HUP142
```

To evaluate a different subject (e.g., `sub-HUP151`):

```bash
python main.py --subject sub-HUP151
```

### 3. Run Full Cohort Evaluation (Reproducing Tables 3 & 4)

To execute cross-validation across all 22 patients from the paper cohort and export `results/table3_encs_localization.csv` and `results/table4_encs_pe_localization.csv`:

```bash
python main.py --all
```

---

## 5. Output Structure

Running the pipeline organizes results into **dedicated subfolders for each patient** inside `results/`:

```
results/
├── cohort_comparison_master.csv               # Master comparison table (Table 3 vs Table 4 deltas)
├── table3_encs_localization.csv               # Full cohort performance using ENCS
├── table4_encs_pe_localization.csv            # Full cohort performance using ENCS + PE
│
├── sub-HUP117/
│   ├── sub-HUP117_encs_pe_clinical_report.json
│   ├── sub-HUP117_encs_pe_electrode_predictions.csv
│   └── sub-HUP117_encs_pe_localization_evaluation.png
├── sub-HUP130/
│   ├── ...
├── sub-HUP142/
│   ├── sub-HUP142_encs_pe_clinical_report.json
│   ├── sub-HUP142_encs_pe_electrode_predictions.csv
│   └── sub-HUP142_encs_pe_localization_evaluation.png
└── ... (all 22 patients)
```

Each patient folder contains:
1. **`<subject_id>_encs_pe_clinical_report.json`**: Clinical report with SOZ contacts, EZ resection contacts, surgical target, therapy, outcome, and seizure onset timing.
2. **`<subject_id>_encs_pe_electrode_predictions.csv`**: Electrode-level prediction mapping for all contacts (`TP`, `TN`, `FP`, `FN`) with confidence scores.
3. **`<subject_id>_encs_pe_localization_evaluation.png`**: High-resolution ROC Curve and Confusion Matrix plots.

---

## 6. Citation

```bibtex
@article{fang2024epilepsy,
  title={Epilepsy lesion localization method based on brain function network},
  author={Fang, Chunying and Li, Xingyu and Na, Meng and Jiang, Wenhao and He, Yuankun and Wei, Aowei and Huang, Jie and Zhou, Ming},
  journal={Frontiers in Human Neuroscience},
  volume={18},
  pages={1431153},
  year={2024},
  publisher={Frontiers},
  doi={10.3389/fnhum.2024.1431153}
}
```

<!-- How to run the code from terminal:

cd "epilepsy-soz-reimplementation"

# 1. Inspect dataset structure & metadata
python3 main.py --inspect

# 2. Run single subject pipeline (sub-HUP142)
python3 main.py --subject sub-HUP142

# 3. Run full cohort evaluation across all 22 subjects
python3 main.py --all

python3 main.py --subject sub-HUP142 --task ictal
python3 main.py --all --task ictal

 -->
