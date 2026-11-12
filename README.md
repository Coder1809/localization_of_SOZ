# Localization of Seizure Onset Zone (SOZ) Based on Brain Function Network

> **A Deep Sequential Learning and Topological Data Analysis Framework for Drug-Resistant Epilepsy**  
> **National Institute of Technology Raipur — Department of Information Technology**  
> **Minor Project (B.Tech Information Technology)**

**Submitted By:**
* **B. Sasank Reddy** (Roll No: 23118015)
* **Sameer Sevlani** (Roll No: 23118086)

**Project Supervisor:**
* **Dr. Rekh Ram Janghel**, Associate Professor, Department of Information Technology, NIT Raipur

---

## 📌 Executive Summary

Drug-resistant epilepsy (DRE) affects 20% to 30% of epileptic patients worldwide, rendering anti-seizure medication ineffective and establishing surgical resection or laser ablation of the epileptogenic zone as the only curative option. The clinical success of surgery hinges strictly on the millimeter-level localization of the **Seizure Onset Zone (SOZ)** from multi-channel intracranial Stereotactic EEG (SEEG) recordings.

This project delivers:
1. **Faithful End-to-End Reimplementation**: Full empirical replication of the state-of-the-art dual-branch pipeline of **Fang et al. (Frontiers in Human Neuroscience, 2024)** across all 22 subjects (>2,500 electrode contacts) from the OpenNeuro `ds004100` clinical SEEG benchmark.
2. **Diagnosis of the Static Performance Ceiling**: Empirical demonstration that static, memoryless SVM classifiers hit an accuracy ceiling (~72.21%) on continuous clinical SEEG due to temporal coarseness (30-second windows) and inability to track dynamic seizure propagation.
3. **Proposed Bidirectional Recurrent Sequential Framework (Bi-GRU)**: Engineering a novel deep sequential architecture integrating event-aligned micro-windows (10s window, 5s shift), gamma-band wPLI phase-synchrony network node degree (ENCS), multi-dimensional Persistent Homology entropy (TDA), 2-layer Bidirectional GRU, Global Temporal Mean Pooling, and dynamic class-weighted focal loss.
4. **Superior Diagnostic Performance**: Driving cohort-wide localization accuracy to **98.66%**, precision to **98.09%**, recall to **99.36%**, F1-score to **98.71%**, and ROC-AUC to **98.91%**, significantly outperforming both the midsem baseline and published benchmarks.

---

## 🔬 System Architecture & Methodology

```
                                 ┌──────────────────────────────────────┐
                                 │     Multi-Channel SEEG Signals       │
                                 │       (22 Subjects, OpenNeuro)       │
                                 └──────────────────┬───────────────────┘
                                                    │
                                     Preprocessing & Micro-Windowing
                                     • Downsampling to 200 Hz
                                     • CAR Referencing & [0.5, 80] Hz Bandpass
                                     • 6 Micro-Windows (10s duration, 5s shift)
                                                    │
                           ┌────────────────────────┴────────────────────────┐
                           ▼                                                 ▼
             [Stream A: Functional Network]                    [Stream B: Topological Homology]
             Gamma (30–60 Hz) wPLI Matrix                      Takens Attractor Embedding (m=4, τ=28)
                           │                                                 │
             50% Proportional Binarization                     Vietoris-Rips Filtration (Ripser)
                           │                                                 │
             Epileptic Network Node Degree (ENCS)              Persistence Entropy (H0, H1, H2)
                           │                                                 │
                           └────────────────────────┬────────────────────────┘
                                                    │
                                        Multimodal Feature Fusion
                                    (5D Feature Vector per Contact)
                                                    │
                                       Bidirectional GRU Core
                                      (2-Layer Bi-GRU, d_hid=16)
                                 Tracks Forward & Backward Dynamics
                                                    │
                                      Global Temporal Mean Pooling
                                (Summarizes T=6 Micro-Windows into hc)
                                                    │
                                        MLP Classification Head
                                 + Dynamic Class-Weighted Focal Loss
                                                    │
                                                    ▼
                                 Patient-Specific SOZ Localization &
                                  Candidate Surgical Resection Set
```

### Core Methodological Pillars

* **Stream A — Functional Connectivity Network (wPLI)**: We extract instantaneous phase lag using the Hilbert transform in the gamma band ($30\text{–}60\text{ Hz}$). Weighted Phase Lag Index (wPLI) strictly prevents volume conduction artifacts (zero-lag synchronization). A $50\%$ proportional thresholding extracts dominant synchronization backbones, yielding the **ENCS node degree** that isolates hyper-synchronized epileptic hub channels.
* **Stream B — Topological Data Analysis (TDA)**: Clinical SEEG dynamics are highly nonlinear. We reconstruct the multi-dimensional phase-space attractor via Takens delay embedding ($m = 4, \tau = 28$ chosen via FNN and AMI). Vietoris-Rips simplicial complex filtration captures multi-scale connected components ($H_0$), loops ($H_1$), and voids ($H_2$). Multi-scale **Persistence Entropy (PE)** quantifies the geometric collapse of chaotic complexity during seizure onset.
* **Bidirectional GRU Sequence Modeling**: Rather than evaluating disconnected snapshot windows, a 2-layer Bidirectional Gated Recurrent Unit (Bi-GRU) evaluates each contact's feature sequence across time, capturing both the forward wave of seizure propagation and backward pre-ictal initiation.
* **Global Temporal Mean Pooling**: Averages hidden states across all $T = 6$ temporal steps into a fixed-length embedding $\mathbf{h}_c$, retaining global sequence context.
* **Dynamic Class-Weighted Focal Loss**: Solves extreme electrode class imbalance (only $\sim 7\%$ of implanted contacts belong to the true SOZ; $93\%$ are non-SOZ) by dynamically weighting gradients ($w_{\text{pos}} = N_{\text{neg}} / N_{\text{pos}}$), preventing missed seizure foci.

---

## 📊 Experimental Results & Benchmark Comparison

### 1. Midsem Reimplementation Baseline vs. Published Paper (Fang et al., 2024)

| Model / Configuration | Features Used | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Published Paper (Fang et al., 2024)** | ENCS + PE (SVM) | 94.40% | 98.48% | 89.74% | 93.40% | 96.97% |
| **Our Re-implementation (Baseline A)** | ENCS Alone (SVM) | **59.69%** | **60.72%** | **59.44%** | **59.32%** | **62.11%** |
| **Our Re-implementation (Baseline B)** | ENCS + PE (SVM) | **72.21%** | **75.67%** | **67.13%** | **70.67%** | **78.32%** |

*TDA Integration Gain: Incorporating Persistence Entropy produced a **+12.52%** boost in Accuracy, **+14.95%** boost in Precision, and **+16.21%** boost in ROC-AUC across the 22-subject cohort, validating the core hypothesis of topological seizure localization.*

### 2. Master Model Ablation Study (Midsem vs. Endsem Proposed System)

| Model Configuration | Window Segmentation | Features | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1. Paper Benchmark (Fang et al., 2024)** | 30s / 25s shift | ENCS + PE | RBF-SVM ($C=0.5$) | 94.40% | 98.48% | 89.74% | 93.40% | 96.97% |
| **2. ENCS $\to$ SVM (Midsem Baseline)** | 30s / 25s shift | ENCS | RBF-SVM ($C=0.5$) | 67.00% | 66.86% | 73.08% | 69.02% | 69.03% |
| **3. ENCS + PE $\to$ SVM (Midsem Baseline)** | 30s / 25s shift | ENCS + PE | RBF-SVM ($C=0.5$) | 77.43% | 79.51% | 74.52% | 76.89% | 83.81% |
| **4. ENCS $\to$ Bi-GRU** | 30s / 25s shift | ENCS | Bidirectional GRU | 96.36% | 93.50% | 100.00% | 96.59% | 95.74% |
| **5. ENCS + PE $\to$ Bi-GRU** | 30s / 25s shift | ENCS + PE | Bidirectional GRU | 99.14% | 98.51% | 99.80% | 99.15% | 99.10% |
| **6. Proposed Seizure-Centric Framework** | **10s / 5s shift** | **ENCS + PE** | **Bidirectional GRU** | **98.66%** | **98.09%** | **99.36%** | **98.71%** | **98.91%** |

---

## 📂 Repository Structure

```
localization_of_SOZ/
├── README.md                             # Comprehensive project documentation
├── .gitignore                            # Excludes datasets, bytecode, and temp files
├── Report.pdf                            # 34-page Minor Project Report (NIT Raipur)
├── presentation.pptx                     # 19-slide Final Presentation Deck
└── epilepsy-soz-reimplementation/         # Modular Python codebase
    ├── config.py                         # Centralized hyperparameter configuration
    ├── main.py                           # CLI entrypoint for inspection, single patient & cohort
    ├── requirements.txt                  # Python dependencies
    ├── src/
    │   ├── __init__.py
    │   ├── dataset.py                    # BIDS dataset loader, bad channel rejection, labels
    │   ├── preprocessing.py              # CAR, downsampling, notch, bandpass, micro-windows
    │   ├── wpli.py                       # Hilbert transform, cross-spectrum, gamma wPLI
    │   ├── brain_network.py              # 50% proportional binarization, ENCS node degree
    │   ├── topology.py                   # Takens delay embedding, Vietoris-Rips, PE entropy
    │   ├── features.py                   # Multimodal feature fusion, class balancing
    │   ├── classifier.py                 # SVM pipeline (PCA+RBF) & PyTorch Bi-GRU model
    │   └── evaluation.py                 # Stratified 5-fold CV, metrics, ROC/confusion plots
    ├── scripts/
    │   ├── inspect_dataset.py            # Dataset validator and channel statistics
    │   ├── test_single_subject.py        # Single-subject end-to-end evaluation
    │   └── run_all_subjects.py           # Cohort-wide cross-validation (Tables 3 & 4)
    └── results/                          # Clinical reports, prediction CSVs, ROC curves
```

---

## 🚀 Installation & Reproduction

### 1. Prerequisites
* Python 3.8 to 3.12
* PyTorch 2.0+
* OpenNeuro `ds004100` (Version 1.1.3) SEEG Dataset placed in the parent directory:
  ```bash
  # Dataset should reside at: ../ds004100-1.1.3/
  ```

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/Coder1809/localization_of_SOZ.git
cd localization_of_SOZ/epilepsy-soz-reimplementation

# Install dependencies
pip install -r requirements.txt
```

### 3. Execution Commands

#### A. Inspect the Clinical Dataset
Inspect discovered SEEG channels, electrode montages, bad channel rejection, and ground-truth SOZ contacts:
```bash
python main.py --inspect
```

#### B. Run Single-Subject Pipeline (Default: sub-HUP142)
Executes preprocessing, gamma wPLI, Takens embedding, persistence entropy, 5-fold cross-validation, and exports clinical report and ROC curve:
```bash
python main.py --subject sub-HUP142
```
To test other cohort patients (e.g., `sub-HUP151`):
```bash
python main.py --subject sub-HUP151
```

#### C. Run Full 22-Patient Cohort Evaluation
Reproduces cohort results across all 22 subjects from Table 1 and exports master comparison CSV tables:
```bash
python main.py --all
```

---

## 📑 Core Project Deliverables

* 📄 **[Report.pdf](Report.pdf)**: Complete 34-Page Minor Project Report containing formal mathematical formulations (Cauchy principal value Hilbert transforms, cross-spectral density, proportional adjacency binarization, Takens delay embedding, Vietoris-Rips complexes, persistence diagrams, Bi-GRU recurrence, dynamic focal loss), clinical demographics, windowing benchmarks, and ablation studies.
* 📊 **[presentation.pptx](presentation.pptx)**: 19-Slide Final Defense Presentation Deck featuring the dual-stream architecture pipeline, 4 core methodological pillars, step-by-step execution algorithm, textbook evaluation metric cards, baseline comparisons, and master ablation benchmarks.

---

## 📜 References & Citation

1. **Fang, C., Li, X., Na, M., Jiang, W., He, Y., Wei, A., Huang, J., & Zhou, M. (2024).**  
   *Epilepsy lesion localization method based on brain function network.*  
   *Frontiers in Human Neuroscience*, 18, 1431153.  
   DOI: [10.3389/fnhum.2024.1431153](https://doi.org/10.3389/fnhum.2024.1431153)

2. **OpenNeuro ds004100 Dataset:**  
   *Human Intracranial SEEG Recordings for Epilepsy Surgery Localization.*  
   OpenNeuro. DOI: [10.18112/openneuro.ds004100.v1.1.3](https://openneuro.org/datasets/ds004100/versions/1.1.3)

---

## ⚖️ License & Academic Integrity
This repository is submitted as the official Minor Project submission for the Degree of Bachelor of Technology in Information Technology at the **National Institute of Technology Raipur**. All code, figures, and deliverables are provided for academic research and evaluation purposes.
