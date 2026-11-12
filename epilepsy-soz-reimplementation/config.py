"""
config.py - Configuration parameters for Epilepsy Lesion Localization Reimplementation
Paper: Fang et al., Frontiers in Human Neuroscience, 2024 (DOI: 10.3389/fnhum.2024.1431153)
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATASET_DIR = WORKSPACE_ROOT / "ds004100-1.1.3"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Preprocessing parameters (Section 2.1)
TARGET_SAMPLING_RATE = 200.0  # Hz, downsampled from original 512 Hz / 1024 Hz
NOTCH_FREQUENCY = 60.0        # Hz, trap / notch filter for line noise
NOTCH_Q = 30.0                # Quality factor for notch filter
BANDPASS_LOW = 0.5            # Hz
BANDPASS_HIGH = 80.0          # Hz
FILTER_ORDER = 4              # 4th-order Butterworth zero-phase filter

# Frequency bands (Hz)
FREQUENCY_BANDS = {
    "delta": (1.0, 3.0),
    "theta": (3.0, 7.0),
    "alpha": (7.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 60.0),
}
TARGET_BAND = "gamma"         # Paper specifically selects Gamma band for SOZ localization

# Windowing and Sample Balancing (Section 2.1)
WINDOW_LENGTH_SEC = 30.0      # Slicing window duration
WINDOW_SHIFT_SEC = 25.0       # Slicing step (5s overlap)
SOZ_OVERSAMPLE_SLICES = 5     # 5 slices for SOZ channels, 1 slice for non-SOZ channels

# Brain Functional Network (Section 2.2 & 2.3)
CONNECTIVITY_METRIC = "wpli"  # Weighted Phase Lag Index
PROPORTIONAL_THRESHOLD = 0.50 # 50% top connections preserved as binary edges

# Topological Data Analysis (Section 2.5)
TAKENS_DELAY = 28             # tau selected via Average Mutual Information (AMI)
TAKENS_DIMENSION = 4          # m selected via False Nearest Neighbors (FNN)
HOMOLOGY_DIMENSIONS = [0, 1, 2] # H0 (components), H1 (loops), H2 (voids)
MAX_TDA_POINTS = 100          # Point cloud subsampling limit for fast Vietoris-Rips filtration

# Classification & Validation (Section 2.6 & Section 3)
PCA_N_COMPONENTS = None       # If None, retains all orthogonal components or set int (e.g. 2)
SVM_KERNEL = "rbf"
SVM_C = 0.5                   # Regularization parameter C = 0.5 as specified in the paper
CV_FOLDS = 5                  # 5-fold stratified cross validation
RANDOM_STATE = 42

# Cohort of 22 SEEG Patients from Table 1
PAPER_COHORT = [
    "sub-HUP117", "sub-HUP130", "sub-HUP133", "sub-HUP138",
    "sub-HUP142", "sub-HUP144", "sub-HUP150", "sub-HUP151",
    "sub-HUP157", "sub-HUP160", "sub-HUP162", "sub-HUP164",
    "sub-HUP166", "sub-HUP171", "sub-HUP172", "sub-HUP173",
    "sub-HUP180", "sub-HUP181", "sub-HUP185", "sub-HUP187",
    "sub-HUP188", "sub-HUP190"
]
