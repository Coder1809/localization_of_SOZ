"""
main.py - Main CLI Entrypoint for Epilepsy SOZ Localization Reimplementation
Paper: "Epilepsy lesion localization method based on brain function network"
Authors: Chunying Fang et al., Frontiers in Human Neuroscience, 2024.
"""

import sys
import argparse
from pathlib import Path

# Add project directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_DATASET_DIR, PAPER_COHORT
from scripts.inspect_dataset import main as inspect_main
from scripts.test_single_subject import run_single_subject_pipeline
from scripts.run_all_subjects import run_cohort_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Epilepsy Lesion Localization via Functional Brain Networks and Persistent Homology (Fang et al., 2024)"
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Inspect dataset integrity, channel counts, and cohort metadata"
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="Subject ID to evaluate (e.g. sub-HUP142, sub-HUP151, or 'all')"
    )
    parser.add_argument(
        "--task", type=str, default="interictal", choices=["interictal", "ictal"],
        help="SEEG task recording type (default: interictal)"
    )
    parser.add_argument(
        "--run", type=str, default="01",
        help="Recording run index (default: 01)"
    )
    parser.add_argument(
        "--mode", type=str, default="both", choices=["encs", "encs_pe", "both"],
        help="Feature mode: 'encs' (Table 3), 'encs_pe' (Table 4), or 'both'"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run full cohort evaluation across all 22 subjects from Table 1"
    )
    parser.add_argument(
        "--max-subjects", type=int, default=None,
        help="Limit number of subjects when running cohort evaluation"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable generation of evaluation plots"
    )

    args = parser.parse_args()

    if args.inspect:
        inspect_main()
        return

    if args.all or (args.subject and args.subject.lower() == "all"):
        run_cohort_evaluation(
            task=args.task,
            max_subjects=args.max_subjects,
            save_csv=True
        )
        return

    # Default to single subject (sub-HUP142 if not specified)
    target_subject = args.subject or "sub-HUP142"
    run_single_subject_pipeline(
        subject_id=target_subject,
        task=args.task,
        run=args.run,
        mode=args.mode,
        generate_plot=not args.no_plot
    )


if __name__ == "__main__":
    main()
