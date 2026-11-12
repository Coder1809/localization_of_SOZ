"""
scripts/inspect_dataset.py - Dataset Inspection and Metadata Validation Utility
Discovers all SEEG subjects in ds004100, parses channels.tsv, extracts SOZ counts,
and displays a structured overview of the dataset.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_DATASET_DIR, PAPER_COHORT
from src.dataset import BIDSDataSetLoader


def main():
    print("=" * 80)
    print(" HUP-iEEG (ds004100) DATASET INSPECTION & INTEGRITY CHECK")
    print("=" * 80)
    print(f"Dataset path: {DEFAULT_DATASET_DIR}\n")

    loader = BIDSDataSetLoader(DEFAULT_DATASET_DIR)
    seeg_subjects = loader.get_seeg_subjects(filter_paper_cohort=False)
    paper_subjects = loader.get_seeg_subjects(filter_paper_cohort=True)

    print(f"Total SEEG subjects discovered in dataset: {len(seeg_subjects)}")
    print(f"Paper cohort subjects (Table 1) available on disk: {len(paper_subjects)} / 22\n")

    summary_rows = []

    for sub_id in paper_subjects:
        recordings = loader.find_recordings(sub_id, task="interictal")
        if not recordings:
            recordings = loader.find_recordings(sub_id, task=None)

        if not recordings:
            summary_rows.append({
                "Subject": sub_id,
                "In_Paper_Cohort": "Yes",
                "Task": "N/A",
                "Total_Channels": "N/A",
                "Good_SEEG": "N/A",
                "Bad_Ch": "N/A",
                "SOZ_Ch": "N/A",
                "Non_SOZ_Ch": "N/A",
                "Balanced_Epochs (5x)": "N/A"
            })
            continue

        rec_info = recordings[0]
        rec_data = loader.load_recording(rec_info)

        n_good = rec_data["n_channels"]
        n_bad = len(rec_data["bad_channel_names"])
        n_soz = rec_data["n_soz"]
        n_non_soz = rec_data["n_non_soz"]
        balanced_epochs = (n_soz * 5) + n_non_soz

        summary_rows.append({
            "Subject": sub_id,
            "In_Paper_Cohort": "Yes",
            "Task": rec_data["task"],
            "Total_Channels": n_good + n_bad,
            "Good_SEEG": n_good,
            "Bad_Ch": n_bad,
            "SOZ_Ch": n_soz,
            "Non_SOZ_Ch": n_non_soz,
            "Balanced_Epochs (5x)": balanced_epochs
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("Example: sub-HUP142 Channel Breakdown:")
    hup142_rec = loader.find_recordings("sub-HUP142", task="interictal")
    if hup142_rec:
        d142 = loader.load_recording(hup142_rec[0])
        print(f"  Total channels in EDF: {d142['n_channels'] + len(d142['bad_channel_names'])}")
        print(f"  Good SEEG channels: {d142['n_channels']}")
        print(f"  Bad channels removed: {len(d142['bad_channel_names'])} {d142['bad_channel_names']}")
        print(f"  Ground-truth SOZ channels ({d142['n_soz']}): {d142['soz_channel_names']}")
        print(f"  Non-SOZ channels: {d142['n_non_soz']}")
        print(f"  5x Balanced Epoch Count: 5 * {d142['n_soz']} + {d142['n_non_soz']} = {(d142['n_soz'] * 5) + d142['n_non_soz']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
