"""
src/dataset.py - BIDS HUP-iEEG Dataset Loader and Parser
Handles automatic subject discovery, EDF reading, bad channel rejection,
and ground-truth SOZ label extraction from BIDS metadata.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import mne

from config import DEFAULT_DATASET_DIR, PAPER_COHORT


class BIDSDataSetLoader:
    """Loader and discovery manager for HUP-iEEG (ds004100) BIDS dataset."""

    def __init__(self, dataset_dir: Optional[Path] = None):
        self.dataset_dir = Path(dataset_dir or DEFAULT_DATASET_DIR)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")
        self.participants_df = self._load_participants()

    def _load_participants(self) -> pd.DataFrame:
        """Loads and cleans participants.tsv."""
        tsv_path = self.dataset_dir / "participants.tsv"
        if not tsv_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(tsv_path, sep="\t")
        obj_cols = {col: df[col].astype(str).str.strip() for col in df.select_dtypes(include="object").columns}
        return df.assign(**obj_cols)

    def get_seeg_subjects(self, filter_paper_cohort: bool = False) -> List[str]:
        """
        Discovers SEEG subjects.
        If filter_paper_cohort is True, returns only subjects from the 22 paper cohort.
        Otherwise, dynamically returns all subjects with SEEG implants.
        """
        if filter_paper_cohort:
            # Check availability of paper cohort subjects on disk
            available = []
            for sub_id in PAPER_COHORT:
                sub_dir = self.dataset_dir / sub_id
                if sub_dir.exists():
                    available.append(sub_id)
            return sorted(available)

        if not self.participants_df.empty and "implant" in self.participants_df.columns:
            seeg_mask = self.participants_df["implant"].str.upper() == "SEEG"
            subjects = self.participants_df.loc[seeg_mask, "participant_id"].tolist()
            # Verify directory exists
            return sorted([s for s in subjects if (self.dataset_dir / s).exists()])

        # Fallback: scan directories for acq-seeg files
        discovered = []
        for sub_path in self.dataset_dir.glob("sub-*"):
            if sub_path.is_dir():
                seeg_files = list(sub_path.glob("**/ieeg/*acq-seeg*ieeg.edf"))
                if seeg_files:
                    discovered.append(sub_path.name)
        return sorted(discovered)

    def find_recordings(
        self,
        subject_id: str,
        task: Optional[str] = "interictal",
        run: Optional[str] = None
    ) -> List[Dict[str, Path]]:
        """
        Finds matched EDF, channels.tsv, and events.tsv files for a subject.
        
        Args:
            subject_id: e.g. 'sub-HUP142'
            task: 'interictal', 'ictal', or None for all
            run: e.g. '01', 'run-01', or None for all
        """
        ieeg_dir = self.dataset_dir / subject_id / "ses-presurgery" / "ieeg"
        if not ieeg_dir.exists():
            ieeg_dir = self.dataset_dir / subject_id / "ieeg"
            if not ieeg_dir.exists():
                return []

        edf_files = list(ieeg_dir.glob("*_ieeg.edf"))
        recordings = []

        for edf_path in sorted(edf_files):
            fname = edf_path.name
            
            # Filter by task if requested
            if task and f"task-{task}" not in fname:
                continue
            
            # Filter by run if requested
            if run:
                run_tag = run if run.startswith("run-") else f"run-{run}"
                if run_tag not in fname:
                    continue

            # Identify matching channels.tsv and events.tsv
            prefix = fname.replace("_ieeg.edf", "")
            channels_path = ieeg_dir / f"{prefix}_channels.tsv"
            events_path = ieeg_dir / f"{prefix}_events.tsv"

            recordings.append({
                "subject_id": subject_id,
                "edf_path": edf_path,
                "channels_path": channels_path if channels_path.exists() else None,
                "events_path": events_path if events_path.exists() else None,
                "filename": fname,
                "task": "interictal" if "task-interictal" in fname else ("ictal" if "task-ictal" in fname else "unknown")
            })

        return recordings

    def load_recording(
        self,
        recording_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Loads the EDF data, parses channels.tsv, filters bad channels,
        and assigns binary ground-truth SOZ labels.
        
        Returns:
            dict containing:
                - subject_id (str)
                - task (str)
                - sampling_rate (float)
                - channel_names (List[str])
                - raw_data (np.ndarray of shape [n_good_channels, n_samples])
                - labels (np.ndarray of shape [n_good_channels], 1=SOZ, 0=Non-SOZ)
                - soz_channel_names (List[str])
                - non_soz_channel_names (List[str])
                - bad_channel_names (List[str])
        """
        edf_path = recording_info["edf_path"]
        channels_path = recording_info.get("channels_path")

        # 1. Read EDF with MNE
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        orig_srate = raw.info["sfreq"]
        edf_ch_names = raw.ch_names
        edf_data = raw.get_data() # (n_channels, n_samples)

        # Map channel names to lowercase/clean for robust matching
        edf_ch_map = {ch.strip(): i for i, ch in enumerate(edf_ch_names)}

        if channels_path and channels_path.exists():
            ch_df = pd.read_csv(channels_path, sep="\t")
            ch_obj_cols = {col: ch_df[col].astype(str).str.strip() for col in ch_df.select_dtypes(include="object").columns}
            ch_df = ch_df.assign(**ch_obj_cols)
        else:
            # Fallback if channels.tsv is absent: assume all good, non-SOZ
            ch_df = pd.DataFrame({
                "name": edf_ch_names,
                "status": "good",
                "status_description": "n/a",
                "type": "SEEG"
            })

        good_indices = []
        good_ch_names = []
        labels = []
        bad_ch_names = []
        soz_ch_names = []
        non_soz_ch_names = []
        status_descs = []

        for _, row in ch_df.iterrows():
            ch_name = str(row["name"]).strip()
            ch_type = str(row.get("type", "SEEG")).strip().upper()
            status = str(row.get("status", "good")).strip().lower()
            status_desc = str(row.get("status_description", "n/a")).strip().lower()

            # Check if this channel exists in the EDF file
            if ch_name not in edf_ch_map:
                # Try case-insensitive matching
                matched_key = next((k for k in edf_ch_map if k.lower() == ch_name.lower()), None)
                if not matched_key:
                    continue
                ch_idx = edf_ch_map[matched_key]
            else:
                ch_idx = edf_ch_map[ch_name]

            # Reject bad channels or non-SEEG channels (e.g., EKG, DC, trig)
            if status == "bad" or ch_type != "SEEG" or ch_name.upper().startswith(("EKG", "ECG", "DC", "TRIG")):
                bad_ch_names.append(ch_name)
                continue

            # Derive Ground Truth SOZ label ('soz' or 'resect,soz')
            is_soz = 1 if "soz" in status_desc else 0

            good_indices.append(ch_idx)
            good_ch_names.append(ch_name)
            labels.append(is_soz)
            status_descs.append(status_desc)

            if is_soz:
                soz_ch_names.append(ch_name)
            else:
                non_soz_ch_names.append(ch_name)

        filtered_data = edf_data[good_indices, :]

        # Identify Resected / Surgical Area (Epileptogenic Zone targets)
        ez_resect_ch_names = [
            ch for ch, desc in zip(good_ch_names, status_descs)
            if "resect" in desc
        ]

        # Clinical metadata from participants.tsv
        sub_meta = {}
        if not self.participants_df.empty:
            match = self.participants_df[self.participants_df["participant_id"] == recording_info["subject_id"]]
            if not match.empty:
                sub_meta = match.iloc[0].to_dict()

        # Parse seizure onset/offset events if events.tsv exists
        events_info = {"has_seizure_events": False, "seizure_onset_sec": None, "seizure_offset_sec": None, "seizure_duration_sec": None}
        events_path = recording_info.get("events_path")
        if events_path and Path(events_path).exists():
            try:
                ev_df = pd.read_csv(events_path, sep="\t")
                sz_onsets = ev_df[ev_df["trial_type"].astype(str).str.lower().str.contains("sz onset|seizure onset|onset")]
                sz_offsets = ev_df[ev_df["trial_type"].astype(str).str.lower().str.contains("sz offset|seizure offset|offset")]
                
                if not sz_onsets.empty:
                    onset_val = float(sz_onsets.iloc[0]["onset"])
                    events_info["has_seizure_events"] = True
                    events_info["seizure_onset_sec"] = onset_val
                    if not sz_offsets.empty:
                        offset_val = float(sz_offsets.iloc[0]["onset"])
                        events_info["seizure_offset_sec"] = offset_val
                        events_info["seizure_duration_sec"] = round(offset_val - onset_val, 2)
            except Exception:
                pass

        return {
            "subject_id": recording_info["subject_id"],
            "task": recording_info["task"],
            "filename": recording_info["filename"],
            "sampling_rate": orig_srate,
            "channel_names": good_ch_names,
            "raw_data": filtered_data,
            "labels": np.array(labels, dtype=int),
            "status_descriptions": status_descs,
            "soz_channel_names": soz_ch_names,
            "non_soz_channel_names": non_soz_ch_names,
            "ez_resect_channel_names": ez_resect_ch_names,
            "bad_channel_names": bad_ch_names,
            "n_channels": len(good_ch_names),
            "n_soz": len(soz_ch_names),
            "n_non_soz": len(non_soz_ch_names),
            "n_ez_resect": len(ez_resect_ch_names),
            "duration_sec": filtered_data.shape[1] / orig_srate,
            "clinical_metadata": sub_meta,
            "events_info": events_info
        }
