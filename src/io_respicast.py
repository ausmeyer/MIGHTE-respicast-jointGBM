from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional

import pandas as pd

TARGETS = ("ILI incidence", "ARI incidence")
TARGET_TO_STEM = {
    "ILI incidence": "ILI_incidence",
    "ARI incidence": "ARI_incidence",
}


@dataclass(frozen=True)
class DataFileSpec:
    source: str
    file_kind: str  # latest | snapshot
    target: str
    path: Path
    snapshot_date: Optional[pd.Timestamp]


def _parse_snapshot_date(path: Path) -> Optional[pd.Timestamp]:
    # Snapshot names: YYYY-MM-DD-ILI_incidence.csv / YYYY-MM-DD-ARI_incidence.csv
    match = re.match(r"^(\d{4}-\d{2}-\d{2})-(ILI|ARI)_incidence\.csv$", path.name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), errors="coerce")


def discover_target_files(target_data_dir: Path) -> List[DataFileSpec]:
    specs: List[DataFileSpec] = []
    # Top-level latest files (highest-priority canonical view from target-data root).
    for target in TARGETS:
        stem = TARGET_TO_STEM[target]
        latest_path = target_data_dir / f"latest-{stem}.csv"
        if latest_path.exists():
            specs.append(
                DataFileSpec(
                    source="target-data-root",
                    file_kind="latest",
                    target=target,
                    path=latest_path,
                    snapshot_date=None,
                )
            )

    for source_dir in sorted(p for p in target_data_dir.iterdir() if p.is_dir()):
        source = source_dir.name
        for target in TARGETS:
            stem = TARGET_TO_STEM[target]
            latest_path = source_dir / f"latest-{stem}.csv"
            if latest_path.exists():
                specs.append(
                    DataFileSpec(
                        source=source,
                        file_kind="latest",
                        target=target,
                        path=latest_path,
                        snapshot_date=None,
                    )
                )

            snap_dir = source_dir / "snapshots"
            if not snap_dir.exists():
                continue
            for path in sorted(snap_dir.glob(f"*-{stem}.csv")):
                snap_date = _parse_snapshot_date(path)
                if snap_date is None:
                    continue
                specs.append(
                    DataFileSpec(
                        source=source,
                        file_kind="snapshot",
                        target=target,
                        path=path,
                        snapshot_date=snap_date,
                    )
                )
    return specs


def read_target_file(spec: DataFileSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.path)
    expected = {"target", "location", "truth_date", "year_week", "value"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"{spec.path} missing required columns: {sorted(missing)}")

    out = df.loc[:, ["target", "location", "truth_date", "year_week", "value"]].copy()
    out["target"] = out["target"].astype(str)
    out["location"] = out["location"].astype(str)
    out["truth_date"] = pd.to_datetime(out["truth_date"], errors="coerce")
    out["year_week"] = out["year_week"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out.dropna(subset=["target", "location", "truth_date", "value"])
    out = out[out["target"].isin(TARGETS)].copy()

    out["source"] = spec.source
    out["file_kind"] = spec.file_kind
    out["file_path"] = str(spec.path)
    out["snapshot_date"] = spec.snapshot_date
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    return out


def load_locations_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"location_name", "iso2_code"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = df.loc[:, ["location_name", "iso2_code"]].copy()
    out["location_name"] = out["location_name"].astype(str)
    out["location"] = out["iso2_code"].astype(str)
    return out.loc[:, ["location", "location_name"]]


def source_priority_map() -> Dict[str, int]:
    # Lower rank = higher priority when all else ties.
    return {
        "target-data-root": -1,
        "ERVISS": 0,
        "FluID": 1,
    }
