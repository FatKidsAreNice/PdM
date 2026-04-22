from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DefectLabel:
    label_id: str
    source_stage: str
    event_start: str
    event_end: str
    defect_type: str
    repaired_at: str
    repaired_what: str
    pre_failure_hours: int
    target_metric: str
    notes: str
    created_at: str


class DefectLabelService:
    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def load_dataframe(self) -> pd.DataFrame:
        records = self._load_records()
        if not records:
            return pd.DataFrame(
                columns=[
                    "label_id",
                    "source_stage",
                    "event_start",
                    "event_end",
                    "repaired_at",
                    "learning_window_start",
                    "learning_window_end",
                    "defect_type",
                    "repaired_what",
                    "pre_failure_hours",
                    "target_metric",
                    "notes",
                    "created_at",
                ]
            )

        df = pd.DataFrame(records)
        df["event_start"] = pd.to_datetime(df["event_start"], errors="coerce")
        df["event_end"] = pd.to_datetime(df["event_end"], errors="coerce")
        df["repaired_at"] = pd.to_datetime(df["repaired_at"], errors="coerce")
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        pre_failure_hours = pd.to_numeric(df["pre_failure_hours"], errors="coerce").fillna(24).astype(int)
        df["pre_failure_hours"] = pre_failure_hours
        learning_end = df["repaired_at"].where(df["repaired_at"].notna(), df["event_end"])
        df["learning_window_end"] = learning_end
        df["learning_window_start"] = learning_end - pd.to_timedelta(df["pre_failure_hours"], unit="h")
        return df.sort_values("event_start", ascending=False).reset_index(drop=True)

    def upsert(
        self,
        *,
        label_id: str | None,
        source_stage: str,
        event_start: str,
        event_end: str,
        defect_type: str,
        repaired_at: str,
        repaired_what: str,
        pre_failure_hours: int,
        target_metric: str,
        notes: str,
    ) -> str:
        records = self._load_records()
        resolved_id = label_id or f"LBL-{uuid.uuid4().hex[:10].upper()}"
        payload = DefectLabel(
            label_id=resolved_id,
            source_stage=source_stage,
            event_start=event_start,
            event_end=event_end,
            defect_type=defect_type,
            repaired_at=repaired_at,
            repaired_what=repaired_what,
            pre_failure_hours=int(pre_failure_hours),
            target_metric=target_metric,
            notes=notes,
            created_at=pd.Timestamp.utcnow().isoformat(),
        )

        updated = False
        for index, record in enumerate(records):
            if record.get("label_id") == resolved_id:
                records[index] = asdict(payload)
                updated = True
                break

        if not updated:
            records.append(asdict(payload))

        self._write_records(records)
        return resolved_id

    def delete(self, label_id: str) -> None:
        records = [record for record in self._load_records() if record.get("label_id") != label_id]
        self._write_records(records)

    def _load_records(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8-sig") as file:
            raw = json.load(file)
        if not isinstance(raw, list):
            return []
        return [record for record in raw if isinstance(record, dict)]

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
