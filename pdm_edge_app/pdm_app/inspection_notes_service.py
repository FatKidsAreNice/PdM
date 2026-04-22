from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pdm_app.event_utils import make_event_signature


@dataclass
class EventInspectionNote:
    event_signature: str
    stage: str
    start_timestamp: str
    end_timestamp: str
    verdict: str
    notes: str
    updated_at: str


class EventInspectionNoteService:
    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def load_dataframe(self) -> pd.DataFrame:
        records = self._load_records()
        if not records:
            return pd.DataFrame(
                columns=[
                    "event_signature",
                    "stage",
                    "start_timestamp",
                    "end_timestamp",
                    "verdict",
                    "notes",
                    "updated_at",
                ]
            )
        df = pd.DataFrame(records)
        for column in ["start_timestamp", "end_timestamp", "updated_at"]:
            df[column] = pd.to_datetime(df[column], errors="coerce")
        return df.sort_values("updated_at", ascending=False).reset_index(drop=True)

    def upsert(self, *, stage: str, start_timestamp: str, end_timestamp: str, verdict: str, notes: str) -> str:
        records = self._load_records()
        signature = make_event_signature(stage, start_timestamp, end_timestamp)
        payload = EventInspectionNote(
            event_signature=signature,
            stage=stage,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            verdict=verdict,
            notes=notes,
            updated_at=pd.Timestamp.utcnow().isoformat(),
        )
        updated = False
        for index, record in enumerate(records):
            if record.get("event_signature") == signature:
                records[index] = asdict(payload)
                updated = True
                break
        if not updated:
            records.append(asdict(payload))
        self._write_records(records)
        return signature

    def get_record(self, *, stage: str, start_timestamp: object, end_timestamp: object) -> dict[str, Any] | None:
        signature = make_event_signature(stage, start_timestamp, end_timestamp)
        for record in self._load_records():
            if record.get("event_signature") == signature:
                return record
        return None

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
