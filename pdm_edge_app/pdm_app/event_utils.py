from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SEVERITY_RANK: dict[str, int] = {
    "NORMAL": 0,
    "WARNING": 1,
    "ANOMALY": 2,
}


@dataclass
class AlertContextSummary:
    row_alerts_total: int
    row_alerts_reduced_quality: int
    row_alerts_near_gap: int
    row_alerts_both: int
    events_total: int
    events_reduced_quality: int
    events_near_gap: int
    events_both: int

    @staticmethod
    def _percent(value: int, total: int) -> float:
        return 0.0 if total == 0 else (100.0 * value / total)

    def to_multiline_text(self) -> str:
        return (
            f"Warnungszeilen gesamt: {self.row_alerts_total}\n"
            f"- Reduced Quality: {self.row_alerts_reduced_quality} ({self._percent(self.row_alerts_reduced_quality, self.row_alerts_total):.1f}%)\n"
            f"- Gap-nah: {self.row_alerts_near_gap} ({self._percent(self.row_alerts_near_gap, self.row_alerts_total):.1f}%)\n"
            f"- Beides: {self.row_alerts_both} ({self._percent(self.row_alerts_both, self.row_alerts_total):.1f}%)\n"
            f"Events gesamt: {self.events_total}\n"
            f"- Events mit Reduced Quality: {self.events_reduced_quality} ({self._percent(self.events_reduced_quality, self.events_total):.1f}%)\n"
            f"- Events gap-nah: {self.events_near_gap} ({self._percent(self.events_near_gap, self.events_total):.1f}%)\n"
            f"- Events mit beidem: {self.events_both} ({self._percent(self.events_both, self.events_total):.1f}%)"
        )


def get_alert_rows_with_event_ids(df: pd.DataFrame, *, stage_prefix: str) -> pd.DataFrame:
    severity_column = f"{stage_prefix}_severity"
    message_column = f"{stage_prefix}_message"

    required_columns = [
        "__row_id",
        "timestamp",
        "segment_id",
        "quality_flag",
        "near_gap_flag",
        severity_column,
        message_column,
    ]
    optional_columns = [
        column for column in df.columns
        if column.startswith(stage_prefix)
        or column in {"avg_decibel_robust_z", "peak_minus_avg_robust_z", "peak_minus_min_robust_z"}
    ]
    selected_columns = list(dict.fromkeys(required_columns + optional_columns))

    alerts = df.loc[df[severity_column].isin(["WARNING", "ANOMALY"]), selected_columns].copy()
    if alerts.empty:
        alerts["event_id"] = pd.Series(dtype=int)
        return alerts

    alerts["__new_event"] = (
        alerts["__row_id"].diff().ne(1)
        | alerts["segment_id"].diff().ne(0)
    ).fillna(True)
    alerts["event_id"] = alerts["__new_event"].cumsum().astype(int)
    alerts["is_reduced_quality"] = alerts["quality_flag"].eq("reduced_quality")
    alerts["is_near_gap"] = alerts["near_gap_flag"].fillna(False).astype(bool)
    return alerts


def build_alert_events(
    df: pd.DataFrame,
    *,
    stage_prefix: str,
    stage_name: str,
    expected_interval_minutes: float,
    score_column: str | None = None,
) -> pd.DataFrame:
    severity_column = f"{stage_prefix}_severity"
    message_column = f"{stage_prefix}_message"

    alerts = get_alert_rows_with_event_ids(df, stage_prefix=stage_prefix)
    if alerts.empty:
        columns = [
            "event_id",
            "start_timestamp",
            "end_timestamp",
            "duration_minutes",
            "stage",
            "severity",
            "message",
            "row_count",
            "reduced_quality_rows",
            "near_gap_rows",
            "has_reduced_quality",
            "has_near_gap",
        ]
        if score_column is not None:
            columns.append(score_column)
        return pd.DataFrame(columns=columns)

    event_rows: list[dict[str, object]] = []
    for event_id, group in alerts.groupby("event_id", sort=True):
        severity = _max_severity(group[severity_column])
        messages = group[message_column].dropna().astype(str)
        message = " | ".join(messages.value_counts().index[:2]) if not messages.empty else ""

        event_row: dict[str, object] = {
            "event_id": int(event_id),
            "start_timestamp": group["timestamp"].min(),
            "end_timestamp": group["timestamp"].max(),
            "duration_minutes": float(len(group) * expected_interval_minutes),
            "stage": stage_name,
            "severity": severity,
            "message": message,
            "row_count": int(len(group)),
            "reduced_quality_rows": int(group["is_reduced_quality"].sum()),
            "near_gap_rows": int(group["is_near_gap"].sum()),
            "has_reduced_quality": bool(group["is_reduced_quality"].any()),
            "has_near_gap": bool(group["is_near_gap"].any()),
        }
        if score_column is not None and score_column in group.columns:
            event_row[score_column] = float(group[score_column].max())
        event_rows.append(event_row)

    return pd.DataFrame(event_rows).sort_values("start_timestamp").reset_index(drop=True)


def summarize_alert_context(
    df: pd.DataFrame,
    *,
    stage_prefix: str,
    events: pd.DataFrame,
) -> AlertContextSummary:
    severity_column = f"{stage_prefix}_severity"
    alerts = df.loc[df[severity_column].isin(["WARNING", "ANOMALY"])].copy()
    row_alerts_total = int(len(alerts))
    row_alerts_reduced_quality = int(alerts["quality_flag"].eq("reduced_quality").sum()) if row_alerts_total else 0
    row_alerts_near_gap = int(alerts["near_gap_flag"].fillna(False).sum()) if row_alerts_total else 0
    row_alerts_both = int(
        (alerts["quality_flag"].eq("reduced_quality") & alerts["near_gap_flag"].fillna(False)).sum()
    ) if row_alerts_total else 0

    events_total = int(len(events))
    events_reduced_quality = int(events["has_reduced_quality"].sum()) if events_total else 0
    events_near_gap = int(events["has_near_gap"].sum()) if events_total else 0
    events_both = int((events["has_reduced_quality"] & events["has_near_gap"]).sum()) if events_total else 0

    return AlertContextSummary(
        row_alerts_total=row_alerts_total,
        row_alerts_reduced_quality=row_alerts_reduced_quality,
        row_alerts_near_gap=row_alerts_near_gap,
        row_alerts_both=row_alerts_both,
        events_total=events_total,
        events_reduced_quality=events_reduced_quality,
        events_near_gap=events_near_gap,
        events_both=events_both,
    )


def filter_events_by_timerange(
    events: pd.DataFrame,
    start_timestamp: pd.Timestamp | None,
    end_timestamp: pd.Timestamp | None,
) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    filtered = events.copy()
    if start_timestamp is not None:
        filtered = filtered.loc[filtered["end_timestamp"] >= start_timestamp]
    if end_timestamp is not None:
        filtered = filtered.loc[filtered["start_timestamp"] <= end_timestamp]
    return filtered.reset_index(drop=True)


def build_event_window(df: pd.DataFrame, event_row: dict[str, object], *, hours_before_after: int) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    start = pd.to_datetime(event_row.get("start_timestamp"))
    end = pd.to_datetime(event_row.get("end_timestamp"))
    window_start = start - pd.Timedelta(hours=hours_before_after)
    window_end = end + pd.Timedelta(hours=hours_before_after)
    subset = df.loc[(df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)].copy()
    return subset, window_start, window_end


def make_event_signature(stage: str, start_timestamp: object, end_timestamp: object) -> str:
    start_text = pd.to_datetime(start_timestamp).strftime("%Y%m%dT%H%M%S") if pd.notna(start_timestamp) else "NA"
    end_text = pd.to_datetime(end_timestamp).strftime("%Y%m%dT%H%M%S") if pd.notna(end_timestamp) else "NA"
    return f"{stage}|{start_text}|{end_text}"


def _max_severity(severity_series: pd.Series) -> str:
    mapped = severity_series.astype(str).map(SEVERITY_RANK).fillna(0)
    max_rank = int(mapped.max()) if not mapped.empty else 0
    for name, rank in SEVERITY_RANK.items():
        if rank == max_rank:
            return name
    return "NORMAL"
