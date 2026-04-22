from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pdm_app.config_loader import AppConfig
from pdm_app.event_utils import (
    AlertContextSummary,
    build_alert_events,
    get_alert_rows_with_event_ids,
    summarize_alert_context,
)


RULE_LABELS: dict[str, str] = {
    "warn_avg": "Pegel",
    "warn_spike": "Spike",
    "warn_range": "Spannweite",
}


@dataclass
class Stage1Result:
    dataframe: pd.DataFrame
    baseline_table: pd.DataFrame
    baseline_hourly: pd.DataFrame
    baseline_weekday_hourly: pd.DataFrame
    alerts: pd.DataFrame
    events: pd.DataFrame
    context_summary: AlertContextSummary
    rule_summary_text: str
    computed_at: pd.Timestamp
    applied_baseline_mode: str


class Stage1Service:
    BASELINE_METRICS: tuple[str, ...] = (
        "avg_decibel",
        "peak_minus_avg",
        "peak_minus_min",
    )

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger

    def run(self, df: pd.DataFrame) -> Stage1Result:
        self._logger.info("Stage 1 startet: Baseline und Regelwerk")
        working_df = df.copy()

        baseline_hourly = self.build_baseline_table(working_df, granularity="Stunde")
        baseline_weekday_hourly = self.build_baseline_table(working_df, granularity="Stunde + Wochentag")
        working_df = self._apply_baseline(working_df, baseline_hourly, baseline_weekday_hourly)
        working_df = self._apply_rules(working_df)
        alerts = self._extract_alerts(working_df)
        events = build_alert_events(
            working_df,
            stage_prefix="stage1",
            stage_name="Stage 1",
            expected_interval_minutes=self._config.analysis.expected_interval_minutes,
        )
        events = self._enrich_events(working_df, events)
        context_summary = summarize_alert_context(working_df, stage_prefix="stage1", events=events)
        rule_summary_text = self.build_rule_summary_text(working_df, events)
        computed_at = pd.Timestamp.now()

        self._logger.info(
            "Stage 1 abgeschlossen: %s Warnungszeilen, %s Events erkannt",
            len(alerts),
            len(events),
        )
        return Stage1Result(
            dataframe=working_df,
            baseline_table=baseline_hourly,
            baseline_hourly=baseline_hourly,
            baseline_weekday_hourly=baseline_weekday_hourly,
            alerts=alerts,
            events=events,
            context_summary=context_summary,
            rule_summary_text=rule_summary_text,
            computed_at=computed_at,
            applied_baseline_mode=self._config.analysis.baseline.default_granularity,
        )

    def build_baseline_table(self, df: pd.DataFrame, *, granularity: str) -> pd.DataFrame:
        if granularity == "Stunde + Wochentag":
            group_columns = ["weekday", "hour_of_day"]
            seed_columns = group_columns
        else:
            group_columns = ["hour_of_day"]
            seed_columns = group_columns

        baseline_source = df.dropna(subset=seed_columns + list(self.BASELINE_METRICS)).copy()
        if baseline_source.empty:
            return pd.DataFrame(columns=seed_columns)

        grouped = baseline_source.groupby(group_columns)
        baseline_table = grouped.size().reset_index(name="baseline_count")
        for metric in self.BASELINE_METRICS:
            medians = grouped[metric].median().reset_index(name=f"{metric}_baseline_median")
            mads = grouped[metric].apply(self._mad).reset_index(name=f"{metric}_baseline_mad")
            baseline_table = baseline_table.merge(medians, on=group_columns, how="left")
            baseline_table = baseline_table.merge(mads, on=group_columns, how="left")

        return baseline_table.sort_values(group_columns).reset_index(drop=True)

    def build_rule_summary_text(self, df: pd.DataFrame, events: pd.DataFrame | None = None) -> str:
        events = events if events is not None else pd.DataFrame()
        warn_avg_rows = int(df.get("warn_avg", pd.Series(dtype=bool)).fillna(False).sum())
        warn_range_rows = int(df.get("warn_range", pd.Series(dtype=bool)).fillna(False).sum())
        warn_spike_rows = int(df.get("warn_spike", pd.Series(dtype=bool)).fillna(False).sum())

        dominant_line = ""
        if not events.empty and "dominant_rule" in events.columns:
            counts = events["dominant_rule"].value_counts()
            if not counts.empty:
                dominant_line = f"Dominante Eventursache: {counts.index[0]} ({int(counts.iloc[0])} Events)"

        lines = [
            f"Pegel-Regel ausgelöst: {warn_avg_rows} Zeilen",
            f"Spannweiten-Regel ausgelöst: {warn_range_rows} Zeilen",
            f"Spike-Regel ausgelöst: {warn_spike_rows} Zeilen",
        ]
        if dominant_line:
            lines.append(dominant_line)
        return "\n".join(lines)

    def _apply_baseline(self, df: pd.DataFrame, baseline_hourly: pd.DataFrame, baseline_weekday_hourly: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(
            baseline_hourly.add_prefix("hourly_").rename(columns={"hourly_hour_of_day": "hour_of_day"}),
            on="hour_of_day",
            how="left",
        )
        merged = merged.merge(
            baseline_weekday_hourly.add_prefix("weekday_hourly_").rename(
                columns={
                    "weekday_hourly_weekday": "weekday",
                    "weekday_hourly_hour_of_day": "hour_of_day",
                }
            ),
            on=["weekday", "hour_of_day"],
            how="left",
        )

        use_weekday_hour = (
            self._config.analysis.baseline.default_granularity == "Stunde + Wochentag"
        )
        min_count = self._config.analysis.baseline.weekday_hour_min_count

        for metric in self.BASELINE_METRICS:
            hourly_median = merged[f"hourly_{metric}_baseline_median"]
            hourly_mad = merged[f"hourly_{metric}_baseline_mad"]
            weekday_median = merged.get(f"weekday_hourly_{metric}_baseline_median")
            weekday_mad = merged.get(f"weekday_hourly_{metric}_baseline_mad")
            weekday_count = merged.get("weekday_hourly_baseline_count")

            if use_weekday_hour and weekday_median is not None and weekday_count is not None:
                weekday_mask = weekday_count.fillna(0) >= min_count
            else:
                weekday_mask = pd.Series(False, index=merged.index)

            merged[f"{metric}_baseline_median"] = np.where(weekday_mask, weekday_median, hourly_median)
            merged[f"{metric}_baseline_mad"] = np.where(weekday_mask, weekday_mad, hourly_mad)
            merged[f"{metric}_baseline_scope"] = np.where(weekday_mask, "Stunde + Wochentag", "Stunde")

            robust_scale = (1.4826 * merged[f"{metric}_baseline_mad"]).replace(0, np.nan).fillna(1e-9)
            merged[f"{metric}_robust_z"] = (merged[metric] - merged[f"{metric}_baseline_median"]) / robust_scale

        return merged

    def _apply_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        stage1_config = self._config.analysis.stage1

        df["warn_avg"] = df["avg_decibel_robust_z"].abs() >= stage1_config.robust_z_threshold
        df["warn_range"] = df["peak_minus_min_robust_z"].abs() >= stage1_config.range_z_threshold
        df["warn_spike"] = df["peak_minus_avg_robust_z"].abs() >= stage1_config.range_z_threshold
        df["warning_now"] = df[["warn_avg", "warn_range", "warn_spike"]].any(axis=1)

        df["persistent_warning"] = (
            df.groupby("segment_id")["warning_now"]
            .transform(
                lambda series: (
                    series.astype(int)
                    .rolling(window=stage1_config.persistence_windows, min_periods=1)
                    .sum()
                    >= stage1_config.persistence_windows
                )
            )
            .astype(bool)
        )

        df["stage1_severity"] = np.select(
            condlist=[df["persistent_warning"], df["warning_now"]],
            choicelist=["ANOMALY", "WARNING"],
            default="NORMAL",
        )

        df["stage1_message"] = np.select(
            condlist=[
                df["persistent_warning"] & df["warn_avg"],
                df["persistent_warning"] & df["warn_range"],
                df["warning_now"] & df["warn_spike"],
                df["warning_now"],
            ],
            choicelist=[
                "Persistente Pegelabweichung gegen Baseline",
                "Persistente Spannweitenabweichung gegen Baseline",
                "Kurzfristig erhöhte Peak-Dynamik",
                "Einzelwarnung gegen Baseline",
            ],
            default="Normalbereich",
        )
        return df

    def _extract_alerts(self, df: pd.DataFrame) -> pd.DataFrame:
        alerts = df.loc[
            df["stage1_severity"].isin(["WARNING", "ANOMALY"]),
            [
                "timestamp",
                "stage1_severity",
                "stage1_message",
                "avg_decibel",
                "peak_decibel",
                "min_decibel",
                "peak_minus_min",
                "quality_flag",
                "near_gap_flag",
            ],
        ].copy()
        alerts = alerts.rename(columns={"stage1_severity": "severity", "stage1_message": "message"})
        alerts["stage"] = "Stage 1"
        return alerts[
            [
                "timestamp",
                "stage",
                "severity",
                "message",
                "avg_decibel",
                "peak_decibel",
                "min_decibel",
                "peak_minus_min",
                "quality_flag",
                "near_gap_flag",
            ]
        ]

    def _enrich_events(self, df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            for column in [
                "dominant_rule",
                "avg_decibel_robust_z_max",
                "peak_minus_avg_robust_z_max",
                "peak_minus_min_robust_z_max",
                "quality_label",
            ]:
                events[column] = pd.Series(dtype="object")
            return events

        alert_rows = get_alert_rows_with_event_ids(df, stage_prefix="stage1")
        details: list[dict[str, object]] = []
        for event_id, group in alert_rows.groupby("event_id", sort=True):
            rule_counts = {
                RULE_LABELS[rule_name]: int(group[rule_name].fillna(False).sum())
                for rule_name in RULE_LABELS
                if rule_name in group.columns
            }
            dominant_rule = max(rule_counts, key=rule_counts.get) if rule_counts else "Unklar"
            details.append(
                {
                    "event_id": int(event_id),
                    "dominant_rule": dominant_rule,
                    "avg_decibel_robust_z_max": float(group["avg_decibel_robust_z"].abs().max()),
                    "peak_minus_avg_robust_z_max": float(group["peak_minus_avg_robust_z"].abs().max()),
                    "peak_minus_min_robust_z_max": float(group["peak_minus_min_robust_z"].abs().max()),
                    "quality_label": "reduced_quality" if bool(group["is_reduced_quality"].any()) else "normal",
                }
            )
        detail_df = pd.DataFrame(details)
        merged = events.merge(detail_df, on="event_id", how="left")
        return merged

    @staticmethod
    def _mad(series: pd.Series) -> float:
        median = series.median()
        return float(np.median(np.abs(series.dropna() - median)))
