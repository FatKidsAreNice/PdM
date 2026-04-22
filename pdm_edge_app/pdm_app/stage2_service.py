from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pdm_app.config_loader import AppConfig
from pdm_app.event_utils import AlertContextSummary, build_alert_events, summarize_alert_context


@dataclass
class Stage2Result:
    dataframe: pd.DataFrame
    correlation_matrix: pd.DataFrame
    alerts: pd.DataFrame
    feature_columns: list[str]
    events: pd.DataFrame
    context_summary: AlertContextSummary
    computed_at: pd.Timestamp


class Stage2Service:
    BASE_FEATURE_COLUMNS: tuple[str, ...] = (
        "avg_decibel",
        "peak_decibel",
        "min_decibel",
        "peak_minus_avg",
        "avg_minus_min",
        "peak_minus_min",
    )

    OPTIONAL_FEATURE_COLUMNS: tuple[str, ...] = (
        "low_share",
        "mid_share",
        "high_share",
    )

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger

    def run(self, df: pd.DataFrame) -> Stage2Result:
        self._logger.info("Stage 2 startet: multivariate Muster")
        if "__row_id" not in df.columns:
            raise ValueError("Interne Zeilen-ID '__row_id' fehlt. Daten bitte neu laden.")

        working_df = df.copy()
        feature_columns = [column for column in self.BASE_FEATURE_COLUMNS if column in working_df.columns]
        feature_columns.extend([column for column in self.OPTIONAL_FEATURE_COLUMNS if column in working_df.columns])

        clean_df = working_df.dropna(subset=feature_columns).copy()
        if clean_df.empty:
            raise ValueError("Stage 2 kann nicht berechnet werden, da keine vollständigen Feature-Zeilen vorliegen.")

        correlation_matrix = clean_df[feature_columns].corr()
        feature_matrix = clean_df[feature_columns].to_numpy(dtype=float)
        means = feature_matrix.mean(axis=0)
        stds = feature_matrix.std(axis=0)
        stds[stds == 0] = 1.0
        standardized = (feature_matrix - means) / stds

        _, _, vt = np.linalg.svd(standardized, full_matrices=False)
        components = vt[:2].T
        scores = standardized @ components
        clean_df["pc1"] = scores[:, 0]
        clean_df["pc2"] = scores[:, 1] if scores.shape[1] > 1 else 0.0
        clean_df["multivariate_score"] = np.sqrt(np.sum(standardized**2, axis=1))

        threshold = float(clean_df["multivariate_score"].quantile(self._config.analysis.stage2.distance_quantile))
        clean_df["stage2_warning_now"] = clean_df["multivariate_score"] >= threshold
        clean_df["stage2_persistent_warning"] = (
            clean_df.groupby("segment_id")["stage2_warning_now"]
            .transform(
                lambda series: (
                    series.astype(int).rolling(
                        window=self._config.analysis.stage2.persistence_windows,
                        min_periods=1,
                    ).sum()
                    >= self._config.analysis.stage2.persistence_windows
                )
            )
            .astype(bool)
        )
        clean_df["stage2_severity"] = np.select(
            condlist=[clean_df["stage2_persistent_warning"], clean_df["stage2_warning_now"]],
            choicelist=["ANOMALY", "WARNING"],
            default="NORMAL",
        )
        clean_df["stage2_message"] = np.select(
            condlist=[clean_df["stage2_persistent_warning"], clean_df["stage2_warning_now"]],
            choicelist=["Persistente multivariate Abweichung", "Multivariate Einzelwarnung"],
            default="Normalbereich",
        )

        working_df = working_df.merge(
            clean_df[["__row_id", "pc1", "pc2", "multivariate_score", "stage2_severity", "stage2_message"]],
            on="__row_id",
            how="left",
            validate="one_to_one",
        )
        working_df["stage2_severity"] = working_df["stage2_severity"].fillna("NORMAL")
        working_df["stage2_message"] = working_df["stage2_message"].fillna("Normalbereich")

        alerts = working_df.loc[
            working_df["stage2_severity"].isin(["WARNING", "ANOMALY"]),
            ["timestamp", "stage2_severity", "stage2_message", "multivariate_score", "quality_flag", "near_gap_flag"],
        ].copy()
        alerts = alerts.rename(columns={"stage2_severity": "severity", "stage2_message": "message"})
        alerts["stage"] = "Stage 2"
        alerts = alerts[["timestamp", "stage", "severity", "message", "multivariate_score", "quality_flag", "near_gap_flag"]]

        events = build_alert_events(
            working_df,
            stage_prefix="stage2",
            stage_name="Stage 2",
            expected_interval_minutes=self._config.analysis.expected_interval_minutes,
            score_column="multivariate_score",
        )
        context_summary = summarize_alert_context(working_df, stage_prefix="stage2", events=events)
        computed_at = pd.Timestamp.now()

        self._logger.info("Stage 2 abgeschlossen: %s Warnungszeilen, %s Events", len(alerts), len(events))
        return Stage2Result(
            dataframe=working_df,
            correlation_matrix=correlation_matrix,
            alerts=alerts,
            feature_columns=feature_columns,
            events=events,
            context_summary=context_summary,
            computed_at=computed_at,
        )
