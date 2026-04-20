from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pdm_app.config_loader import AppConfig


@dataclass
class Stage1Result:
    dataframe: pd.DataFrame
    baseline_table: pd.DataFrame
    alerts: pd.DataFrame


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

        baseline_table = self._build_hourly_baseline(working_df)
        working_df = self._apply_baseline(working_df, baseline_table)
        working_df = self._apply_rules(working_df)
        alerts = self._extract_alerts(working_df)

        self._logger.info(
            "Stage 1 abgeschlossen: %s Warnungen/Anomalien erkannt",
            len(alerts),
        )
        return Stage1Result(dataframe=working_df, baseline_table=baseline_table, alerts=alerts)

    def _build_hourly_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby("hour_of_day")

        baseline_table = pd.DataFrame({"hour_of_day": sorted(df["hour_of_day"].dropna().unique())})
        for metric in self.BASELINE_METRICS:
            medians = grouped[metric].median().reset_index(name=f"{metric}_baseline_median")
            mads = grouped[metric].apply(self._mad).reset_index(name=f"{metric}_baseline_mad")
            baseline_table = baseline_table.merge(medians, on="hour_of_day", how="left")
            baseline_table = baseline_table.merge(mads, on="hour_of_day", how="left")

        return baseline_table.sort_values("hour_of_day").reset_index(drop=True)

    def _apply_baseline(self, df: pd.DataFrame, baseline_table: pd.DataFrame) -> pd.DataFrame:
        merged = df.merge(baseline_table, on="hour_of_day", how="left")

        for metric in self.BASELINE_METRICS:
            median_column = f"{metric}_baseline_median"
            mad_column = f"{metric}_baseline_mad"
            robust_scale = (1.4826 * merged[mad_column]).replace(0, np.nan)
            robust_scale = robust_scale.fillna(1e-9)
            merged[f"{metric}_robust_z"] = (merged[metric] - merged[median_column]) / robust_scale

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
            ],
        ].copy()
        alerts = alerts.rename(
            columns={
                "stage1_severity": "severity",
                "stage1_message": "message",
            }
        )
        alerts["stage"] = "Stage 1"
        alerts = alerts[[
            "timestamp",
            "stage",
            "severity",
            "message",
            "avg_decibel",
            "peak_decibel",
            "min_decibel",
            "peak_minus_min",
            "quality_flag",
        ]]
        return alerts

    @staticmethod
    def _mad(series: pd.Series) -> float:
        median = series.median()
        return float(np.median(np.abs(series.dropna() - median)))
