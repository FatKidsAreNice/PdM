from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlotConfig:
    max_points_per_series: int


@dataclass(frozen=True)
class Stage1Config:
    robust_z_threshold: float
    range_z_threshold: float
    persistence_windows: int


@dataclass(frozen=True)
class Stage2Config:
    distance_quantile: float
    persistence_windows: int


@dataclass(frozen=True)
class AnalysisConfig:
    expected_interval_minutes: float
    gap_threshold_minutes: float
    reference_samples_count: int
    reduced_quality_min_samples: int
    baseline_grouping: str
    plot: PlotConfig
    stage1: Stage1Config
    stage2: Stage2Config


@dataclass(frozen=True)
class CsvConfig:
    separator: str
    encoding: str


@dataclass(frozen=True)
class DateTimeConfig:
    source_column: str
    format: str


@dataclass(frozen=True)
class AppConfig:
    csv_path: str
    csv: CsvConfig
    datetime: DateTimeConfig
    columns: dict[str, str]
    analysis: AnalysisConfig


class ConfigLoader:
    @staticmethod
    def load(path: str) -> AppConfig:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            raw: dict[str, Any] = json.load(file)

        analysis_raw = raw["analysis"]
        return AppConfig(
            csv_path=raw["csv_path"],
            csv=CsvConfig(**raw["csv"]),
            datetime=DateTimeConfig(**raw["datetime"]),
            columns=raw["columns"],
            analysis=AnalysisConfig(
                expected_interval_minutes=analysis_raw["expected_interval_minutes"],
                gap_threshold_minutes=analysis_raw["gap_threshold_minutes"],
                reference_samples_count=analysis_raw["reference_samples_count"],
                reduced_quality_min_samples=analysis_raw["reduced_quality_min_samples"],
                baseline_grouping=analysis_raw["baseline_grouping"],
                plot=PlotConfig(**analysis_raw["plot"]),
                stage1=Stage1Config(**analysis_raw["stage1"]),
                stage2=Stage2Config(**analysis_raw["stage2"]),
            ),
        )
