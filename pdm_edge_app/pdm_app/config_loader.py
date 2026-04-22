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
class Stage3Config:
    contamination: float
    persistence_windows: int
    lof_neighbors: int
    one_class_nu: float
    exclude_reduced_quality_from_training: bool
    exclude_near_gap_from_training: bool
    random_state: int


@dataclass(frozen=True)
class BaselineConfig:
    default_granularity: str
    weekday_hour_min_count: int


@dataclass(frozen=True)
class InspectionConfig:
    window_hours: int


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
    stage3: Stage3Config
    baseline: BaselineConfig
    inspection: InspectionConfig


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
    labels_path: str
    inspection_notes_path: str


class ConfigLoader:
    @staticmethod
    def load(path: str) -> AppConfig:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")

        with config_path.open("r", encoding="utf-8-sig") as file:
            raw: dict[str, Any] = json.load(file)

        analysis_raw = raw.get("analysis", {})
        stage1_raw = analysis_raw.get("stage1", {})
        stage2_raw = analysis_raw.get("stage2", {})
        stage3_raw = analysis_raw.get("stage3", {})
        baseline_raw = analysis_raw.get("baseline", {})
        inspection_raw = analysis_raw.get("inspection", {})

        return AppConfig(
            csv_path=raw["csv_path"],
            csv=CsvConfig(
                separator=raw.get("csv", {}).get("separator", ","),
                encoding=raw.get("csv", {}).get("encoding", "utf-8"),
            ),
            datetime=DateTimeConfig(
                source_column=raw.get("datetime", {}).get("source_column", "created_at"),
                format=raw.get("datetime", {}).get("format", "%d.%m.%Y %H:%M"),
            ),
            columns=raw.get("columns", {}),
            labels_path=raw.get("labels_path", "defect_labels.json"),
            inspection_notes_path=raw.get("inspection_notes_path", "event_inspection_notes.json"),
            analysis=AnalysisConfig(
                expected_interval_minutes=float(analysis_raw.get("expected_interval_minutes", 1.0)),
                gap_threshold_minutes=float(analysis_raw.get("gap_threshold_minutes", 1.5)),
                reference_samples_count=int(analysis_raw.get("reference_samples_count", 114)),
                reduced_quality_min_samples=int(analysis_raw.get("reduced_quality_min_samples", 100)),
                baseline_grouping=str(analysis_raw.get("baseline_grouping", "hour")),
                plot=PlotConfig(
                    max_points_per_series=int(analysis_raw.get("plot", {}).get("max_points_per_series", 5000))
                ),
                stage1=Stage1Config(
                    robust_z_threshold=float(stage1_raw.get("robust_z_threshold", 3.0)),
                    range_z_threshold=float(stage1_raw.get("range_z_threshold", 3.0)),
                    persistence_windows=int(stage1_raw.get("persistence_windows", 5)),
                ),
                stage2=Stage2Config(
                    distance_quantile=float(stage2_raw.get("distance_quantile", 0.99)),
                    persistence_windows=int(stage2_raw.get("persistence_windows", 5)),
                ),
                stage3=Stage3Config(
                    contamination=float(stage3_raw.get("contamination", 0.01)),
                    persistence_windows=int(stage3_raw.get("persistence_windows", 5)),
                    lof_neighbors=int(stage3_raw.get("lof_neighbors", 35)),
                    one_class_nu=float(stage3_raw.get("one_class_nu", 0.03)),
                    exclude_reduced_quality_from_training=bool(
                        stage3_raw.get("exclude_reduced_quality_from_training", True)
                    ),
                    exclude_near_gap_from_training=bool(stage3_raw.get("exclude_near_gap_from_training", True)),
                    random_state=int(stage3_raw.get("random_state", 42)),
                ),
                baseline=BaselineConfig(
                    default_granularity=str(baseline_raw.get("default_granularity", "Stunde")),
                    weekday_hour_min_count=int(baseline_raw.get("weekday_hour_min_count", 12)),
                ),
                inspection=InspectionConfig(
                    window_hours=int(inspection_raw.get("window_hours", 2)),
                ),
            ),
        )
