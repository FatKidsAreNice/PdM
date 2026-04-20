from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pdm_app.config_loader import AppConfig


@dataclass
class LoadedData:
    dataframe: pd.DataFrame
    numeric_columns: list[str]


class CsvDataService:
    REQUIRED_COLUMNS: tuple[str, ...] = (
        "id",
        "avg_decibel",
        "peak_decibel",
        "min_decibel",
        "measurement_duration",
        "samples_count",
        "created_at",
    )

    OPTIONAL_NUMERIC_COLUMNS: tuple[str, ...] = (
        "low_freq_avg",
        "mid_freq_avg",
        "high_freq_avg",
    )

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger

    def load(self) -> LoadedData:
        self._logger.info("CSV wird geladen: %s", self._config.csv_path)

        raw_df = pd.read_csv(
            self._config.csv_path,
            sep=self._config.csv.separator,
            encoding=self._config.csv.encoding,
            low_memory=False,
        )

        inverted_rename_map = {
            original_name: internal_name
            for internal_name, original_name in self._config.columns.items()
            if original_name in raw_df.columns
        }

        df = raw_df.rename(columns=inverted_rename_map).copy()
        df["__row_id"] = np.arange(len(df), dtype=np.int64)

        missing = [column for column in self.REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Pflichtspalten fehlen in der CSV: {missing}")

        created_text = df["created_at"].astype(str).str.strip()

        timestamp = pd.to_datetime(
            created_text,
            format=self._config.datetime.format,
            errors="coerce",
        )

        if timestamp.isna().all():
            timestamp = pd.to_datetime(
                created_text,
                dayfirst=True,
                errors="coerce",
            )

        df["timestamp"] = timestamp

        invalid_timestamp_count = int(df["timestamp"].isna().sum())
        if invalid_timestamp_count == len(df):
            sample_values = created_text.head(5).tolist()
            raise ValueError(
                "Die Zeitspalte konnte nicht geparst werden. "
                f"Format in config.json prüfen. Beispielwerte: {sample_values}"
            )

        if invalid_timestamp_count > 0:
            self._logger.warning(
                "Zeitspalte teilweise ungültig: %s von %s Zeilen konnten nicht geparst werden.",
                invalid_timestamp_count,
                len(df),
            )

        numeric_columns = list(self.REQUIRED_COLUMNS[:-1]) + list(self.OPTIONAL_NUMERIC_COLUMNS)
        for column in numeric_columns:
            if column in df.columns:
                df[column] = df[column].apply(self._parse_numeric_value)

        df = df.sort_values(["timestamp", "__row_id"], na_position="last").reset_index(drop=True)
        self._create_quality_columns(df)
        self._create_derived_columns(df)

        available_numeric_columns = [
            column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])
        ]

        self._logger.info("CSV geladen: %s Zeilen", len(df))
        self._logger.info("Verfügbare numerische Spalten: %s", ", ".join(available_numeric_columns))

        return LoadedData(dataframe=df, numeric_columns=available_numeric_columns)

    def _create_quality_columns(self, df: pd.DataFrame) -> None:
        df["delta_minutes"] = df["timestamp"].diff().dt.total_seconds().div(60.0)
        df["gap_flag"] = df["delta_minutes"].fillna(0).gt(self._config.analysis.gap_threshold_minutes)
        df["segment_id"] = df["gap_flag"].cumsum()
        df["quality_flag"] = np.where(
            df["samples_count"] < self._config.analysis.reduced_quality_min_samples,
            "reduced_quality",
            "normal",
        )
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["date"] = df["timestamp"].dt.date

    def _create_derived_columns(self, df: pd.DataFrame) -> None:
        df["peak_minus_avg"] = df["peak_decibel"] - df["avg_decibel"]
        df["avg_minus_min"] = df["avg_decibel"] - df["min_decibel"]
        df["peak_minus_min"] = df["peak_decibel"] - df["min_decibel"]

        optional_bands = [column for column in self.OPTIONAL_NUMERIC_COLUMNS if column in df.columns]
        if len(optional_bands) == 3:
            band_sum = df[optional_bands].sum(axis=1)
            safe_sum = band_sum.replace(0, np.nan)
            df["low_share"] = df["low_freq_avg"] / safe_sum
            df["mid_share"] = df["mid_freq_avg"] / safe_sum
            df["high_share"] = df["high_freq_avg"] / safe_sum

    @staticmethod
    def _parse_numeric_value(value: object) -> float:
        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)

        text = str(value).strip()
        if text == "":
            return np.nan

        text = text.replace(" ", "")

        if "," in text and "." in text:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif text.count(".") > 1 and "," not in text:
            first_dot_index = text.find(".")
            text = text[: first_dot_index + 1] + text[first_dot_index + 1 :].replace(".", "")
        elif text.count(",") > 1 and "." not in text:
            first_comma_index = text.find(",")
            text = text[: first_comma_index + 1] + text[first_comma_index + 1 :].replace(",", "")
        else:
            text = text.replace(",", ".")

        try:
            return float(text)
        except ValueError:
            return np.nan