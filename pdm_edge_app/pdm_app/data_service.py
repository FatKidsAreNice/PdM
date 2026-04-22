from __future__ import annotations

import logging
import re
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

    FREQ_BAND_COLUMNS: tuple[str, ...] = (
        "low_freq_avg",
        "mid_freq_avg",
        "high_freq_avg",
    )

    FREQ_PATTERN = re.compile(r"^\d{2}\.\d{3}\.\d{3}\.\d{3}\.\d{3}$")

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

        standard_numeric_columns = [
            "id",
            "avg_decibel",
            "peak_decibel",
            "min_decibel",
            "measurement_duration",
            "samples_count",
        ]

        for column in standard_numeric_columns:
            if column in df.columns:
                df[column] = df[column].apply(self._parse_standard_numeric_value)

        for column in self.FREQ_BAND_COLUMNS:
            if column in df.columns:
                df[f"{column}_raw"] = df[column]
                parsed = df[column].apply(self._parse_frequency_band_value)
                df[column] = parsed.map(lambda item: item[0])
                df[f"{column}_format_valid"] = parsed.map(lambda item: item[1])

        freq_validity_columns = [f"{column}_format_valid" for column in self.FREQ_BAND_COLUMNS if f"{column}_format_valid" in df.columns]
        if freq_validity_columns:
            df["freq_format_error"] = ~df[freq_validity_columns].all(axis=1)
        else:
            df["freq_format_error"] = False

        df = df.sort_values(["timestamp", "__row_id"], na_position="last").reset_index(drop=True)

        self._create_quality_columns(df)
        self._create_derived_columns(df)

        available_numeric_columns = [
            column for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column])
        ]

        invalid_freq_rows = int(df["freq_format_error"].sum()) if "freq_format_error" in df.columns else 0
        if invalid_freq_rows > 0:
            self._logger.warning(
                "Frequenzformatfehler erkannt: %s Zeilen enthalten ungültige low/mid/high-Werte und werden für Frequenznutzung ignoriert.",
                invalid_freq_rows,
            )

        self._logger.info("CSV geladen: %s Zeilen", len(df))
        self._logger.info("Verfügbare numerische Spalten: %s", ", ".join(available_numeric_columns))

        return LoadedData(dataframe=df, numeric_columns=available_numeric_columns)

    def _create_quality_columns(self, df: pd.DataFrame) -> None:
        df["delta_minutes"] = df["timestamp"].diff().dt.total_seconds().div(60.0)
        df["gap_flag"] = df["delta_minutes"].fillna(0).gt(self._config.analysis.gap_threshold_minutes)
        df["segment_id"] = df["gap_flag"].cumsum()

        next_gap = df["gap_flag"].shift(-1).fillna(False)
        df["near_gap_flag"] = df["gap_flag"].fillna(False) | next_gap

        df["quality_flag"] = np.where(
            df["samples_count"] < self._config.analysis.reduced_quality_min_samples,
            "reduced_quality",
            "normal",
        )
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["weekday"] = df["timestamp"].dt.weekday
        df["date"] = df["timestamp"].dt.date

    def _create_derived_columns(self, df: pd.DataFrame) -> None:
        df["peak_minus_avg"] = df["peak_decibel"] - df["avg_decibel"]
        df["avg_minus_min"] = df["avg_decibel"] - df["min_decibel"]
        df["peak_minus_min"] = df["peak_decibel"] - df["min_decibel"]

        if all(column in df.columns for column in self.FREQ_BAND_COLUMNS):
            valid_freq_mask = ~df["freq_format_error"]
            band_sum = df.loc[valid_freq_mask, list(self.FREQ_BAND_COLUMNS)].sum(axis=1)
            safe_sum = band_sum.replace(0, np.nan)

            df["low_share"] = np.nan
            df["mid_share"] = np.nan
            df["high_share"] = np.nan

            df.loc[valid_freq_mask, "low_share"] = df.loc[valid_freq_mask, "low_freq_avg"] / safe_sum
            df.loc[valid_freq_mask, "mid_share"] = df.loc[valid_freq_mask, "mid_freq_avg"] / safe_sum
            df.loc[valid_freq_mask, "high_share"] = df.loc[valid_freq_mask, "high_freq_avg"] / safe_sum

    @staticmethod
    def _parse_standard_numeric_value(value: object) -> float:
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

    def _parse_frequency_band_value(self, raw: object) -> tuple[float, bool]:
        if pd.isna(raw):
            return np.nan, False

        text = str(raw).strip()
        if text == "":
            return np.nan, False

        if not self.FREQ_PATTERN.fullmatch(text):
            return np.nan, False

        parts = text.split(".")
        normalized = parts[0] + "." + "".join(parts[1:])

        try:
            return float(normalized), True
        except ValueError:
            return np.nan, False
