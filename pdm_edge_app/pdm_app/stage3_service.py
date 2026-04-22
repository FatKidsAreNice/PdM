from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from pdm_app.config_loader import AppConfig
from pdm_app.event_utils import (
    AlertContextSummary,
    build_alert_events,
    get_alert_rows_with_event_ids,
    summarize_alert_context,
)


AMPEL_LABELS: dict[int, str] = {
    0: "normal",
    1: "lokal auffällig",
    2: "konsistent auffällig",
    3: "persistent anomal",
}


@dataclass
class Stage3Result:
    dataframe: pd.DataFrame
    alerts: pd.DataFrame
    events: pd.DataFrame
    context_summary: AlertContextSummary
    feature_columns: list[str]
    training_row_count: int
    training_excluded_reduced_quality: bool
    training_excluded_near_gap: bool
    threshold_map: dict[str, float]
    ampel_event_counts: dict[str, int]
    overall_ampel_label: str
    computed_at: pd.Timestamp


class Stage3Service:
    BASE_FEATURE_COLUMNS: tuple[str, ...] = (
        "avg_decibel",
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

    def run(self, df: pd.DataFrame) -> Stage3Result:
        self._logger.info("Stage 3 startet: klassische unüberwachte ML-Verfahren")
        working_df = df.copy()

        base_features = [column for column in self.BASE_FEATURE_COLUMNS if column in working_df.columns]
        if len(base_features) < 2:
            raise ValueError("Stage 3 benötigt mindestens zwei numerische Basis-Feature-Spalten.")

        feature_columns, valid_df = self._select_feature_set(working_df, base_features)
        if valid_df.empty:
            raise ValueError("Stage 3 kann nicht berechnet werden, da keine vollständigen Zeilen vorliegen.")

        train_mask = pd.Series(True, index=valid_df.index)
        if self._config.analysis.stage3.exclude_reduced_quality_from_training:
            train_mask &= valid_df["quality_flag"].ne("reduced_quality")
        if self._config.analysis.stage3.exclude_near_gap_from_training:
            train_mask &= ~valid_df["near_gap_flag"].fillna(False)

        train_df = valid_df.loc[train_mask].copy()

        if len(valid_df) < 10:
            raise ValueError("Stage 3 benötigt mindestens 10 gültige Zeilen für die Modellbildung.")

        if len(train_df) < max(250, 5 * len(feature_columns)):
            self._logger.warning(
                "Stage 3: zu wenige Trainingszeilen nach Qualitätsfiltern (%s). Es wird auf alle validen Zeilen zurückgefallen.",
                len(train_df),
            )
            train_df = valid_df.copy()

        train_matrix = train_df[feature_columns].to_numpy(dtype=float)
        all_matrix = valid_df[feature_columns].to_numpy(dtype=float)

        means = train_matrix.mean(axis=0)
        stds = train_matrix.std(axis=0)
        stds[stds == 0] = 1.0

        train_scaled = (train_matrix - means) / stds
        all_scaled = (all_matrix - means) / stds

        contamination = min(max(self._config.analysis.stage3.contamination, 0.001), 0.2)
        lof_neighbors = max(2, min(self._config.analysis.stage3.lof_neighbors, max(2, len(train_df) - 1)))
        one_class_nu = min(max(self._config.analysis.stage3.one_class_nu, 0.001), 0.49)

        iforest = IsolationForest(
            contamination=contamination,
            random_state=self._config.analysis.stage3.random_state,
            n_estimators=200,
        )
        iforest.fit(train_scaled)
        iforest_score = -iforest.decision_function(all_scaled)
        iforest_threshold = float(np.quantile(-iforest.decision_function(train_scaled), 1.0 - contamination))
        iforest_flag = iforest_score >= iforest_threshold

        lof = LocalOutlierFactor(n_neighbors=lof_neighbors, novelty=True, contamination=contamination)
        lof.fit(train_scaled)
        lof_score = -lof.score_samples(all_scaled)
        lof_threshold = float(np.quantile(-lof.score_samples(train_scaled), 1.0 - contamination))
        lof_flag = lof_score >= lof_threshold

        ocsvm = OneClassSVM(nu=one_class_nu, kernel="rbf", gamma="scale")
        ocsvm.fit(train_scaled)
        ocsvm_score = -ocsvm.decision_function(all_scaled).ravel()
        ocsvm_threshold = float(np.quantile(-ocsvm.decision_function(train_scaled).ravel(), 1.0 - contamination))
        ocsvm_flag = ocsvm_score >= ocsvm_threshold

        score_frame = pd.DataFrame(
            {
                "__row_id": valid_df["__row_id"].to_numpy(),
                "stage3_iforest_score": iforest_score,
                "stage3_lof_score": lof_score,
                "stage3_ocsvm_score": ocsvm_score,
                "stage3_iforest_flag": iforest_flag,
                "stage3_lof_flag": lof_flag,
                "stage3_ocsvm_flag": ocsvm_flag,
            }
        )

        for score_column in ["stage3_iforest_score", "stage3_lof_score", "stage3_ocsvm_score"]:
            score_frame[f"{score_column}_norm"] = self._normalize_series(score_frame[score_column])

        score_frame["stage3_votes"] = score_frame[
            ["stage3_iforest_flag", "stage3_lof_flag", "stage3_ocsvm_flag"]
        ].astype(int).sum(axis=1)

        score_frame["stage3_consensus_score"] = score_frame[
            [
                "stage3_iforest_score_norm",
                "stage3_lof_score_norm",
                "stage3_ocsvm_score_norm",
            ]
        ].mean(axis=1)

        score_frame["stage3_warning_now"] = score_frame["stage3_votes"] >= 1
        score_frame["stage3_strong_now"] = score_frame["stage3_votes"] >= 2

        working_df = working_df.merge(score_frame, on="__row_id", how="left", validate="one_to_one")

        bool_columns = [
            "stage3_iforest_flag",
            "stage3_lof_flag",
            "stage3_ocsvm_flag",
            "stage3_warning_now",
            "stage3_strong_now",
        ]
        for column in bool_columns:
            working_df[column] = working_df[column].fillna(False).astype(bool)

        working_df["stage3_votes"] = working_df["stage3_votes"].fillna(0).astype(int)
        working_df["stage3_consensus_score"] = working_df["stage3_consensus_score"].fillna(0.0)

        for column in ["stage3_iforest_score", "stage3_lof_score", "stage3_ocsvm_score"]:
            working_df[column] = working_df[column].fillna(0.0)

        working_df["stage3_persistent_warning"] = (
            working_df.groupby("segment_id")["stage3_strong_now"]
            .transform(
                lambda series: (
                    series.astype(int)
                    .rolling(window=self._config.analysis.stage3.persistence_windows, min_periods=1)
                    .sum()
                    >= self._config.analysis.stage3.persistence_windows
                )
            )
            .astype(bool)
        )

        working_df["stage3_severity"] = np.select(
            condlist=[working_df["stage3_persistent_warning"], working_df["stage3_warning_now"]],
            choicelist=["ANOMALY", "WARNING"],
            default="NORMAL",
        )

        working_df["stage3_message"] = np.select(
            condlist=[
                working_df["stage3_persistent_warning"] & (working_df["stage3_votes"] == 3),
                working_df["stage3_persistent_warning"] & (working_df["stage3_votes"] >= 2),
                working_df["stage3_warning_now"] & (working_df["stage3_votes"] == 1),
                working_df["stage3_warning_now"],
            ],
            choicelist=[
                "Persistente Konsens-Anomalie in allen drei Modellen",
                "Persistente Konsens-Anomalie in mehreren Modellen",
                "Einzelmodell-Warnung",
                "Mehrmodell-Warnung",
            ],
            default="Normalbereich",
        )

        working_df = self._apply_ampel_to_rows(working_df)

        alerts = working_df.loc[
            working_df["stage3_severity"].isin(["WARNING", "ANOMALY"]),
            [
                "timestamp",
                "stage3_severity",
                "stage3_message",
                "stage3_consensus_score",
                "stage3_votes",
                "stage3_ampel_label",
                "quality_flag",
                "near_gap_flag",
            ],
        ].copy()

        alerts = alerts.rename(
            columns={
                "stage3_severity": "severity",
                "stage3_message": "message",
            }
        )
        alerts["stage"] = "Stage 3"
        alerts = alerts[
            [
                "timestamp",
                "stage",
                "severity",
                "message",
                "stage3_consensus_score",
                "stage3_votes",
                "stage3_ampel_label",
                "quality_flag",
                "near_gap_flag",
            ]
        ]

        events = build_alert_events(
            working_df,
            stage_prefix="stage3",
            stage_name="Stage 3",
            expected_interval_minutes=self._config.analysis.expected_interval_minutes,
            score_column="stage3_consensus_score",
        )
        events = self._apply_ampel_to_events(working_df, events)
        events = self._enrich_events(working_df, events)

        context_summary = summarize_alert_context(working_df, stage_prefix="stage3", events=events)

        threshold_map = {
            "Isolation Forest": iforest_threshold,
            "LOF": lof_threshold,
            "One-Class SVM": ocsvm_threshold,
        }

        ampel_event_counts = {
            label: int(events["ampel_label"].eq(label).sum()) if not events.empty else 0
            for label in AMPEL_LABELS.values()
        }

        overall_ampel_label = self._resolve_overall_ampel(events)
        computed_at = pd.Timestamp.now()

        self._logger.info(
            "Stage 3 abgeschlossen: %s Warnungszeilen, %s Events, Training=%s Zeilen, Gesamtstatus=%s, Features=%s",
            len(alerts),
            len(events),
            len(train_df),
            overall_ampel_label,
            ", ".join(feature_columns),
        )

        return Stage3Result(
            dataframe=working_df,
            alerts=alerts,
            events=events,
            context_summary=context_summary,
            feature_columns=feature_columns,
            training_row_count=int(len(train_df)),
            training_excluded_reduced_quality=self._config.analysis.stage3.exclude_reduced_quality_from_training,
            training_excluded_near_gap=self._config.analysis.stage3.exclude_near_gap_from_training,
            threshold_map=threshold_map,
            ampel_event_counts=ampel_event_counts,
            overall_ampel_label=overall_ampel_label,
            computed_at=computed_at,
        )

    def _select_feature_set(
        self,
        df: pd.DataFrame,
        base_features: list[str],
    ) -> tuple[list[str], pd.DataFrame]:
        base_valid_df = df.dropna(subset=base_features + ["timestamp"]).copy()
        if base_valid_df.empty:
            return base_features, base_valid_df

        optional_features_present = [column for column in self.OPTIONAL_FEATURE_COLUMNS if column in df.columns]
        if len(optional_features_present) != len(self.OPTIONAL_FEATURE_COLUMNS):
            self._logger.info("Stage 3 nutzt nur Basisfeatures, da nicht alle Frequency-Shares vorhanden sind.")
            return base_features, base_valid_df

        candidate_features = base_features + list(self.OPTIONAL_FEATURE_COLUMNS)
        candidate_valid_df = df.dropna(subset=candidate_features + ["timestamp"]).copy()

        min_rows_with_frequency = max(250, 5 * len(candidate_features))
        if len(candidate_valid_df) < min_rows_with_frequency:
            self._logger.warning(
                "Stage 3 nutzt keine Frequency-Shares, da nur %s vollständige Zeilen mit low/mid/high_share vorliegen. Fallback auf Basisfeatures.",
                len(candidate_valid_df),
            )
            return base_features, base_valid_df

        self._logger.info(
            "Stage 3 nutzt erweiterte Features inkl. Frequency-Shares: %s",
            ", ".join(candidate_features),
        )
        return candidate_features, candidate_valid_df

    def _apply_ampel_to_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        raw_rank = np.zeros(len(df), dtype=int)
        raw_rank = np.where(df["stage3_warning_now"], 1, raw_rank)
        raw_rank = np.where(
            df["stage3_strong_now"] | (df["stage3_warning_now"] & (df["stage3_consensus_score"] >= 0.65)),
            2,
            raw_rank,
        )
        raw_rank = np.where(df["stage3_persistent_warning"] & (df["stage3_votes"] >= 2), 3, raw_rank)

        discount = (
            df["quality_flag"].eq("reduced_quality").astype(int)
            + df["near_gap_flag"].fillna(False).astype(int)
        )
        adjusted_rank = np.clip(raw_rank - discount, 0, 3)

        df["stage3_ampel_rank_raw"] = raw_rank.astype(int)
        df["stage3_ampel_rank"] = adjusted_rank.astype(int)
        df["stage3_ampel_label"] = pd.Series(df["stage3_ampel_rank"]).map(AMPEL_LABELS).fillna("normal")
        df["stage3_ampel_discounted"] = df["stage3_ampel_rank"] < df["stage3_ampel_rank_raw"]
        return df

    def _apply_ampel_to_events(self, df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            events["ampel_rank"] = pd.Series(dtype=int)
            events["ampel_label"] = pd.Series(dtype=str)
            return events

        alert_rows = get_alert_rows_with_event_ids(df, stage_prefix="stage3")
        event_rows: list[dict[str, object]] = []
        persistence_rows = max(2, int(self._config.analysis.stage3.persistence_windows))

        for event_id, group in alert_rows.groupby("event_id", sort=True):
            base_rank = int(group["stage3_ampel_rank"].max())
            if base_rank < 2 and (
                int(group["stage3_strong_now"].sum()) >= 2
                or (int(group["stage3_votes"].max()) >= 2 and len(group) >= 2)
            ):
                base_rank = 2
            if base_rank < 3 and int(group["stage3_strong_now"].sum()) >= persistence_rows:
                base_rank = 3

            discount = int(group["is_reduced_quality"].any()) + int(group["is_near_gap"].any())
            rank = int(np.clip(base_rank - discount, 0, 3))
            event_rows.append(
                {
                    "event_id": int(event_id),
                    "ampel_rank": rank,
                    "ampel_label": AMPEL_LABELS.get(rank, "normal"),
                }
            )

        event_ampel_df = pd.DataFrame(event_rows)
        merged = events.merge(event_ampel_df, on="event_id", how="left")
        merged["ampel_rank"] = merged["ampel_rank"].fillna(0).astype(int)
        merged["ampel_label"] = merged["ampel_label"].fillna("normal")
        return merged

    def _enrich_events(self, df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            for column in [
                "consensus_count",
                "persistent_flag",
                "ampel_reason",
                "stage3_consensus_score_max",
                "stage3_iforest_score_max",
                "stage3_lof_score_max",
                "stage3_ocsvm_score_max",
            ]:
                events[column] = pd.Series(dtype=object)
            return events

        alert_rows = get_alert_rows_with_event_ids(df, stage_prefix="stage3")
        details: list[dict[str, object]] = []

        for event_id, group in alert_rows.groupby("event_id", sort=True):
            consensus_count = int(group["stage3_votes"].max())
            persistent_flag = bool(group["stage3_persistent_warning"].any())
            score_max = float(group["stage3_consensus_score"].max())
            iforest_max = float(group["stage3_iforest_score"].max())
            lof_max = float(group["stage3_lof_score"].max())
            ocsvm_max = float(group["stage3_ocsvm_score"].max())
            duration = float(len(group) * self._config.analysis.expected_interval_minutes)
            reduced = bool(group["is_reduced_quality"].any())
            near_gap = bool(group["is_near_gap"].any())
            ampel_label = AMPEL_LABELS.get(int(np.clip(group["stage3_ampel_rank"].max(), 0, 3)), "normal")

            ampel_reason = self._build_ampel_reason(
                consensus_count=consensus_count,
                persistent_flag=persistent_flag,
                duration_minutes=duration,
                reduced_quality=reduced,
                near_gap=near_gap,
                ampel_label=ampel_label,
            )

            details.append(
                {
                    "event_id": int(event_id),
                    "consensus_count": consensus_count,
                    "persistent_flag": persistent_flag,
                    "stage3_consensus_score_max": score_max,
                    "stage3_iforest_score_max": iforest_max,
                    "stage3_lof_score_max": lof_max,
                    "stage3_ocsvm_score_max": ocsvm_max,
                    "ampel_reason": ampel_reason,
                }
            )

        detail_df = pd.DataFrame(details)
        merged = events.merge(detail_df, on="event_id", how="left")
        return merged

    @staticmethod
    def _build_ampel_reason(
        *,
        consensus_count: int,
        persistent_flag: bool,
        duration_minutes: float,
        reduced_quality: bool,
        near_gap: bool,
        ampel_label: str,
    ) -> str:
        base = f"{consensus_count} Modell(e) gleichzeitig, Dauer {duration_minutes:.0f} min"
        if persistent_flag:
            base += ", Persistenz erkannt"
        else:
            base += ", keine Persistenz"

        if reduced_quality and near_gap:
            base += ", Reduced Quality + Gap-Nähe => Abschlag"
        elif reduced_quality:
            base += ", Reduced Quality => Abschlag"
        elif near_gap:
            base += ", Gap-Nähe => Abschlag"

        return f"{ampel_label}: {base}"

    @staticmethod
    def _resolve_overall_ampel(events: pd.DataFrame) -> str:
        if events.empty or "ampel_rank" not in events.columns:
            return "normal"
        max_rank = int(events["ampel_rank"].max())
        return AMPEL_LABELS.get(max_rank, "normal")

    @staticmethod
    def _normalize_series(series: pd.Series) -> pd.Series:
        lower = float(series.quantile(0.05))
        upper = float(series.quantile(0.95))
        if np.isclose(lower, upper):
            return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
        normalized = (series - lower) / (upper - lower)
        return normalized.clip(lower=0.0, upper=1.0)