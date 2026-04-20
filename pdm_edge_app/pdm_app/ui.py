from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd

from pdm_app.config_loader import AppConfig
from pdm_app.data_service import CsvDataService, LoadedData
from pdm_app.stage1_service import Stage1Result, Stage1Service
from pdm_app.stage2_service import Stage2Result, Stage2Service


@dataclass
class ApplicationState:
    loaded_data: LoadedData | None = None
    stage1_result: Stage1Result | None = None
    stage2_result: Stage2Result | None = None


class TkTextHandler(logging.Handler):
    def __init__(self, widget: tk.Text) -> None:
        super().__init__()
        self._widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        self._widget.after(0, self._append, message)

    def _append(self, message: str) -> None:
        self._widget.configure(state="normal")
        self._widget.insert("end", message + "\n")
        self._widget.see("end")
        self._widget.configure(state="disabled")


class PdmEdgeApplication(tk.Tk):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.title("PDM Edge CSV App")
        self.geometry("1560x920")
        self.minsize(1280, 800)

        self._config = config
        self._state = ApplicationState()
        self._logger = logging.getLogger("pdm_edge_app")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        self._data_service = CsvDataService(config, self._logger)
        self._stage1_service = Stage1Service(config, self._logger)
        self._stage2_service = Stage2Service(config, self._logger)

        self._build_layout()
        self._configure_logging()
        self._load_and_run_pipeline()

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        toolbar = ttk.Frame(self, padding=8)
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(5, weight=1)

        ttk.Button(toolbar, text="CSV neu laden", command=self._load_and_run_pipeline).grid(row=0, column=0, padx=4)
        ttk.Button(toolbar, text="Nur Stage 1 neu rechnen", command=self._rerun_stage1).grid(row=0, column=1, padx=4)
        ttk.Button(toolbar, text="Nur Stage 2 neu rechnen", command=self._rerun_stage2).grid(row=0, column=2, padx=4)
        ttk.Label(toolbar, text=f"Config: {Path('config.json').resolve()}").grid(row=0, column=6, sticky="e")

        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=1, column=0, sticky="nsew")

        self._overview_tab = ttk.Frame(self._notebook)
        self._stage1_tab = ttk.Frame(self._notebook)
        self._stage2_tab = ttk.Frame(self._notebook)
        self._explorer_tab = ttk.Frame(self._notebook)

        self._notebook.add(self._overview_tab, text="Übersicht")
        self._notebook.add(self._stage1_tab, text="Stage 1")
        self._notebook.add(self._stage2_tab, text="Stage 2")
        self._notebook.add(self._explorer_tab, text="Explorer")

        self._build_overview_tab()
        self._build_stage1_tab()
        self._build_stage2_tab()
        self._build_explorer_tab()
        self._build_terminal()

    def _build_terminal(self) -> None:
        terminal_frame = ttk.LabelFrame(self, text="Terminal / Warnungen", padding=6)
        terminal_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        terminal_frame.columnconfigure(0, weight=1)
        terminal_frame.rowconfigure(0, weight=1)

        self._terminal_text = tk.Text(terminal_frame, height=10, wrap="word", state="disabled")
        self._terminal_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(terminal_frame, orient="vertical", command=self._terminal_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._terminal_text.configure(yscrollcommand=scrollbar.set)

    def _configure_logging(self) -> None:
        handler = TkTextHandler(self._terminal_text)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        self._logger.addHandler(handler)

    def _build_overview_tab(self) -> None:
        self._overview_tab.columnconfigure(0, weight=2)
        self._overview_tab.columnconfigure(1, weight=1)
        self._overview_tab.rowconfigure(1, weight=1)

        summary_frame = ttk.LabelFrame(self._overview_tab, text="Zusammenfassung", padding=8)
        summary_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        summary_frame.columnconfigure(1, weight=1)

        self._summary_var = tk.StringVar(value="Noch keine Daten geladen")
        ttk.Label(summary_frame, textvariable=self._summary_var, justify="left").grid(row=0, column=0, sticky="w")

        chart_frame = ttk.LabelFrame(self._overview_tab, text="Basis-Zeitreihe", padding=8)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self._overview_figure, self._overview_axis = plt.subplots(figsize=(10, 5))
        self._overview_canvas = FigureCanvasTkAgg(self._overview_figure, master=chart_frame)
        self._overview_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        alerts_frame = ttk.LabelFrame(self._overview_tab, text="Aktive Meldungen", padding=8)
        alerts_frame.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)
        alerts_frame.columnconfigure(0, weight=1)
        alerts_frame.rowconfigure(0, weight=1)
        self._overview_alerts_tree = self._create_alert_tree(alerts_frame)
        self._overview_alerts_tree.grid(row=0, column=0, sticky="nsew")

    def _build_stage1_tab(self) -> None:
        self._stage1_tab.columnconfigure(0, weight=1)
        self._stage1_tab.rowconfigure(1, weight=1)
        self._stage1_tab.rowconfigure(2, weight=1)

        controls = ttk.Frame(self._stage1_tab, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="Metrik:").grid(row=0, column=0, padx=4)
        self._stage1_metric = tk.StringVar(value="avg_decibel")
        stage1_metric_box = ttk.Combobox(
            controls,
            textvariable=self._stage1_metric,
            state="readonly",
            values=[
                "avg_decibel",
                "peak_decibel",
                "min_decibel",
                "peak_minus_avg",
                "avg_minus_min",
                "peak_minus_min",
                "avg_decibel_robust_z",
                "peak_minus_min_robust_z",
                "peak_minus_avg_robust_z",
            ],
            width=28,
        )
        stage1_metric_box.grid(row=0, column=1, padx=4)
        ttk.Button(controls, text="Diagramm aktualisieren", command=self._refresh_stage1_plots).grid(row=0, column=2, padx=4)

        chart_frame = ttk.LabelFrame(self._stage1_tab, text="Stage 1 Zeitverlauf", padding=8)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self._stage1_figure, self._stage1_axis = plt.subplots(figsize=(11, 4.5))
        self._stage1_canvas = FigureCanvasTkAgg(self._stage1_figure, master=chart_frame)
        self._stage1_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        lower_frame = ttk.Frame(self._stage1_tab, padding=8)
        lower_frame.grid(row=2, column=0, sticky="nsew")
        lower_frame.columnconfigure(0, weight=1)
        lower_frame.columnconfigure(1, weight=1)
        lower_frame.rowconfigure(0, weight=1)

        baseline_frame = ttk.LabelFrame(lower_frame, text="Baseline nach Stunde", padding=8)
        baseline_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        baseline_frame.columnconfigure(0, weight=1)
        baseline_frame.rowconfigure(0, weight=1)
        self._stage1_baseline_figure, self._stage1_baseline_axis = plt.subplots(figsize=(6, 4))
        self._stage1_baseline_canvas = FigureCanvasTkAgg(self._stage1_baseline_figure, master=baseline_frame)
        self._stage1_baseline_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        alerts_frame = ttk.LabelFrame(lower_frame, text="Stage 1 Meldungen", padding=8)
        alerts_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        alerts_frame.columnconfigure(0, weight=1)
        alerts_frame.rowconfigure(0, weight=1)
        self._stage1_alerts_tree = self._create_alert_tree(alerts_frame)
        self._stage1_alerts_tree.grid(row=0, column=0, sticky="nsew")

    def _build_stage2_tab(self) -> None:
        self._stage2_tab.columnconfigure(0, weight=1)
        self._stage2_tab.rowconfigure(1, weight=1)
        self._stage2_tab.rowconfigure(2, weight=1)

        controls = ttk.Frame(self._stage2_tab, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="Darstellung:").grid(row=0, column=0, padx=4)
        self._stage2_view_mode = tk.StringVar(value="Multivariate Score")
        view_box = ttk.Combobox(
            controls,
            textvariable=self._stage2_view_mode,
            state="readonly",
            values=["Multivariate Score", "PCA Scatter"],
            width=24,
        )
        view_box.grid(row=0, column=1, padx=4)
        ttk.Button(controls, text="Diagramm aktualisieren", command=self._refresh_stage2_plots).grid(row=0, column=2, padx=4)

        charts_frame = ttk.Frame(self._stage2_tab, padding=8)
        charts_frame.grid(row=1, column=0, sticky="nsew")
        charts_frame.columnconfigure(0, weight=1)
        charts_frame.columnconfigure(1, weight=1)
        charts_frame.rowconfigure(0, weight=1)

        left_frame = ttk.LabelFrame(charts_frame, text="Korrelationen", padding=8)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        self._stage2_corr_figure, self._stage2_corr_axis = plt.subplots(figsize=(6, 5))
        self._stage2_corr_canvas = FigureCanvasTkAgg(self._stage2_corr_figure, master=left_frame)
        self._stage2_corr_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        right_frame = ttk.LabelFrame(charts_frame, text="Multivariate Ansicht", padding=8)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        self._stage2_view_figure, self._stage2_view_axis = plt.subplots(figsize=(6, 5))
        self._stage2_view_canvas = FigureCanvasTkAgg(self._stage2_view_figure, master=right_frame)
        self._stage2_view_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        alerts_frame = ttk.LabelFrame(self._stage2_tab, text="Stage 2 Meldungen", padding=8)
        alerts_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
        alerts_frame.columnconfigure(0, weight=1)
        alerts_frame.rowconfigure(0, weight=1)
        self._stage2_alerts_tree = self._create_alert_tree(alerts_frame)
        self._stage2_alerts_tree.grid(row=0, column=0, sticky="nsew")

    def _build_explorer_tab(self) -> None:
        self._explorer_tab.columnconfigure(0, weight=0)
        self._explorer_tab.columnconfigure(1, weight=1)
        self._explorer_tab.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(self._explorer_tab, text="CSV Explorer", padding=8)
        controls.grid(row=0, column=0, sticky="ns", padx=8, pady=8)

        ttk.Label(controls, text="X-Spalte:").grid(row=0, column=0, sticky="w")
        self._explorer_x_column = tk.StringVar(value="timestamp")
        self._explorer_x_combo = ttk.Combobox(controls, textvariable=self._explorer_x_column, state="readonly", width=30)
        self._explorer_x_combo.grid(row=1, column=0, pady=(0, 8), sticky="ew")

        ttk.Label(controls, text="Y-Spalten:").grid(row=2, column=0, sticky="w")
        self._explorer_y_listbox = tk.Listbox(controls, selectmode="multiple", height=22, exportselection=False, width=30)
        self._explorer_y_listbox.grid(row=3, column=0, sticky="nsew")
        ttk.Button(controls, text="Explorer aktualisieren", command=self._refresh_explorer_plot).grid(row=4, column=0, pady=8, sticky="ew")

        chart_frame = ttk.LabelFrame(self._explorer_tab, text="Freie Diagramme", padding=8)
        chart_frame.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self._explorer_figure, self._explorer_axis = plt.subplots(figsize=(10, 6))
        self._explorer_canvas = FigureCanvasTkAgg(self._explorer_figure, master=chart_frame)
        self._explorer_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _load_and_run_pipeline(self) -> None:
        try:
            self._state.loaded_data = self._data_service.load()
            self._state.stage1_result = self._stage1_service.run(self._state.loaded_data.dataframe)
            self._state.stage2_result = self._stage2_service.run(self._state.stage1_result.dataframe)
            self._push_alerts_to_terminal()
            self._refresh_all_views()
        except Exception as error:  # noqa: BLE001
            self._logger.exception("Fehler beim Laden oder Berechnen")
            messagebox.showerror("Fehler", str(error))

    def _rerun_stage1(self) -> None:
        if self._state.loaded_data is None:
            messagebox.showwarning("Hinweis", "Es sind noch keine Daten geladen.")
            return
        try:
            self._state.stage1_result = self._stage1_service.run(self._state.loaded_data.dataframe)
            self._state.stage2_result = self._stage2_service.run(self._state.stage1_result.dataframe)
            self._push_alerts_to_terminal()
            self._refresh_all_views()
        except Exception as error:  # noqa: BLE001
            self._logger.exception("Fehler in Stage 1")
            messagebox.showerror("Fehler", str(error))

    def _rerun_stage2(self) -> None:
        if self._state.stage1_result is None:
            messagebox.showwarning("Hinweis", "Stage 1 wurde noch nicht berechnet.")
            return
        try:
            self._state.stage2_result = self._stage2_service.run(self._state.stage1_result.dataframe)
            self._push_alerts_to_terminal()
            self._refresh_all_views()
        except Exception as error:  # noqa: BLE001
            self._logger.exception("Fehler in Stage 2")
            messagebox.showerror("Fehler", str(error))

    def _push_alerts_to_terminal(self) -> None:
        stage1_alerts = self._state.stage1_result.alerts if self._state.stage1_result is not None else pd.DataFrame()
        stage2_alerts = self._state.stage2_result.alerts if self._state.stage2_result is not None else pd.DataFrame()

        for _, row in stage1_alerts.head(20).iterrows():
            self._logger.warning("%s | %s | %s", row["timestamp"], row["severity"], row["message"])
        for _, row in stage2_alerts.head(20).iterrows():
            self._logger.warning("%s | %s | %s", row["timestamp"], row["severity"], row["message"])

    def _refresh_all_views(self) -> None:
        self._refresh_overview()
        self._refresh_stage1_plots()
        self._refresh_stage2_plots()
        self._refresh_explorer_controls()
        self._refresh_explorer_plot()

    def _refresh_overview(self) -> None:
        if self._state.loaded_data is None or self._state.stage1_result is None or self._state.stage2_result is None:
            return

        df = self._state.loaded_data.dataframe
        stage1_alerts = self._state.stage1_result.alerts
        stage2_alerts = self._state.stage2_result.alerts
        gap_count = int(df["gap_flag"].sum())
        reduced_quality_count = int((df["quality_flag"] == "reduced_quality").sum())
        summary_text = (
            f"CSV: {self._config.csv_path}\n"
            f"Zeilen: {len(df):,}\n"
            f"Messlücken: {gap_count}\n"
            f"Reduced Quality: {reduced_quality_count}\n"
            f"Stage 1 Meldungen: {len(stage1_alerts)}\n"
            f"Stage 2 Meldungen: {len(stage2_alerts)}"
        )
        self._summary_var.set(summary_text)

        self._overview_axis.clear()
        plot_df = self._get_plot_df(self._state.stage1_result.dataframe, ["avg_decibel", "peak_decibel", "min_decibel"])
        for column in ["avg_decibel", "peak_decibel", "min_decibel"]:
            self._overview_axis.plot(plot_df["timestamp"], plot_df[column], label=column)
        self._overview_axis.set_title("Basis-Zeitreihe")
        self._overview_axis.set_xlabel("Zeit")
        self._overview_axis.set_ylabel("dB")
        self._overview_axis.legend(loc="upper right")
        self._overview_axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        self._overview_figure.autofmt_xdate()
        self._overview_canvas.draw_idle()

        self._fill_alert_tree(self._overview_alerts_tree, pd.concat([stage1_alerts, stage2_alerts], ignore_index=True).head(200))

    def _refresh_stage1_plots(self) -> None:
        if self._state.stage1_result is None:
            return

        metric = self._stage1_metric.get()
        df = self._state.stage1_result.dataframe
        plot_df = self._get_plot_df(df, [metric])

        self._stage1_axis.clear()
        self._stage1_axis.plot(plot_df["timestamp"], plot_df[metric], label=metric)

        alert_df = plot_df[plot_df["stage1_severity"].isin(["WARNING", "ANOMALY"])]
        if not alert_df.empty:
            self._stage1_axis.scatter(alert_df["timestamp"], alert_df[metric], s=12, label="Stage1 Alert")

        self._stage1_axis.set_title(f"Stage 1 Verlauf: {metric}")
        self._stage1_axis.set_xlabel("Zeit")
        self._stage1_axis.set_ylabel(metric)
        self._stage1_axis.legend(loc="upper right")
        self._stage1_axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        self._stage1_figure.autofmt_xdate()
        self._stage1_canvas.draw_idle()

        baseline_table = self._state.stage1_result.baseline_table
        self._stage1_baseline_axis.clear()
        baseline_metric = metric if metric in ("avg_decibel", "peak_minus_avg", "peak_minus_min") else "avg_decibel"
        self._stage1_baseline_axis.plot(
            baseline_table["hour_of_day"],
            baseline_table[f"{baseline_metric}_baseline_median"],
            marker="o",
        )
        self._stage1_baseline_axis.set_title(f"Baseline-Stundenprofil: {baseline_metric}")
        self._stage1_baseline_axis.set_xlabel("Stunde")
        self._stage1_baseline_axis.set_ylabel("Median")
        self._stage1_baseline_axis.set_xticks(range(0, 24, 2))
        self._stage1_baseline_canvas.draw_idle()

        self._fill_alert_tree(self._stage1_alerts_tree, self._state.stage1_result.alerts.head(200))

    def _refresh_stage2_plots(self) -> None:
        if self._state.stage2_result is None:
            return

        self._stage2_corr_axis.clear()
        correlation_matrix = self._state.stage2_result.correlation_matrix
        image = self._stage2_corr_axis.imshow(correlation_matrix.values, aspect="auto")
        self._stage2_corr_axis.set_xticks(range(len(correlation_matrix.columns)))
        self._stage2_corr_axis.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
        self._stage2_corr_axis.set_yticks(range(len(correlation_matrix.index)))
        self._stage2_corr_axis.set_yticklabels(correlation_matrix.index)
        self._stage2_corr_axis.set_title("Korrelationsmatrix")
        self._stage2_corr_figure.colorbar(image, ax=self._stage2_corr_axis, fraction=0.046, pad=0.04)
        self._stage2_corr_canvas.draw_idle()

        self._stage2_view_axis.clear()
        df = self._state.stage2_result.dataframe
        plot_df = self._get_plot_df(df, ["multivariate_score", "pc1", "pc2"])
        mode = self._stage2_view_mode.get()
        if mode == "PCA Scatter":
            self._stage2_view_axis.scatter(plot_df["pc1"], plot_df["pc2"], s=10)
            self._stage2_view_axis.set_xlabel("PC1")
            self._stage2_view_axis.set_ylabel("PC2")
            self._stage2_view_axis.set_title("PCA Scatter")
        else:
            self._stage2_view_axis.plot(plot_df["timestamp"], plot_df["multivariate_score"], label="multivariate_score")
            alert_df = plot_df[plot_df["stage2_severity"].isin(["WARNING", "ANOMALY"])]
            if not alert_df.empty:
                self._stage2_view_axis.scatter(alert_df["timestamp"], alert_df["multivariate_score"], s=12, label="Stage2 Alert")
            self._stage2_view_axis.set_xlabel("Zeit")
            self._stage2_view_axis.set_ylabel("Score")
            self._stage2_view_axis.set_title("Multivariate Score")
            self._stage2_view_axis.legend(loc="upper right")
            self._stage2_view_axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
            self._stage2_view_figure.autofmt_xdate()
        self._stage2_view_canvas.draw_idle()

        self._fill_alert_tree(self._stage2_alerts_tree, self._state.stage2_result.alerts.head(200))

    def _refresh_explorer_controls(self) -> None:
        if self._state.stage2_result is None:
            return

        df = self._state.stage2_result.dataframe
        numeric_columns = [
            column for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column]) and column not in {"segment_id"}
        ]
        x_values = ["timestamp"] + numeric_columns
        self._explorer_x_combo["values"] = x_values
        if self._explorer_x_column.get() not in x_values:
            self._explorer_x_column.set("timestamp")

        self._explorer_y_listbox.delete(0, "end")
        for column in numeric_columns:
            self._explorer_y_listbox.insert("end", column)

        preferred_columns = ["avg_decibel", "peak_decibel", "min_decibel"]
        for index, column in enumerate(numeric_columns):
            if column in preferred_columns:
                self._explorer_y_listbox.selection_set(index)

    def _refresh_explorer_plot(self) -> None:
        if self._state.stage2_result is None:
            return

        df = self._state.stage2_result.dataframe
        x_column = self._explorer_x_column.get()
        selected_indices = self._explorer_y_listbox.curselection()
        selected_columns = [self._explorer_y_listbox.get(index) for index in selected_indices]
        if not selected_columns:
            selected_columns = ["avg_decibel"] if "avg_decibel" in df.columns else []

        if not selected_columns:
            return

        plot_df = self._get_plot_df(df, selected_columns + ([x_column] if x_column != "timestamp" else []))

        self._explorer_axis.clear()
        for column in selected_columns:
            self._explorer_axis.plot(plot_df[x_column], plot_df[column], label=column)
        self._explorer_axis.set_title("CSV Explorer")
        self._explorer_axis.set_xlabel(x_column)
        self._explorer_axis.set_ylabel("Wert")
        self._explorer_axis.legend(loc="upper right")
        if x_column == "timestamp":
            self._explorer_axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
            self._explorer_figure.autofmt_xdate()
        self._explorer_canvas.draw_idle()

    def _get_plot_df(self, df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
        existing_columns = [
            column for column in ["timestamp", "stage1_severity", "stage2_severity"] + required_columns
            if column in df.columns
        ]
        required_existing = [column for column in required_columns if column in df.columns]
        working_df = df[list(dict.fromkeys(existing_columns))].dropna(subset=required_existing).copy()
        max_points = self._config.analysis.plot.max_points_per_series
        if len(working_df) <= max_points:
            return working_df

        step = max(1, len(working_df) // max_points)
        return working_df.iloc[::step, :].copy()

    @staticmethod
    def _create_alert_tree(parent: ttk.Frame) -> ttk.Treeview:
        columns = ("timestamp", "stage", "severity", "message")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=10)
        tree.heading("timestamp", text="Zeit")
        tree.heading("stage", text="Stufe")
        tree.heading("severity", text="Schwere")
        tree.heading("message", text="Meldung")
        tree.column("timestamp", width=160, anchor="w")
        tree.column("stage", width=80, anchor="w")
        tree.column("severity", width=80, anchor="w")
        tree.column("message", width=420, anchor="w")

        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")
        return tree

    @staticmethod
    def _fill_alert_tree(tree: ttk.Treeview, alerts: pd.DataFrame) -> None:
        for item in tree.get_children():
            tree.delete(item)

        if alerts.empty:
            return

        for _, row in alerts.iterrows():
            timestamp = row.get("timestamp", "")
            if pd.notna(timestamp):
                timestamp_text = pd.to_datetime(timestamp).strftime("%d.%m.%Y %H:%M")
            else:
                timestamp_text = ""
            tree.insert(
                "",
                "end",
                values=(
                    timestamp_text,
                    row.get("stage", ""),
                    row.get("severity", ""),
                    row.get("message", ""),
                ),
            )
