from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
import pandas as pd

from pdm_app.config_loader import AppConfig
from pdm_app.data_service import CsvDataService, LoadedData
from pdm_app.event_utils import (
    build_event_window,
    filter_events_by_timerange,
    make_event_signature,
    summarize_alert_context,
)
from pdm_app.inspection_notes_service import EventInspectionNoteService
from pdm_app.labels_service import DefectLabelService
from pdm_app.stage1_service import Stage1Result, Stage1Service
from pdm_app.stage2_service import Stage2Result, Stage2Service
from pdm_app.stage3_service import Stage3Result, Stage3Service


@dataclass
class ApplicationState:
    loaded_data: LoadedData | None = None
    stage1_result: Stage1Result | None = None
    stage2_result: Stage2Result | None = None
    stage3_result: Stage3Result | None = None
    labels_df: pd.DataFrame | None = None
    inspection_notes_df: pd.DataFrame | None = None


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


class ScrollableTabFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")

        self.content = ttk.Frame(self.canvas)
        self.content.columnconfigure(0, weight=1)

        self._window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_content_configure(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _on_mousewheel(self, event) -> None:
        if getattr(event, 'delta', 0):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif getattr(event, 'num', None) == 4:
            self.canvas.yview_scroll(-1, "units")
        elif getattr(event, 'num', None) == 5:
            self.canvas.yview_scroll(1, "units")

    def _bind_mousewheel(self, _event=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

class PdmEdgeApplication(tk.Tk):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.title("PDM Edge CSV App")
        self.geometry("1820x1080")
        self.minsize(1500, 940)

        self._config = config
        self._state = ApplicationState()
        self._logger = logging.getLogger("pdm_edge_app")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        self._data_service = CsvDataService(config, self._logger)
        self._stage1_service = Stage1Service(config, self._logger)
        self._stage2_service = Stage2Service(config, self._logger)
        self._stage3_service = Stage3Service(config, self._logger)
        self._labels_service = DefectLabelService(config.labels_path)
        self._inspection_notes_service = EventInspectionNoteService(config.inspection_notes_path)

        self._selected_event_record: dict[str, object] | None = None
        self._build_layout()
        self._configure_logging()
        self._load_and_run_pipeline()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        toolbar = ttk.Frame(self, padding=8)
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(8, weight=1)
        ttk.Button(toolbar, text="CSV neu laden", command=self._load_and_run_pipeline).grid(row=0, column=0, padx=4)
        ttk.Button(toolbar, text="Nur Stage 1 neu rechnen", command=self._rerun_stage1).grid(row=0, column=1, padx=4)
        ttk.Button(toolbar, text="Nur Stage 2 neu rechnen", command=self._rerun_stage2).grid(row=0, column=2, padx=4)
        ttk.Button(toolbar, text="Nur Stage 3 neu rechnen", command=self._rerun_stage3).grid(row=0, column=3, padx=4)
        ttk.Button(toolbar, text="Defektlabels neu laden", command=self._reload_labels).grid(row=0, column=4, padx=4)
        ttk.Button(toolbar, text="Inspektionsnotizen neu laden", command=self._reload_inspection_notes).grid(row=0, column=5, padx=4)
        ttk.Label(toolbar, text=f"Config: {Path('config.json').resolve()}").grid(row=0, column=8, sticky="e")

        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=1, column=0, sticky="nsew")

        self._overview_tab = ttk.Frame(self._notebook)
        self._stage1_tab = ttk.Frame(self._notebook)
        self._stage2_tab = ttk.Frame(self._notebook)
        self._stage3_tab = ttk.Frame(self._notebook)
        self._inspection_tab = ttk.Frame(self._notebook)
        self._labels_tab = ttk.Frame(self._notebook)
        self._explorer_tab = ttk.Frame(self._notebook)

        self._notebook.add(self._overview_tab, text="Übersicht")
        self._notebook.add(self._stage1_tab, text="Stage 1")
        self._notebook.add(self._stage2_tab, text="Stage 2")
        self._notebook.add(self._stage3_tab, text="Stage 3")
        self._notebook.add(self._inspection_tab, text="Event-Inspektion")
        self._notebook.add(self._labels_tab, text="Defektlabels")
        self._notebook.add(self._explorer_tab, text="Explorer")

        for tab in [
            self._overview_tab,
            self._stage1_tab,
            self._stage2_tab,
            self._stage3_tab,
            self._labels_tab,
            self._explorer_tab,
        ]:
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)

        self._overview_scroll = ScrollableTabFrame(self._overview_tab)
        self._overview_scroll.grid(row=0, column=0, sticky="nsew")
        self._overview_tab_content = self._overview_scroll.content

        self._stage1_scroll = ScrollableTabFrame(self._stage1_tab)
        self._stage1_scroll.grid(row=0, column=0, sticky="nsew")
        self._stage1_tab_content = self._stage1_scroll.content

        self._stage2_scroll = ScrollableTabFrame(self._stage2_tab)
        self._stage2_scroll.grid(row=0, column=0, sticky="nsew")
        self._stage2_tab_content = self._stage2_scroll.content

        self._stage3_scroll = ScrollableTabFrame(self._stage3_tab)
        self._stage3_scroll.grid(row=0, column=0, sticky="nsew")
        self._stage3_tab_content = self._stage3_scroll.content

        self._labels_scroll = ScrollableTabFrame(self._labels_tab)
        self._labels_scroll.grid(row=0, column=0, sticky="nsew")
        self._labels_tab_content = self._labels_scroll.content

        self._explorer_scroll = ScrollableTabFrame(self._explorer_tab)
        self._explorer_scroll.grid(row=0, column=0, sticky="nsew")
        self._explorer_tab_content = self._explorer_scroll.content

        self._build_overview_tab()
        self._build_stage1_tab()
        self._build_stage2_tab()
        self._build_stage3_tab()
        self._build_inspection_tab()
        self._build_labels_tab()
        self._build_explorer_tab()
        self._build_terminal()


    def _get_scrollable_content(self, tab_frame: ttk.Frame, attr_name: str) -> ttk.Frame:
        existing = getattr(self, attr_name, None)
        if existing is not None:
            return existing

        tab_frame.columnconfigure(0, weight=1)
        tab_frame.rowconfigure(0, weight=1)

        outer = ttk.Frame(tab_frame)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        content = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def _on_content_configure(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(window_id, width=event.width)

        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass

        content.bind("<Configure>", _on_content_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        setattr(self, attr_name, content)
        return content

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

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------
    def _build_overview_tab(self) -> None:
        parent_content = self._get_scrollable_content(self._overview_tab, "_overview_tab_content")
        parent = parent_content
        parent.columnconfigure(0, weight=2)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        summary_frame = ttk.LabelFrame(parent, text="Zusammenfassung", padding=8)
        summary_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        summary_frame.columnconfigure(0, weight=1)
        self._summary_var = tk.StringVar(value="Noch keine Daten geladen")
        ttk.Label(summary_frame, textvariable=self._summary_var, justify="left").grid(row=0, column=0, sticky="w")

        chart_frame = ttk.LabelFrame(parent, text="Basis-Zeitreihe", padding=8)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self._overview_figure, self._overview_axis = plt.subplots(figsize=(10, 5))
        self._overview_canvas = FigureCanvasTkAgg(self._overview_figure, master=chart_frame)
        self._overview_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        right_panel = ttk.Frame(parent)
        right_panel.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)

        context_frame = ttk.LabelFrame(right_panel, text="Warnungskontext", padding=8)
        context_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        context_frame.columnconfigure(0, weight=1)
        self._overview_context_var = tk.StringVar(value="")
        ttk.Label(context_frame, textvariable=self._overview_context_var, justify="left").grid(row=0, column=0, sticky="w")

        alerts_frame = ttk.LabelFrame(right_panel, text="Aktive Events", padding=8)
        alerts_frame.grid(row=1, column=0, sticky="nsew")
        alerts_frame.columnconfigure(0, weight=1)
        alerts_frame.rowconfigure(0, weight=1)
        self._overview_events_tree = self._create_event_tree(alerts_frame)
        self._overview_events_tree.grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    def _build_stage1_tab(self) -> None:
        parent_content = self._get_scrollable_content(self._stage1_tab, "_stage1_tab_content")
        parent = parent_content
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        parent.rowconfigure(3, weight=1)

        controls = ttk.Frame(parent, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="Metrik:").grid(row=0, column=0, padx=4)
        self._stage1_metric = tk.StringVar(value="avg_decibel")
        ttk.Combobox(
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
                "peak_minus_avg_robust_z",
                "peak_minus_min_robust_z",
            ],
            width=24,
        ).grid(row=0, column=1, padx=4)
        ttk.Label(controls, text="Zeitraum:").grid(row=0, column=2, padx=4)
        self._stage1_timeframe = tk.StringVar(value="Gesamt")
        ttk.Combobox(controls, textvariable=self._stage1_timeframe, state="readonly", values=self._timeframe_values(), width=18).grid(row=0, column=3, padx=4)
        ttk.Label(controls, text="Baseline-Modus:").grid(row=0, column=4, padx=4)
        self._stage1_baseline_mode = tk.StringVar(value="Global")
        ttk.Combobox(controls, textvariable=self._stage1_baseline_mode, state="readonly", values=["Global", "Aktueller Zeitraum"], width=18).grid(row=0, column=5, padx=4)
        ttk.Label(controls, text="Granularität:").grid(row=0, column=6, padx=4)
        self._stage1_baseline_granularity = tk.StringVar(value=self._config.analysis.baseline.default_granularity)
        ttk.Combobox(controls, textvariable=self._stage1_baseline_granularity, state="readonly", values=["Stunde", "Stunde + Wochentag"], width=18).grid(row=0, column=7, padx=4)
        ttk.Label(controls, text="Wochentag:").grid(row=0, column=8, padx=4)
        self._stage1_baseline_weekday = tk.StringVar(value="Auto")
        ttk.Combobox(controls, textvariable=self._stage1_baseline_weekday, state="readonly", values=self._weekday_values(), width=10).grid(row=0, column=9, padx=4)
        ttk.Button(controls, text="Aktualisieren", command=self._refresh_stage1_plots).grid(row=0, column=10, padx=6)
        ttk.Button(controls, text="i Baseline", command=self._show_stage1_baseline_info).grid(row=0, column=11, padx=3)
        ttk.Button(controls, text="i Pegel", command=self._show_stage1_level_info).grid(row=0, column=12, padx=3)
        ttk.Button(controls, text="i Spike/Spannweite", command=self._show_stage1_range_info).grid(row=0, column=13, padx=3)

        status_frame = ttk.LabelFrame(parent, text="Status", padding=8)
        status_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self._stage1_status_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self._stage1_status_var, justify="left").grid(row=0, column=0, sticky="w")

        chart_frame = ttk.LabelFrame(parent, text="Stage 1 Hauptdiagramm", padding=8)
        chart_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self._stage1_figure, self._stage1_axis = plt.subplots(figsize=(11.4, 4.6))
        self._stage1_figure.subplots_adjust(bottom=0.22, right=0.97, top=0.90, left=0.08)
        self._stage1_canvas = FigureCanvasTkAgg(self._stage1_figure, master=chart_frame)
        self._stage1_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        lower = ttk.Frame(parent, padding=8)
        lower.grid(row=3, column=0, sticky="nsew")
        lower.columnconfigure(0, weight=1)
        lower.columnconfigure(1, weight=1)
        lower.rowconfigure(0, weight=1)

        baseline_frame = ttk.LabelFrame(lower, text="Baseline", padding=8)
        baseline_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        baseline_frame.columnconfigure(0, weight=1)
        baseline_frame.rowconfigure(0, weight=1)
        self._stage1_baseline_figure, self._stage1_baseline_axis = plt.subplots(figsize=(6.2, 4.2))
        self._stage1_baseline_canvas = FigureCanvasTkAgg(self._stage1_baseline_figure, master=baseline_frame)
        self._stage1_baseline_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        right_panel = ttk.Frame(lower)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(2, weight=1)

        context_frame = ttk.LabelFrame(right_panel, text="Warnungskontext", padding=8)
        context_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._stage1_context_var = tk.StringVar(value="")
        ttk.Label(context_frame, textvariable=self._stage1_context_var, justify="left").grid(row=0, column=0, sticky="w")

        detail_frame = ttk.LabelFrame(right_panel, text="Stage-1-Eventdetails", padding=8)
        detail_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        self._stage1_detail_text = self._create_scrolled_text(detail_frame, height=8)
        self._stage1_detail_text.grid(row=0, column=0, sticky="nsew")

        events_frame = ttk.LabelFrame(right_panel, text="Stage 1 Events", padding=8)
        events_frame.grid(row=2, column=0, sticky="nsew")
        events_frame.columnconfigure(0, weight=1)
        events_frame.rowconfigure(0, weight=1)
        self._stage1_events_tree = self._create_event_tree(events_frame)
        self._stage1_events_tree.grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    def _build_stage2_tab(self) -> None:
        parent_content = self._get_scrollable_content(self._stage2_tab, "_stage2_tab_content")
        parent = parent_content
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=0)
        parent.rowconfigure(3, weight=1)

        controls = ttk.Frame(parent, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="Darstellung:").grid(row=0, column=0, padx=4)
        self._stage2_view_mode = tk.StringVar(value="Multivariate Score")
        ttk.Combobox(controls, textvariable=self._stage2_view_mode, state="readonly", values=["Multivariate Score", "PCA Scatter"], width=22).grid(row=0, column=1, padx=4)
        ttk.Label(controls, text="Zeitraum:").grid(row=0, column=2, padx=4)
        self._stage2_timeframe = tk.StringVar(value="Gesamt")
        ttk.Combobox(controls, textvariable=self._stage2_timeframe, state="readonly", values=self._timeframe_values(), width=18).grid(row=0, column=3, padx=4)
        ttk.Label(controls, text="Baseline-Modus:").grid(row=0, column=4, padx=4)
        self._stage2_baseline_mode = tk.StringVar(value="Global")
        ttk.Combobox(controls, textvariable=self._stage2_baseline_mode, state="readonly", values=["Global", "Aktueller Zeitraum"], width=18).grid(row=0, column=5, padx=4)
        ttk.Label(controls, text="Granularität:").grid(row=0, column=6, padx=4)
        self._stage2_baseline_granularity = tk.StringVar(value=self._config.analysis.baseline.default_granularity)
        ttk.Combobox(controls, textvariable=self._stage2_baseline_granularity, state="readonly", values=["Stunde", "Stunde + Wochentag"], width=18).grid(row=0, column=7, padx=4)
        ttk.Label(controls, text="Wochentag:").grid(row=0, column=8, padx=4)
        self._stage2_baseline_weekday = tk.StringVar(value="Auto")
        ttk.Combobox(controls, textvariable=self._stage2_baseline_weekday, state="readonly", values=self._weekday_values(), width=10).grid(row=0, column=9, padx=4)
        ttk.Button(controls, text="Aktualisieren", command=self._refresh_stage2_plots).grid(row=0, column=10, padx=6)
        ttk.Button(controls, text="i Korrelation", command=self._show_stage2_corr_info).grid(row=0, column=11, padx=3)
        ttk.Button(controls, text="i Darstellung", command=self._show_stage2_view_info).grid(row=0, column=12, padx=3)

        status_frame = ttk.LabelFrame(parent, text="Status", padding=8)
        status_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self._stage2_status_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self._stage2_status_var, justify="left").grid(row=0, column=0, sticky="w")

        charts_frame = ttk.Frame(parent, padding=8)
        charts_frame.grid(row=2, column=0, sticky="ew")
        charts_frame.columnconfigure(0, weight=1)
        charts_frame.columnconfigure(1, weight=1)
        charts_frame.rowconfigure(0, weight=1)

        corr_frame = ttk.LabelFrame(charts_frame, text="Korrelationen", padding=8)
        corr_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        corr_frame.columnconfigure(0, weight=1)
        corr_frame.rowconfigure(0, weight=1)
        self._stage2_corr_figure, self._stage2_corr_axis = plt.subplots(figsize=(5.8, 4.2))
        self._stage2_corr_canvas = FigureCanvasTkAgg(self._stage2_corr_figure, master=corr_frame)
        self._stage2_corr_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        view_frame = ttk.LabelFrame(charts_frame, text="Multivariate Ansicht", padding=8)
        view_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        view_frame.columnconfigure(0, weight=1)
        view_frame.rowconfigure(0, weight=1)
        self._stage2_view_frame = view_frame
        self._stage2_view_figure, self._stage2_view_axis = plt.subplots(figsize=(5.8, 4.2))
        self._stage2_view_figure.subplots_adjust(bottom=0.22, right=0.97, top=0.90, left=0.10)
        self._stage2_view_canvas = FigureCanvasTkAgg(self._stage2_view_figure, master=view_frame)
        self._stage2_view_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        lower = ttk.Frame(parent, padding=8)
        lower.grid(row=3, column=0, sticky="nsew")
        lower.columnconfigure(0, weight=1)
        lower.columnconfigure(1, weight=1)
        lower.rowconfigure(0, weight=1)

        baseline_frame = ttk.LabelFrame(lower, text="Baseline", padding=8)
        baseline_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        baseline_frame.columnconfigure(0, weight=1)
        baseline_frame.rowconfigure(0, weight=1)
        self._stage2_baseline_figure, self._stage2_baseline_axis = plt.subplots(figsize=(5.8, 3.6))
        self._stage2_baseline_canvas = FigureCanvasTkAgg(self._stage2_baseline_figure, master=baseline_frame)
        self._stage2_baseline_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        right_panel, right_inner = self._create_scrollable_block(lower)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        context_frame = ttk.LabelFrame(right_inner, text="Warnungskontext", padding=8)
        context_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        context_frame.columnconfigure(0, weight=1)
        self._stage2_context_var = tk.StringVar(value="")
        ttk.Label(context_frame, textvariable=self._stage2_context_var, justify="left", wraplength=560).grid(row=0, column=0, sticky="w")

        events_frame = ttk.LabelFrame(right_inner, text="Stage 2 Events", padding=8)
        events_frame.grid(row=1, column=0, sticky="nsew")
        events_frame.columnconfigure(0, weight=1)
        events_frame.rowconfigure(0, weight=1)
        self._stage2_events_tree = self._create_event_tree(events_frame)
        self._stage2_events_tree.configure(height=14)
        self._stage2_events_tree.grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------
    def _build_stage3_tab(self) -> None:
        parent_content = self._get_scrollable_content(self._stage3_tab, "_stage3_tab_content")
        parent = parent_content
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=0)
        parent.rowconfigure(3, weight=1)

        controls = ttk.Frame(parent, padding=8)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="Darstellung:").grid(row=0, column=0, padx=4)
        self._stage3_view_mode = tk.StringVar(value="Consensus Score")
        ttk.Combobox(controls, textvariable=self._stage3_view_mode, state="readonly", values=["Consensus Score", "Isolation Forest", "LOF", "One-Class SVM"], width=20).grid(row=0, column=1, padx=4)
        ttk.Label(controls, text="Zeitraum:").grid(row=0, column=2, padx=4)
        self._stage3_timeframe = tk.StringVar(value="Gesamt")
        ttk.Combobox(controls, textvariable=self._stage3_timeframe, state="readonly", values=self._timeframe_values(), width=18).grid(row=0, column=3, padx=4)
        ttk.Label(controls, text="Baseline-Modus:").grid(row=0, column=4, padx=4)
        self._stage3_baseline_mode = tk.StringVar(value="Global")
        ttk.Combobox(controls, textvariable=self._stage3_baseline_mode, state="readonly", values=["Global", "Aktueller Zeitraum"], width=18).grid(row=0, column=5, padx=4)
        ttk.Label(controls, text="Granularität:").grid(row=0, column=6, padx=4)
        self._stage3_baseline_granularity = tk.StringVar(value=self._config.analysis.baseline.default_granularity)
        ttk.Combobox(controls, textvariable=self._stage3_baseline_granularity, state="readonly", values=["Stunde", "Stunde + Wochentag"], width=18).grid(row=0, column=7, padx=4)
        ttk.Label(controls, text="Wochentag:").grid(row=0, column=8, padx=4)
        self._stage3_baseline_weekday = tk.StringVar(value="Auto")
        ttk.Combobox(controls, textvariable=self._stage3_baseline_weekday, state="readonly", values=self._weekday_values(), width=10).grid(row=0, column=9, padx=4)
        ttk.Button(controls, text="Aktualisieren", command=self._refresh_stage3_plots).grid(row=0, column=10, padx=6)
        ttk.Button(controls, text="i Stage 3", command=self._show_stage3_info).grid(row=0, column=11, padx=3)
        ttk.Button(controls, text="i Ampel", command=self._show_stage3_ampel_info).grid(row=0, column=12, padx=3)

        status_frame = ttk.LabelFrame(parent, text="Status", padding=8)
        status_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        status_frame.columnconfigure(0, weight=1)
        self._stage3_status_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self._stage3_status_var, justify="left").grid(row=0, column=0, sticky="w")
        self._stage3_ampel_badge = tk.Label(status_frame, text="Gesamtstatus: -", fg="white", bg="#6b7280", padx=10, pady=6, font=("TkDefaultFont", 10, "bold"))
        self._stage3_ampel_badge.grid(row=0, column=1, sticky="e", padx=(8, 0))

        chart_frame = ttk.LabelFrame(parent, text="Stage 3 Hauptdiagramm", padding=8)
        chart_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=8)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self._stage3_figure, self._stage3_axis = plt.subplots(figsize=(11.4, 3.3))
        self._stage3_figure.subplots_adjust(bottom=0.22, right=0.97, top=0.90, left=0.08)
        self._stage3_canvas = FigureCanvasTkAgg(self._stage3_figure, master=chart_frame)
        self._stage3_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        lower = ttk.Frame(parent, padding=8)
        lower.grid(row=3, column=0, sticky="nsew")
        lower.columnconfigure(0, weight=1)
        lower.columnconfigure(1, weight=1)
        lower.rowconfigure(0, weight=1)

        baseline_frame = ttk.LabelFrame(lower, text="Baseline", padding=8)
        baseline_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        baseline_frame.columnconfigure(0, weight=1)
        baseline_frame.rowconfigure(0, weight=1)
        self._stage3_baseline_figure, self._stage3_baseline_axis = plt.subplots(figsize=(5.8, 3.6))
        self._stage3_baseline_canvas = FigureCanvasTkAgg(self._stage3_baseline_figure, master=baseline_frame)
        self._stage3_baseline_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        right_panel, right_inner = self._create_scrollable_block(lower)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        context_frame = ttk.LabelFrame(right_inner, text="Warnungskontext", padding=8)
        context_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        context_frame.columnconfigure(0, weight=1)
        self._stage3_context_var = tk.StringVar(value="")
        ttk.Label(context_frame, textvariable=self._stage3_context_var, justify="left", wraplength=560).grid(row=0, column=0, sticky="w")

        detail_frame = ttk.LabelFrame(right_inner, text="Stage-3-Eventdetails", padding=8)
        detail_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        self._stage3_detail_text = self._create_scrolled_text(detail_frame, height=8)
        self._stage3_detail_text.grid(row=0, column=0, sticky="nsew")

        events_frame = ttk.LabelFrame(right_inner, text="Stage 3 Events", padding=8)
        events_frame.grid(row=2, column=0, sticky="nsew")
        events_frame.columnconfigure(0, weight=1)
        events_frame.rowconfigure(0, weight=1)
        self._stage3_events_tree = self._create_event_tree(events_frame)
        self._stage3_events_tree.configure(height=14)
        self._stage3_events_tree.grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------
    # Event Inspection
    # ------------------------------------------------------------------
    def _build_inspection_tab(self) -> None:
        self._inspection_tab.columnconfigure(0, weight=1)
        self._inspection_tab.rowconfigure(1, weight=1)
        self._inspection_tab.rowconfigure(2, weight=0)

        top = ttk.Frame(self._inspection_tab, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="Inspektionsfenster [h] vor/nach Event:").grid(row=0, column=0, padx=4)
        self._inspection_window_hours = tk.IntVar(value=self._config.analysis.inspection.window_hours)
        ttk.Spinbox(top, from_=1, to=48, textvariable=self._inspection_window_hours, width=8, command=self._refresh_event_inspection).grid(row=0, column=1, padx=4)
        ttk.Button(top, text="Aus ausgewähltem Event übernehmen", command=self._refresh_event_inspection).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Fenster exportieren", command=self._export_inspection_window).grid(row=0, column=3, padx=4)
        self._inspection_summary_var = tk.StringVar(value="Noch kein Event ausgewählt")
        ttk.Label(top, textvariable=self._inspection_summary_var, justify="left").grid(row=0, column=4, padx=12, sticky="w")

        chart_frame = ttk.LabelFrame(self._inspection_tab, text="Event-Inspektion", padding=8)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self._inspection_figure, (self._inspection_axis_top, self._inspection_axis_bottom) = plt.subplots(2, 1, figsize=(11.5, 6.5), sharex=True)
        self._inspection_figure.subplots_adjust(hspace=0.25, bottom=0.18, top=0.92, left=0.08, right=0.94)
        self._inspection_canvas = FigureCanvasTkAgg(self._inspection_figure, master=chart_frame)
        self._inspection_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        bottom = ttk.Frame(self._inspection_tab, padding=8)
        bottom.grid(row=2, column=0, sticky="nsew")
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(0, weight=1)

        detail_frame = ttk.LabelFrame(bottom, text="Event-Zusammenfassung", padding=8)
        detail_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        self._inspection_detail_text = self._create_scrolled_text(detail_frame, height=10)
        self._inspection_detail_text.grid(row=0, column=0, sticky="nsew")

        notes_frame = ttk.LabelFrame(bottom, text="Inspektionsnotiz", padding=8)
        notes_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        notes_frame.columnconfigure(1, weight=1)
        ttk.Label(notes_frame, text="Bewertung").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self._inspection_verdict = tk.StringVar(value="manuell geprüft")
        ttk.Combobox(notes_frame, textvariable=self._inspection_verdict, state="readonly", values=["manuell geprüft", "wahrscheinlich Lastwechsel", "unklar", "technisch interessant", "mit Wartung abgleichen"], width=28).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(notes_frame, text="Notiz").grid(row=1, column=0, sticky="nw", padx=(0, 8), pady=4)
        self._inspection_notes_text = ScrolledText(notes_frame, wrap="word", height=10)
        self._inspection_notes_text.grid(row=1, column=1, sticky="nsew", pady=4)
        notes_frame.rowconfigure(1, weight=1)
        button_row = ttk.Frame(notes_frame)
        button_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(button_row, text="Notiz speichern", command=self._save_inspection_note).pack(side="left")

    # ------------------------------------------------------------------
    # Labels / Explorer
    # ------------------------------------------------------------------
    def _build_labels_tab(self) -> None:
        parent_content = self._get_scrollable_content(self._labels_tab, "_labels_tab_content")
        parent = parent_content
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        left_frame = ttk.LabelFrame(parent, text="Gespeicherte Defektlabels", padding=8)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        self._labels_tree = self._create_label_tree(left_frame)
        self._labels_tree.grid(row=0, column=0, sticky="nsew")
        self._labels_tree.bind("<<TreeviewSelect>>", self._on_label_tree_selected)

        right_frame = ttk.LabelFrame(parent, text="Label bearbeiten / anlegen", padding=8)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right_frame.columnconfigure(1, weight=1)

        self._label_id_var = tk.StringVar(value="")
        self._label_source_stage_var = tk.StringVar(value="")
        self._label_event_start_var = tk.StringVar(value="")
        self._label_event_end_var = tk.StringVar(value="")
        self._label_defect_type_var = tk.StringVar(value="")
        self._label_repaired_at_var = tk.StringVar(value="")
        self._label_repaired_what_var = tk.StringVar(value="")
        self._label_pre_failure_hours_var = tk.StringVar(value="24")
        self._label_target_metric_var = tk.StringVar(value="next_avg_decibel")
        self._label_notes_var = tk.StringVar(value="")
        self._labels_info_var = tk.StringVar(value="")

        self._add_form_row(right_frame, 0, "Label-ID", self._label_id_var, readonly=True)
        self._add_form_row(right_frame, 1, "Quelle", self._label_source_stage_var)
        self._add_form_row(right_frame, 2, "Event Start", self._label_event_start_var)
        self._add_form_row(right_frame, 3, "Event Ende", self._label_event_end_var)
        self._add_form_row(right_frame, 4, "Defekt", self._label_defect_type_var)
        self._add_form_row(right_frame, 5, "Repariert am", self._label_repaired_at_var)
        self._add_form_row(right_frame, 6, "Was repariert", self._label_repaired_what_var)
        self._add_form_row(right_frame, 7, "Lernfenster vorher [h]", self._label_pre_failure_hours_var)
        ttk.Label(right_frame, text="Zielmetrik").grid(row=8, column=0, sticky="w", pady=4, padx=(0, 8))
        ttk.Combobox(right_frame, textvariable=self._label_target_metric_var, state="readonly", values=["next_avg_decibel", "next_peak_minus_min"], width=36).grid(row=8, column=1, sticky="ew", pady=4)
        ttk.Label(right_frame, text="Notizen").grid(row=9, column=0, sticky="w", pady=4, padx=(0, 8))
        ttk.Entry(right_frame, textvariable=self._label_notes_var).grid(row=9, column=1, sticky="ew", pady=4)

        button_frame = ttk.Frame(right_frame)
        button_frame.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(12, 8))
        ttk.Button(button_frame, text="Aus ausgewähltem Event übernehmen", command=self._fill_label_form_from_selected_event).grid(row=0, column=0, padx=4)
        ttk.Button(button_frame, text="Label speichern", command=self._save_label).grid(row=0, column=1, padx=4)
        ttk.Button(button_frame, text="Label löschen", command=self._delete_selected_label).grid(row=0, column=2, padx=4)
        ttk.Button(button_frame, text="Formular leeren", command=self._clear_label_form).grid(row=0, column=3, padx=4)

        info_frame = ttk.LabelFrame(right_frame, text="Hinweis für spätere Prognose", padding=8)
        info_frame.grid(row=11, column=0, columnspan=2, sticky="ew")
        ttk.Label(info_frame, textvariable=self._labels_info_var, justify="left").grid(row=0, column=0, sticky="w")
        self._labels_info_var.set(
            "Gespeicherte Labels bilden die Basis für spätere Stufe-4-Modelle.\n"
            "Das Lernfenster vorher legt fest, welcher Zeitraum vor dem reparierten Zustand\n"
            "als potenziell degradierter Verlauf für next_avg_decibel oder next_peak_minus_min markiert wird."
        )

    def _build_explorer_tab(self) -> None:
        parent_content = self._get_scrollable_content(self._explorer_tab, "_explorer_tab_content")
        parent = parent_content
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(parent, text="CSV Explorer", padding=8)
        controls.grid(row=0, column=0, sticky="ns", padx=8, pady=8)
        ttk.Label(controls, text="X-Spalte:").grid(row=0, column=0, sticky="w")
        self._explorer_x_column = tk.StringVar(value="timestamp")
        self._explorer_x_combo = ttk.Combobox(controls, textvariable=self._explorer_x_column, state="readonly", width=30)
        self._explorer_x_combo.grid(row=1, column=0, pady=(0, 8), sticky="ew")
        ttk.Label(controls, text="Y-Spalten:").grid(row=2, column=0, sticky="w")
        self._explorer_y_listbox = tk.Listbox(controls, selectmode="multiple", height=22, exportselection=False, width=30)
        self._explorer_y_listbox.grid(row=3, column=0, sticky="nsew")
        ttk.Button(controls, text="Explorer aktualisieren", command=self._refresh_explorer_plot).grid(row=4, column=0, pady=8, sticky="ew")

        chart_frame = ttk.LabelFrame(parent, text="Freie Diagramme", padding=8)
        chart_frame.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self._explorer_figure, self._explorer_axis = plt.subplots(figsize=(10, 6))
        self._explorer_canvas = FigureCanvasTkAgg(self._explorer_figure, master=chart_frame)
        self._explorer_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def _load_and_run_pipeline(self) -> None:
        try:
            self._state.loaded_data = self._data_service.load()
            self._state.stage1_result = self._stage1_service.run(self._state.loaded_data.dataframe)
            self._state.stage2_result = self._stage2_service.run(self._state.stage1_result.dataframe)
            self._state.stage3_result = self._stage3_service.run(self._state.stage2_result.dataframe)
            self._state.labels_df = self._labels_service.load_dataframe()
            self._state.inspection_notes_df = self._inspection_notes_service.load_dataframe()
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
            self._state.stage3_result = self._stage3_service.run(self._state.stage2_result.dataframe)
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
            self._state.stage3_result = self._stage3_service.run(self._state.stage2_result.dataframe)
            self._push_alerts_to_terminal()
            self._refresh_all_views()
        except Exception as error:  # noqa: BLE001
            self._logger.exception("Fehler in Stage 2")
            messagebox.showerror("Fehler", str(error))

    def _rerun_stage3(self) -> None:
        if self._state.stage2_result is None:
            messagebox.showwarning("Hinweis", "Stage 2 wurde noch nicht berechnet.")
            return
        try:
            self._state.stage3_result = self._stage3_service.run(self._state.stage2_result.dataframe)
            self._push_alerts_to_terminal()
            self._refresh_all_views()
        except Exception as error:  # noqa: BLE001
            self._logger.exception("Fehler in Stage 3")
            messagebox.showerror("Fehler", str(error))

    def _reload_labels(self) -> None:
        self._state.labels_df = self._labels_service.load_dataframe()
        self._refresh_labels_tab()
        self._refresh_stage3_plots()

    def _reload_inspection_notes(self) -> None:
        self._state.inspection_notes_df = self._inspection_notes_service.load_dataframe()
        self._refresh_event_inspection()

    def _push_alerts_to_terminal(self) -> None:
        if self._state.stage1_result is None or self._state.stage2_result is None or self._state.stage3_result is None:
            return
        self._logger.info(
            "Pipeline bereit: Stage1=%s Events | Stage2=%s Events | Stage3=%s Events",
            len(self._state.stage1_result.events),
            len(self._state.stage2_result.events),
            len(self._state.stage3_result.events),
        )
        combined_events = pd.concat(
            [
                self._state.stage1_result.events.assign(_priority=1),
                self._state.stage2_result.events.assign(_priority=2),
                self._state.stage3_result.events.assign(_priority=3),
            ],
            ignore_index=True,
        )
        if combined_events.empty:
            return
        top_events = combined_events.sort_values(["start_timestamp", "_priority"], ascending=[False, False]).head(8)
        for _, row in top_events.iterrows():
            self._logger.info(
                "Event | %s | %s | %s bis %s | %s",
                row.get("stage", ""),
                row.get("severity", ""),
                self._format_timestamp(row.get("start_timestamp")),
                self._format_timestamp(row.get("end_timestamp")),
                row.get("message", ""),
            )

    # ------------------------------------------------------------------
    # Refresh orchestration
    # ------------------------------------------------------------------
    def _refresh_all_views(self) -> None:
        if self._state.loaded_data is None or self._state.stage1_result is None or self._state.stage2_result is None or self._state.stage3_result is None:
            return
        self._safe_refresh("Übersicht", self._refresh_overview)
        self._safe_refresh("Stage 1", self._refresh_stage1_plots)
        self._safe_refresh("Stage 2", self._refresh_stage2_plots)
        self._safe_refresh("Stage 3", self._refresh_stage3_plots)
        self._safe_refresh("Defektlabels", self._refresh_labels_tab)
        self._safe_refresh("Explorer Controls", self._refresh_explorer_controls)
        self._safe_refresh("Explorer Plot", self._refresh_explorer_plot)
        self._safe_refresh("Event-Inspektion", self._refresh_event_inspection)

    def _safe_refresh(self, name: str, func) -> None:  # type: ignore[no-untyped-def]
        try:
            func()
        except Exception:  # noqa: BLE001
            self._logger.exception("Fehler beim Aktualisieren von %s", name)

    # ------------------------------------------------------------------
    # Refresh per tab
    # ------------------------------------------------------------------
    def _refresh_overview(self) -> None:
        assert self._state.loaded_data is not None
        assert self._state.stage1_result is not None
        assert self._state.stage2_result is not None
        assert self._state.stage3_result is not None

        loaded = self._state.loaded_data.dataframe
        stage1_result = self._state.stage1_result
        stage2_result = self._state.stage2_result
        stage3_result = self._state.stage3_result

        self._summary_var.set(
            f"Zeilen: {len(loaded):,}\n"
            f"Messlücken: {int(loaded['gap_flag'].sum()):,}\n"
            f"Reduced Quality: {int(loaded['quality_flag'].eq('reduced_quality').sum()):,}\n"
            f"Stage 1 Warnungszeilen: {stage1_result.context_summary.row_alerts_total:,} | Events: {stage1_result.context_summary.events_total:,}\n"
            f"Stage 2 Warnungszeilen: {stage2_result.context_summary.row_alerts_total:,} | Events: {stage2_result.context_summary.events_total:,}\n"
            f"Stage 3 Warnungszeilen: {stage3_result.context_summary.row_alerts_total:,} | Events: {stage3_result.context_summary.events_total:,}"
        )
        self._overview_context_var.set(
            "Stage 1\n" + stage1_result.context_summary.to_multiline_text() + "\n\n"
            + "Stage 2\n" + stage2_result.context_summary.to_multiline_text() + "\n\n"
            + "Stage 3\n" + stage3_result.context_summary.to_multiline_text()
        )

        self._overview_axis.clear()
        plot_df = self._get_plot_df(stage1_result.dataframe, ["avg_decibel", "peak_decibel", "min_decibel"])
        for column in ["avg_decibel", "peak_decibel", "min_decibel"]:
            if column in plot_df.columns:
                self._overview_axis.plot(plot_df["timestamp"], plot_df[column], label=column)
        self._overview_axis.set_title("Basis-Zeitreihe")
        self._overview_axis.set_xlabel("Zeit")
        self._overview_axis.set_ylabel("dB")
        self._overview_axis.legend(loc="upper right")
        self._overview_axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        self._overview_figure.autofmt_xdate()
        self._overview_canvas.draw_idle()
        if hasattr(self, "_overview_scroll"):
            self._overview_scroll._on_content_configure()

        combined_events = pd.concat([stage1_result.events, stage2_result.events, stage3_result.events], ignore_index=True)
        combined_events = combined_events.sort_values("start_timestamp", ascending=False).head(300)
        self._fill_event_tree(self._overview_events_tree, combined_events)

    def _refresh_stage1_plots(self) -> None:
        if self._state.stage1_result is None:
            return
        result = self._state.stage1_result
        metric = self._stage1_metric.get()
        df = result.dataframe
        filtered_df, start_timestamp, end_timestamp = self._filter_df_by_timeframe(df, self._stage1_timeframe.get())
        plot_df = self._get_plot_df(filtered_df, [metric])

        self._stage1_axis.clear()
        if metric in plot_df.columns:
            self._stage1_axis.plot(plot_df["timestamp"], plot_df[metric], label=metric)
            alert_df = plot_df[plot_df["stage1_severity"].isin(["WARNING", "ANOMALY"])]
            if not alert_df.empty:
                self._stage1_axis.scatter(alert_df["timestamp"], alert_df[metric], s=12, c="tab:red", label="Stage1 Alert")
        self._stage1_axis.set_title(f"Stage 1 Verlauf: {metric}")
        self._stage1_axis.set_xlabel("Zeit")
        self._stage1_axis.set_ylabel(metric)
        self._stage1_axis.legend(loc="upper right")
        self._stage1_axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        self._stage1_axis.tick_params(axis="x", labelrotation=20)
        self._stage1_axis.grid(alpha=0.2)
        self._stage1_canvas.draw_idle()

        self._stage1_status_var.set(self._build_status_text(
            timeframe=self._stage1_timeframe.get(),
            baseline_mode=self._stage1_baseline_mode.get(),
            baseline_granularity=self._stage1_baseline_granularity.get(),
            row_count=len(filtered_df),
            event_count=len(filter_events_by_timerange(result.events, start_timestamp, end_timestamp)),
            computed_at=result.computed_at,
        ))

        baseline_source = filtered_df if self._stage1_baseline_mode.get() == "Aktueller Zeitraum" else result.dataframe
        baseline_profile, profile_title = self._build_stage1_profile(baseline_source, metric, self._stage1_baseline_granularity.get(), self._stage1_baseline_weekday.get(), end_timestamp)
        self._plot_profile(self._stage1_baseline_axis, baseline_profile, metric, f"Baseline: {profile_title}")
        self._stage1_baseline_canvas.draw_idle()
        if hasattr(self, "_stage1_scroll"):
            self._stage1_scroll._on_content_configure()

        filtered_events = filter_events_by_timerange(result.events, start_timestamp, end_timestamp)
        filtered_summary = summarize_alert_context(filtered_df, stage_prefix="stage1", events=filtered_events)
        self._stage1_context_var.set(filtered_summary.to_multiline_text() + "\n\n" + result.rule_summary_text)
        self._fill_event_tree(self._stage1_events_tree, filtered_events.sort_values("start_timestamp", ascending=False).head(300))
        self._update_stage1_detail_text()

    def _refresh_stage2_plots(self) -> None:
        if self._state.stage2_result is None:
            return
        result = self._state.stage2_result
        df = result.dataframe
        filtered_df, start_timestamp, end_timestamp = self._filter_df_by_timeframe(df, self._stage2_timeframe.get())
        mode = self._stage2_view_mode.get()

        self._stage2_corr_figure.clf()
        self._stage2_corr_axis = self._stage2_corr_figure.add_subplot(111)
        corr_source = filtered_df.dropna(subset=result.feature_columns) if result.feature_columns else filtered_df
        if corr_source.empty or not result.feature_columns:
            self._stage2_corr_axis.text(0.5, 0.5, "Keine Korrelationsdaten im gewählten Zeitraum", ha="center", va="center")
            self._stage2_corr_axis.set_axis_off()
        else:
            corr_matrix = corr_source[result.feature_columns].corr()
            image = self._stage2_corr_axis.imshow(corr_matrix.values, aspect="auto", vmin=-1.0, vmax=1.0)
            self._stage2_corr_axis.set_xticks(range(len(corr_matrix.columns)))
            self._stage2_corr_axis.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
            self._stage2_corr_axis.set_yticks(range(len(corr_matrix.index)))
            self._stage2_corr_axis.set_yticklabels(corr_matrix.index)
            self._stage2_corr_axis.set_title(f"Korrelationsmatrix | {self._stage2_timeframe.get()}")
            self._stage2_corr_figure.colorbar(image, ax=self._stage2_corr_axis, fraction=0.046, pad=0.04)
        self._stage2_corr_figure.subplots_adjust(bottom=0.28, right=0.96, top=0.92, left=0.18)
        self._stage2_corr_canvas.draw_idle()

        plot_df = self._get_plot_df(filtered_df, ["multivariate_score", "pc1", "pc2"])
        self._stage2_view_axis.clear()
        if mode == "PCA Scatter":
            self._stage2_view_frame.configure(text="PCA Scatter")
            valid_pca_df = plot_df.dropna(subset=["pc1", "pc2"])
            if valid_pca_df.empty:
                self._stage2_view_axis.text(0.5, 0.5, "Keine PCA-Daten verfügbar", ha="center", va="center")
                self._stage2_view_axis.set_axis_off()
            else:
                colors = valid_pca_df["stage2_severity"].map({"NORMAL": "tab:blue", "WARNING": "tab:orange", "ANOMALY": "tab:red"}).fillna("tab:blue")
                self._stage2_view_axis.scatter(valid_pca_df["pc1"], valid_pca_df["pc2"], s=10, c=colors)
                legend_handles = [
                    Line2D([0], [0], marker="o", color="w", label="NORMAL", markerfacecolor="tab:blue", markersize=6),
                    Line2D([0], [0], marker="o", color="w", label="WARNING", markerfacecolor="tab:orange", markersize=6),
                    Line2D([0], [0], marker="o", color="w", label="ANOMALY", markerfacecolor="tab:red", markersize=6),
                ]
                self._stage2_view_axis.legend(handles=legend_handles, loc="upper right")
                self._stage2_view_axis.set_xlabel("PC1")
                self._stage2_view_axis.set_ylabel("PC2")
                self._stage2_view_axis.set_title(f"PCA Scatter | {self._stage2_timeframe.get()}")
                self._stage2_view_axis.grid(alpha=0.2)
        else:
            self._stage2_view_frame.configure(text="Multivariate Ansicht")
            if "multivariate_score" in plot_df.columns:
                self._stage2_view_axis.plot(plot_df["timestamp"], plot_df["multivariate_score"], label="multivariate_score")
                alert_df = plot_df[plot_df["stage2_severity"].isin(["WARNING", "ANOMALY"])]
                if not alert_df.empty:
                    self._stage2_view_axis.scatter(alert_df["timestamp"], alert_df["multivariate_score"], s=12, label="Stage2 Alert")
            self._stage2_view_axis.set_xlabel("Zeit")
            self._stage2_view_axis.set_ylabel("Score")
            self._stage2_view_axis.set_title(f"Multivariate Score | {self._stage2_timeframe.get()}")
            self._stage2_view_axis.legend(loc="upper right")
            self._stage2_view_axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
            self._stage2_view_axis.tick_params(axis="x", labelrotation=20)
            self._stage2_view_axis.grid(alpha=0.2)
        self._stage2_view_canvas.draw_idle()

        self._stage2_status_var.set(self._build_status_text(
            timeframe=self._stage2_timeframe.get(),
            baseline_mode=self._stage2_baseline_mode.get(),
            baseline_granularity=self._stage2_baseline_granularity.get(),
            row_count=len(filtered_df),
            event_count=len(filter_events_by_timerange(result.events, start_timestamp, end_timestamp)),
            computed_at=result.computed_at,
        ))

        self._stage2_baseline_axis.clear()
        if mode == "PCA Scatter":
            self._stage2_baseline_axis.text(0.5, 0.5, "Für PCA ist keine Stunden-Baseline sinnvoll.\nPCA beschreibt Merkmalsräume, nicht einen typischen Tageswert auf einer einzelnen Skala.", ha="center", va="center", wrap=True)
            self._stage2_baseline_axis.set_axis_off()
        else:
            baseline_source = filtered_df if self._stage2_baseline_mode.get() == "Aktueller Zeitraum" else df
            profile, profile_title = self._build_generic_profile(baseline_source, "multivariate_score", self._stage2_baseline_granularity.get(), self._stage2_baseline_weekday.get(), end_timestamp)
            self._plot_profile(self._stage2_baseline_axis, profile, "multivariate_score", f"Baseline: {profile_title}")
        self._stage2_baseline_canvas.draw_idle()
        if hasattr(self, "_stage2_scroll"):
            self._stage2_scroll._on_content_configure()

        filtered_events = filter_events_by_timerange(result.events, start_timestamp, end_timestamp)
        filtered_summary = summarize_alert_context(filtered_df, stage_prefix="stage2", events=filtered_events)
        self._stage2_context_var.set(filtered_summary.to_multiline_text())
        self._fill_event_tree(self._stage2_events_tree, filtered_events.sort_values("start_timestamp", ascending=False).head(300))

    def _refresh_stage3_plots(self) -> None:
        if self._state.stage3_result is None:
            return
        result = self._state.stage3_result
        mode = self._stage3_view_mode.get()
        df = result.dataframe
        filtered_df, start_timestamp, end_timestamp = self._filter_df_by_timeframe(df, self._stage3_timeframe.get())
        score_column_map = {
            "Consensus Score": "stage3_consensus_score",
            "Isolation Forest": "stage3_iforest_score",
            "LOF": "stage3_lof_score",
            "One-Class SVM": "stage3_ocsvm_score",
        }
        score_column = score_column_map.get(mode, "stage3_consensus_score")
        plot_df = self._get_plot_df(filtered_df, [score_column])

        self._stage3_axis.clear()
        if score_column in plot_df.columns:
            self._stage3_axis.plot(plot_df["timestamp"], plot_df[score_column], label=score_column)
            alert_df = plot_df[plot_df["stage3_severity"].isin(["WARNING", "ANOMALY"])]
            if not alert_df.empty:
                colors = alert_df["stage3_ampel_label"].map({
                    "normal": "#2563eb",
                    "lokal auffällig": "#f59e0b",
                    "konsistent auffällig": "#f97316",
                    "persistent anomal": "#dc2626",
                }).fillna("#f59e0b")
                self._stage3_axis.scatter(alert_df["timestamp"], alert_df[score_column], s=16, c=colors, label="Stage3 Eventpunkte")
        if self._state.labels_df is not None and not self._state.labels_df.empty:
            for _, label_row in self._state.labels_df.iterrows():
                start = label_row.get("learning_window_start")
                end = label_row.get("learning_window_end")
                if pd.notna(start) and pd.notna(end):
                    if (start_timestamp is None or end >= start_timestamp) and (end_timestamp is None or start <= end_timestamp):
                        self._stage3_axis.axvspan(start, end, color="grey", alpha=0.08)
                repaired_at = label_row.get("repaired_at")
                if pd.notna(repaired_at):
                    if (start_timestamp is None or repaired_at >= start_timestamp) and (end_timestamp is None or repaired_at <= end_timestamp):
                        self._stage3_axis.axvline(repaired_at, color="grey", alpha=0.25, linestyle="--")

        self._stage3_axis.set_title(f"Stage 3 Verlauf: {mode}")
        self._stage3_axis.set_xlabel("Zeit")
        self._stage3_axis.set_ylabel(self._get_stage3_axis_label(mode))
        self._stage3_axis.legend(loc="upper right")
        self._stage3_axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        self._stage3_axis.tick_params(axis="x", labelrotation=20)
        self._stage3_axis.grid(alpha=0.2)
        self._stage3_canvas.draw_idle()

        filtered_events = filter_events_by_timerange(result.events, start_timestamp, end_timestamp)
        visible_counts = self._count_ampel_labels(filtered_events)
        visible_overall = self._resolve_visible_ampel(filtered_events)
        self._stage3_ampel_badge.configure(text=f"Gesamtstatus: {visible_overall}", bg=self._get_stage3_ampel_color(visible_overall))
        self._stage3_status_var.set(
            self._build_status_text(
                timeframe=self._stage3_timeframe.get(),
                baseline_mode=self._stage3_baseline_mode.get(),
                baseline_granularity=self._stage3_baseline_granularity.get(),
                baseline_weekday=self._stage3_baseline_weekday.get(),
                row_count=len(filtered_df),
                event_count=len(filtered_events),
                computed_at=result.computed_at,
            )
            + f"\nAmpel: normal={visible_counts.get('normal', 0)}, lokal={visible_counts.get('lokal auffällig', 0)}, konsistent={visible_counts.get('konsistent auffällig', 0)}, persistent={visible_counts.get('persistent anomal', 0)}"
        )

        baseline_source = filtered_df if self._stage3_baseline_mode.get() == "Aktueller Zeitraum" else df
        profile, profile_title = self._build_generic_profile(baseline_source, score_column, self._stage3_baseline_granularity.get(), self._stage3_baseline_weekday.get(), end_timestamp)
        self._plot_profile(self._stage3_baseline_axis, profile, score_column, f"Baseline: {profile_title}")
        self._stage3_baseline_canvas.draw_idle()
        if hasattr(self, "_stage3_scroll"):
            self._stage3_scroll._on_content_configure()

        filtered_summary = summarize_alert_context(filtered_df, stage_prefix="stage3", events=filtered_events)
        self._stage3_context_var.set(filtered_summary.to_multiline_text() + f"\n\nTraining: {result.training_row_count} Zeilen | Features: {', '.join(result.feature_columns)}")
        self._fill_event_tree(self._stage3_events_tree, filtered_events.sort_values("start_timestamp", ascending=False).head(300))
        self._update_stage3_detail_text()

    def _refresh_event_inspection(self) -> None:
        if self._selected_event_record is None or self._state.stage3_result is None:
            self._inspection_summary_var.set("Noch kein Event ausgewählt")
            self._clear_axis(self._inspection_axis_top)
            self._clear_axis(self._inspection_axis_bottom)
            self._inspection_canvas.draw_idle()
            self._set_text_widget(self._inspection_detail_text, "Bitte ein Event in Übersicht, Stage 1, Stage 2 oder Stage 3 auswählen.")
            return

        df = self._state.stage3_result.dataframe
        hours = int(self._inspection_window_hours.get())
        subset, window_start, window_end = build_event_window(df, self._selected_event_record, hours_before_after=hours)
        start = pd.to_datetime(self._selected_event_record.get("start_timestamp"))
        end = pd.to_datetime(self._selected_event_record.get("end_timestamp"))

        self._inspection_summary_var.set(
            f"{self._selected_event_record.get('stage', '')} | {self._format_timestamp(start)} bis {self._format_timestamp(end)} | Fenster: {hours}h vor/nach"
        )

        self._inspection_axis_top.clear()
        self._inspection_axis_bottom.clear()
        if not subset.empty:
            self._inspection_axis_top.plot(subset["timestamp"], subset.get("avg_decibel"), label="avg_decibel")
            if "peak_minus_min" in subset.columns:
                self._inspection_axis_top.plot(subset["timestamp"], subset["peak_minus_min"], label="peak_minus_min")
            self._inspection_axis_top.axvspan(start, end, color="orange", alpha=0.15, label="Event")
            self._inspection_axis_top.set_ylabel("Pegel / Range")
            self._inspection_axis_top.legend(loc="upper right")
            self._inspection_axis_top.grid(alpha=0.2)

            if "avg_decibel_robust_z" in subset.columns:
                self._inspection_axis_bottom.plot(subset["timestamp"], subset["avg_decibel_robust_z"], label="stage1 avg_decibel_robust_z")
            if "stage3_consensus_score" in subset.columns:
                self._inspection_axis_bottom.plot(subset["timestamp"], subset["stage3_consensus_score"], label="stage3_consensus_score")
            if "samples_count" in subset.columns:
                ax2 = self._inspection_axis_bottom.twinx()
                ax2.plot(subset["timestamp"], subset["samples_count"], color="tab:gray", alpha=0.35, label="samples_count")
                ax2.set_ylabel("samples_count")
            reduced = subset.loc[subset["quality_flag"].eq("reduced_quality")]
            if not reduced.empty:
                self._inspection_axis_bottom.scatter(reduced["timestamp"], reduced.get("stage3_consensus_score", pd.Series([0]*len(reduced), index=reduced.index)), s=16, c="tab:red", marker="x", label="reduced_quality")
            gaps = subset.loc[subset["near_gap_flag"].fillna(False)]
            for gap_ts in gaps["timestamp"].head(50):
                self._inspection_axis_bottom.axvline(gap_ts, color="tab:purple", alpha=0.1)
            self._inspection_axis_bottom.axvspan(start, end, color="orange", alpha=0.15)
            self._inspection_axis_bottom.set_ylabel("Abweichung / Score")
            self._inspection_axis_bottom.set_xlabel("Zeit")
            self._inspection_axis_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
            self._inspection_axis_bottom.tick_params(axis="x", labelrotation=20)
            self._inspection_axis_bottom.grid(alpha=0.2)
            self._inspection_axis_bottom.legend(loc="upper left")
        else:
            self._inspection_axis_top.text(0.5, 0.5, "Keine Daten im Eventfenster", ha="center", va="center")
            self._inspection_axis_bottom.set_axis_off()
        self._inspection_canvas.draw_idle()

        note_record = self._inspection_notes_service.get_record(
            stage=str(self._selected_event_record.get("stage", "")),
            start_timestamp=self._selected_event_record.get("start_timestamp"),
            end_timestamp=self._selected_event_record.get("end_timestamp"),
        )
        self._inspection_verdict.set(note_record.get("verdict", "manuell geprüft") if note_record else "manuell geprüft")
        self._inspection_notes_text.delete("1.0", "end")
        if note_record:
            self._inspection_notes_text.insert("1.0", str(note_record.get("notes", "")))

        detail_lines = [
            f"Stage: {self._selected_event_record.get('stage', '')}",
            f"Start: {self._format_timestamp(start)}",
            f"Ende: {self._format_timestamp(end)}",
            f"Dauer [min]: {float(self._selected_event_record.get('duration_minutes', 0.0)):.0f}",
            f"Severity: {self._selected_event_record.get('severity', '')}",
            f"Meldung: {self._selected_event_record.get('message', '')}",
            f"Reduced Quality: {'Ja' if bool(self._selected_event_record.get('has_reduced_quality', False)) else 'Nein'}",
            f"Gap-nah: {'Ja' if bool(self._selected_event_record.get('has_near_gap', False)) else 'Nein'}",
        ]
        if "dominant_rule" in self._selected_event_record:
            detail_lines.extend(
                [
                    f"Dominante Regel: {self._selected_event_record.get('dominant_rule', '')}",
                    f"avg_decibel_robust_z_max: {float(self._selected_event_record.get('avg_decibel_robust_z_max', 0.0)):.2f}",
                    f"peak_minus_avg_robust_z_max: {float(self._selected_event_record.get('peak_minus_avg_robust_z_max', 0.0)):.2f}",
                    f"peak_minus_min_robust_z_max: {float(self._selected_event_record.get('peak_minus_min_robust_z_max', 0.0)):.2f}",
                ]
            )
        if "consensus_count" in self._selected_event_record:
            detail_lines.extend(
                [
                    f"Konsens der Modelle: {int(self._selected_event_record.get('consensus_count', 0))}",
                    f"Persistenz: {'Ja' if bool(self._selected_event_record.get('persistent_flag', False)) else 'Nein'}",
                    f"Ampel: {self._selected_event_record.get('ampel_label', '')}",
                    f"Ampelgrund: {self._selected_event_record.get('ampel_reason', '')}",
                    f"Stage3 Score max: {float(self._selected_event_record.get('stage3_consensus_score_max', self._selected_event_record.get('stage3_consensus_score', 0.0))):.3f}",
                ]
            )
        self._set_text_widget(self._inspection_detail_text, "\n".join(detail_lines))

    # ------------------------------------------------------------------
    # Stage-specific detail updates
    # ------------------------------------------------------------------
    def _update_stage1_detail_text(self) -> None:
        record = self._selected_event_record
        if not record or str(record.get("stage", "")) != "Stage 1":
            self._set_text_widget(self._stage1_detail_text, "Bitte ein Stage-1-Event auswählen.")
            return
        text = (
            f"Ursache: {record.get('dominant_rule', '')}\n"
            f"Dauer [min]: {float(record.get('duration_minutes', 0.0)):.0f}\n"
            f"Qualität: {'reduced_quality' if bool(record.get('has_reduced_quality', False)) else 'normal'}\n"
            f"Messlückennähe: {'ja' if bool(record.get('has_near_gap', False)) else 'nein'}\n"
            f"avg_decibel_robust_z_max: {float(record.get('avg_decibel_robust_z_max', 0.0)):.2f}\n"
            f"peak_minus_avg_robust_z_max: {float(record.get('peak_minus_avg_robust_z_max', 0.0)):.2f}\n"
            f"peak_minus_min_robust_z_max: {float(record.get('peak_minus_min_robust_z_max', 0.0)):.2f}\n"
            f"Meldung: {record.get('message', '')}"
        )
        self._set_text_widget(self._stage1_detail_text, text)

    def _update_stage3_detail_text(self) -> None:
        record = self._selected_event_record
        if not record or str(record.get("stage", "")) != "Stage 3":
            self._set_text_widget(self._stage3_detail_text, "Bitte ein Stage-3-Event auswählen.")
            return
        mode = self._stage3_view_mode.get()
        score_key_map = {
            "Consensus Score": "stage3_consensus_score_max",
            "Isolation Forest": "stage3_iforest_score_max",
            "LOF": "stage3_lof_score_max",
            "One-Class SVM": "stage3_ocsvm_score_max",
        }
        score_key = score_key_map.get(mode, "stage3_consensus_score_max")
        text = (
            f"Gewählte Darstellung: {mode}\n"
            f"Max. Score im Event: {float(record.get(score_key, record.get('stage3_consensus_score', 0.0))):.3f}\n"
            f"Konsens der Modelle: {int(record.get('consensus_count', 0))}\n"
            f"Dauer [min]: {float(record.get('duration_minutes', 0.0)):.0f}\n"
            f"Reduced Quality: {'Ja' if bool(record.get('has_reduced_quality', False)) else 'Nein'}\n"
            f"Gap-nah: {'Ja' if bool(record.get('has_near_gap', False)) else 'Nein'}\n"
            f"Ampel: {record.get('ampel_label', '')}\n"
            f"Ampelgrund: {record.get('ampel_reason', '')}\n"
            f"Meldung: {record.get('message', '')}"
        )
        self._set_text_widget(self._stage3_detail_text, text)

    # ------------------------------------------------------------------
    # Labels and inspection notes
    # ------------------------------------------------------------------
    def _fill_label_form_from_selected_event(self) -> None:
        event_row = self._selected_event_record
        if event_row is None:
            messagebox.showwarning("Hinweis", "Bitte zuerst ein Event auswählen.")
            return
        self._label_source_stage_var.set(str(event_row.get("stage", "")))
        self._label_event_start_var.set(self._format_timestamp(event_row.get("start_timestamp")))
        self._label_event_end_var.set(self._format_timestamp(event_row.get("end_timestamp")))
        self._label_repaired_at_var.set(self._format_timestamp(event_row.get("end_timestamp")))
        self._label_id_var.set("")

    def _save_label(self) -> None:
        try:
            event_start = self._parse_label_timestamp(self._label_event_start_var.get())
            event_end = self._parse_label_timestamp(self._label_event_end_var.get())
            repaired_at = self._parse_label_timestamp(self._label_repaired_at_var.get())
            pre_failure_hours = int(self._label_pre_failure_hours_var.get())
        except ValueError as error:
            messagebox.showerror("Fehler", str(error))
            return

        if not self._label_defect_type_var.get().strip():
            messagebox.showwarning("Hinweis", "Bitte einen Defekttyp angeben.")
            return

        self._labels_service.upsert(
            label_id=self._label_id_var.get().strip() or None,
            source_stage=self._label_source_stage_var.get().strip() or "Manuell",
            event_start=event_start.isoformat(),
            event_end=event_end.isoformat(),
            defect_type=self._label_defect_type_var.get().strip(),
            repaired_at=repaired_at.isoformat(),
            repaired_what=self._label_repaired_what_var.get().strip(),
            pre_failure_hours=pre_failure_hours,
            target_metric=self._label_target_metric_var.get().strip(),
            notes=self._label_notes_var.get().strip(),
        )
        self._logger.info("Defektlabel gespeichert: %s", self._label_defect_type_var.get().strip())
        self._state.labels_df = self._labels_service.load_dataframe()
        self._refresh_labels_tab()
        self._refresh_stage3_plots()
        self._clear_label_form()

    def _delete_selected_label(self) -> None:
        label_id = self._label_id_var.get().strip()
        if not label_id:
            messagebox.showwarning("Hinweis", "Bitte zuerst ein gespeichertes Label auswählen.")
            return
        if not messagebox.askyesno("Bestätigen", f"Label {label_id} wirklich löschen?"):
            return
        self._labels_service.delete(label_id)
        self._logger.info("Defektlabel gelöscht: %s", label_id)
        self._state.labels_df = self._labels_service.load_dataframe()
        self._refresh_labels_tab()
        self._refresh_stage3_plots()
        self._clear_label_form()

    def _clear_label_form(self) -> None:
        self._label_id_var.set("")
        self._label_source_stage_var.set("")
        self._label_event_start_var.set("")
        self._label_event_end_var.set("")
        self._label_defect_type_var.set("")
        self._label_repaired_at_var.set("")
        self._label_repaired_what_var.set("")
        self._label_pre_failure_hours_var.set("24")
        self._label_target_metric_var.set("next_avg_decibel")
        self._label_notes_var.set("")

    def _on_label_tree_selected(self, _: object) -> None:
        selected = self._labels_tree.selection()
        if not selected or self._state.labels_df is None:
            return
        index = int(selected[0])
        if index >= len(self._state.labels_df):
            return
        row = self._state.labels_df.iloc[index]
        self._label_id_var.set(str(row.get("label_id", "")))
        self._label_source_stage_var.set(str(row.get("source_stage", "")))
        self._label_event_start_var.set(self._format_timestamp(row.get("event_start")))
        self._label_event_end_var.set(self._format_timestamp(row.get("event_end")))
        self._label_defect_type_var.set(str(row.get("defect_type", "")))
        self._label_repaired_at_var.set(self._format_timestamp(row.get("repaired_at")))
        self._label_repaired_what_var.set(str(row.get("repaired_what", "")))
        self._label_pre_failure_hours_var.set(str(row.get("pre_failure_hours", 24)))
        self._label_target_metric_var.set(str(row.get("target_metric", "next_avg_decibel")))
        self._label_notes_var.set(str(row.get("notes", "")))

    def _save_inspection_note(self) -> None:
        if self._selected_event_record is None:
            messagebox.showwarning("Hinweis", "Bitte zuerst ein Event auswählen.")
            return
        self._inspection_notes_service.upsert(
            stage=str(self._selected_event_record.get("stage", "")),
            start_timestamp=pd.to_datetime(self._selected_event_record.get("start_timestamp")).isoformat(),
            end_timestamp=pd.to_datetime(self._selected_event_record.get("end_timestamp")).isoformat(),
            verdict=self._inspection_verdict.get(),
            notes=self._inspection_notes_text.get("1.0", "end").strip(),
        )
        self._state.inspection_notes_df = self._inspection_notes_service.load_dataframe()
        self._logger.info("Inspektionsnotiz gespeichert für %s", make_event_signature(
            str(self._selected_event_record.get("stage", "")),
            self._selected_event_record.get("start_timestamp"),
            self._selected_event_record.get("end_timestamp"),
        ))

    def _export_inspection_window(self) -> None:
        if self._selected_event_record is None or self._state.stage3_result is None:
            messagebox.showwarning("Hinweis", "Bitte zuerst ein Event auswählen.")
            return
        subset, _, _ = build_event_window(
            self._state.stage3_result.dataframe,
            self._selected_event_record,
            hours_before_after=int(self._inspection_window_hours.get()),
        )
        if subset.empty:
            messagebox.showwarning("Hinweis", "Für das ausgewählte Event sind keine Daten im Fenster verfügbar.")
            return
        target_path = filedialog.asksaveasfilename(
            title="Eventfenster exportieren",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="event_window.csv",
        )
        if not target_path:
            return
        subset.to_csv(target_path, index=False)
        self._logger.info("Eventfenster exportiert: %s", target_path)

    # ------------------------------------------------------------------
    # Explorer
    # ------------------------------------------------------------------
    def _refresh_labels_tab(self) -> None:
        if self._state.labels_df is None:
            self._state.labels_df = self._labels_service.load_dataframe()
        self._fill_label_tree(self._labels_tree, self._state.labels_df)

    def _refresh_explorer_controls(self) -> None:
        source_df = None
        if self._state.stage3_result is not None:
            source_df = self._state.stage3_result.dataframe
        elif self._state.stage2_result is not None:
            source_df = self._state.stage2_result.dataframe
        if source_df is None:
            return
        numeric_columns = [column for column in source_df.columns if pd.api.types.is_numeric_dtype(source_df[column]) and column not in {"segment_id"}]
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
        source_df = None
        if self._state.stage3_result is not None:
            source_df = self._state.stage3_result.dataframe
        elif self._state.stage2_result is not None:
            source_df = self._state.stage2_result.dataframe
        if source_df is None:
            return
        x_column = self._explorer_x_column.get()
        selected_indices = self._explorer_y_listbox.curselection()
        selected_columns = [self._explorer_y_listbox.get(index) for index in selected_indices]
        if not selected_columns:
            selected_columns = ["avg_decibel"] if "avg_decibel" in source_df.columns else []
        if not selected_columns:
            return
        extra_required = [x_column] if x_column != "timestamp" else []
        plot_df = self._get_plot_df(source_df, selected_columns + extra_required)
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
        if hasattr(self, "_explorer_scroll"):
            self._explorer_scroll._on_content_configure()

    # ------------------------------------------------------------------
    # Tree widgets and selection
    # ------------------------------------------------------------------
    def _create_event_tree(self, parent: ttk.Frame) -> ttk.Treeview:
        columns = ("start", "end", "duration", "stage", "severity", "ampel", "rows", "rq", "gap", "message")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=16)
        headings = {
            "start": "Start",
            "end": "Ende",
            "duration": "Dauer [min]",
            "stage": "Stufe",
            "severity": "Schwere",
            "ampel": "Ampel",
            "rows": "Zeilen",
            "rq": "RQ",
            "gap": "Gap",
            "message": "Meldung",
        }
        widths = {
            "start": 130,
            "end": 130,
            "duration": 90,
            "stage": 70,
            "severity": 90,
            "ampel": 140,
            "rows": 60,
            "rq": 45,
            "gap": 45,
            "message": 360,
        }
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="w")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")
        tree.bind("<<TreeviewSelect>>", self._on_event_tree_selected)
        return tree

    def _on_event_tree_selected(self, event) -> None:  # type: ignore[no-untyped-def]
        tree = event.widget
        selected = tree.selection()
        records = getattr(tree, "_event_records", {})
        if selected and selected[0] in records:
            self._selected_event_record = records[selected[0]]
            self._update_stage1_detail_text()
            self._update_stage3_detail_text()
            self._refresh_event_inspection()

    @staticmethod
    def _fill_event_tree(tree: ttk.Treeview, events: pd.DataFrame) -> None:
        for item in tree.get_children():
            tree.delete(item)
        tree._event_records = {}  # type: ignore[attr-defined]
        if events.empty:
            return
        for index, (_, row) in enumerate(events.iterrows()):
            iid = str(index)
            tree._event_records[iid] = row.to_dict()  # type: ignore[attr-defined]
            tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    PdmEdgeApplication._format_timestamp(row.get("start_timestamp")),
                    PdmEdgeApplication._format_timestamp(row.get("end_timestamp")),
                    f"{float(row.get('duration_minutes', 0.0)):.0f}",
                    row.get("stage", ""),
                    row.get("severity", ""),
                    row.get("ampel_label", ""),
                    row.get("row_count", ""),
                    "Ja" if bool(row.get("has_reduced_quality", False)) else "Nein",
                    "Ja" if bool(row.get("has_near_gap", False)) else "Nein",
                    row.get("message", ""),
                ),
            )

    @staticmethod
    def _create_label_tree(parent: ttk.Frame) -> ttk.Treeview:
        columns = ("id", "start", "end", "repair", "target", "defect", "hours", "source")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=12)
        headings = {
            "id": "Label-ID",
            "start": "Event Start",
            "end": "Event Ende",
            "repair": "Repariert am",
            "target": "Ziel",
            "defect": "Defekt",
            "hours": "Vorher [h]",
            "source": "Quelle",
        }
        widths = {"id": 120, "start": 130, "end": 130, "repair": 130, "target": 150, "defect": 180, "hours": 85, "source": 90}
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], anchor="w")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")
        return tree

    @staticmethod
    def _fill_label_tree(tree: ttk.Treeview, labels_df: pd.DataFrame) -> None:
        for item in tree.get_children():
            tree.delete(item)
        if labels_df.empty:
            return
        for index, (_, row) in enumerate(labels_df.iterrows()):
            tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    row.get("label_id", ""),
                    PdmEdgeApplication._format_timestamp(row.get("event_start")),
                    PdmEdgeApplication._format_timestamp(row.get("event_end")),
                    PdmEdgeApplication._format_timestamp(row.get("repaired_at")),
                    row.get("target_metric", ""),
                    row.get("defect_type", ""),
                    row.get("pre_failure_hours", ""),
                    row.get("source_stage", ""),
                ),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _timeframe_values() -> list[str]:
        return ["Gesamt", "Letzte 24h", "Letzte 7 Tage", "Letzte 30 Tage", "Letzte 90 Tage", "Letzter Kalendertag"]

    @staticmethod
    def _weekday_values() -> list[str]:
        return ["Auto", "Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

    @staticmethod
    def _add_form_row(parent: ttk.Frame, row: int, label_text: str, variable: tk.StringVar, readonly: bool = False) -> None:
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 8))
        state = "readonly" if readonly else "normal"
        ttk.Entry(parent, textvariable=variable, state=state).grid(row=row, column=1, sticky="ew", pady=4)

    @staticmethod
    def _create_scrolled_text(parent: ttk.Frame, *, height: int) -> ScrolledText:
        widget = ScrolledText(parent, wrap="word", height=height, font=("Arial", 10))
        widget.configure(state="disabled")
        return widget

    @staticmethod
    def _create_scrollable_block(parent: ttk.Frame) -> tuple[ttk.Frame, ttk.Frame]:
        outer = ttk.Frame(parent)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.columnconfigure(0, weight=1)

        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(window_id, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        return outer, inner

    def _filter_df_by_timeframe(self, df: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
        working_df = df.dropna(subset=["timestamp"]).copy()
        if working_df.empty or timeframe == "Gesamt":
            return working_df, None, None
        max_timestamp = pd.to_datetime(working_df["timestamp"].max())
        if timeframe == "Letzte 24h":
            start_timestamp = max_timestamp - pd.Timedelta(hours=24)
            return working_df.loc[working_df["timestamp"] >= start_timestamp].copy(), start_timestamp, max_timestamp
        if timeframe == "Letzte 7 Tage":
            start_timestamp = max_timestamp - pd.Timedelta(days=7)
            return working_df.loc[working_df["timestamp"] >= start_timestamp].copy(), start_timestamp, max_timestamp
        if timeframe == "Letzte 30 Tage":
            start_timestamp = max_timestamp - pd.Timedelta(days=30)
            return working_df.loc[working_df["timestamp"] >= start_timestamp].copy(), start_timestamp, max_timestamp
        if timeframe == "Letzte 90 Tage":
            start_timestamp = max_timestamp - pd.Timedelta(days=90)
            return working_df.loc[working_df["timestamp"] >= start_timestamp].copy(), start_timestamp, max_timestamp
        if timeframe == "Letzter Kalendertag":
            last_day = max_timestamp.normalize()
            end_timestamp = last_day + pd.Timedelta(days=1)
            filtered = working_df.loc[(working_df["timestamp"] >= last_day) & (working_df["timestamp"] < end_timestamp)].copy()
            return filtered, last_day, end_timestamp
        return working_df, None, None

    def _get_plot_df(self, df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
        base_columns = [
            "timestamp",
            "stage1_severity",
            "stage2_severity",
            "stage3_severity",
            "stage3_ampel_label",
            "quality_flag",
            "near_gap_flag",
        ]
        existing_columns = [column for column in base_columns + required_columns if column in df.columns]
        existing_columns = list(dict.fromkeys(existing_columns))
        required_existing = [column for column in required_columns if column in df.columns]
        working_df = df[existing_columns].dropna(subset=required_existing).copy()
        max_points = self._config.analysis.plot.max_points_per_series
        if len(working_df) <= max_points:
            return working_df
        step = max(1, len(working_df) // max_points)
        return working_df.iloc[::step, :].copy()

    @staticmethod
    def _resolve_selected_weekday(selected_weekday: str, df: pd.DataFrame, end_timestamp: pd.Timestamp | None) -> int | None:
        mapping = {"Mo": 0, "Di": 1, "Mi": 2, "Do": 3, "Fr": 4, "Sa": 5, "So": 6}
        selected = (selected_weekday or "Auto").strip()
        if selected in mapping:
            return mapping[selected]
        working_df = df.dropna(subset=["timestamp", "weekday"]).copy()
        if working_df.empty:
            return None
        if end_timestamp is None:
            return int(working_df["weekday"].iloc[-1])
        return int(pd.to_datetime(end_timestamp).weekday())

    def _build_stage1_profile(self, df: pd.DataFrame, metric: str, granularity: str, selected_weekday: str, end_timestamp: pd.Timestamp | None) -> tuple[pd.DataFrame, str]:
        baseline_metric = metric if metric in ("avg_decibel", "peak_minus_avg", "peak_minus_min") else "avg_decibel"
        if granularity == "Stunde + Wochentag":
            profile = self._stage1_service.build_baseline_table(df, granularity="Stunde + Wochentag")
            if profile.empty:
                return profile, f"{baseline_metric} | {granularity}"

            active_weekday = self._resolve_selected_weekday(selected_weekday, df, end_timestamp)
            if active_weekday is None:
                return profile, f"{baseline_metric} | {granularity}"

            day_profile = profile.loc[profile["weekday"] == active_weekday].copy()
            if day_profile.empty or int(day_profile["baseline_count"].max()) < self._config.analysis.baseline.weekday_hour_min_count:
                fallback = self._stage1_service.build_baseline_table(df, granularity="Stunde")
                return fallback, f"{baseline_metric} | Stunde (Fallback)"
            return day_profile, f"{baseline_metric} | Stunde + Wochentag ({self._weekday_name(active_weekday)})"

        return self._stage1_service.build_baseline_table(df, granularity="Stunde"), f"{baseline_metric} | Stunde"

    @staticmethod
    def _plot_profile(axis, profile: pd.DataFrame, metric_label: str, title: str) -> None:  # type: ignore[no-untyped-def]
        axis.clear()
        axis.set_axis_on()
        if profile.empty:
            axis.text(0.5, 0.5, "Keine Baseline-Daten verfügbar", ha="center", va="center")
            axis.set_axis_off()
            return
        if f"{metric_label}_baseline_median" in profile.columns:
            median_col = f"{metric_label}_baseline_median"
            low_col = f"{metric_label}_baseline_mad"
            axis.plot(profile["hour_of_day"], profile[median_col], marker="o", label="Median")
            if low_col in profile.columns:
                spread = 1.4826 * profile[low_col].fillna(0.0)
                axis.fill_between(profile["hour_of_day"], profile[median_col] - spread, profile[median_col] + spread, alpha=0.15, label="±1.48*MAD")
        else:
            axis.plot(profile["hour_of_day"], profile["median"], marker="o", label="Median")
            if "q10" in profile.columns and "q90" in profile.columns:
                axis.fill_between(profile["hour_of_day"], profile["q10"], profile["q90"], alpha=0.15, label="10-90%")
        axis.set_title(title)
        axis.set_xlabel("Stunde")
        axis.set_ylabel(metric_label)
        axis.set_xticks(range(0, 24, 2))
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")

    @staticmethod
    def _weekday_name(weekday: int) -> str:
        return ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"][int(weekday) % 7]

    @staticmethod
    def _clear_axis(axis) -> None:  # type: ignore[no-untyped-def]
        axis.clear()
        axis.set_axis_off()

    # ------------------------------------------------------------------
    # Info dialogs
    # ------------------------------------------------------------------
    def _show_stage1_baseline_info(self) -> None:
        self._show_info_dialog(
            "Info: Stage 1 Baseline",
            "Stage 1 vergleicht Messwerte mit einer Baseline.\n\n"
            "Baseline-Modus:\n"
            "- Global: Baseline über die gesamte Historie.\n"
            "- Aktueller Zeitraum: Baseline nur aus dem gerade betrachteten Ausschnitt.\n\n"
            "Granularität:\n"
            "- Stunde: typischer Verlauf nur nach Tagesstunde.\n"
            "- Stunde + Wochentag: typischer Verlauf je Wochentag und Stunde.\n"
            "  Wenn pro Slot zu wenig Daten vorhanden sind, wird automatisch auf Stunde zurückgefallen.\n\n"
            "Zielbild: eine Baseline, die den normalen Tagesgang gut beschreibt, damit Warnungen echte Abweichungen und nicht nur Tagesmuster zeigen."
        )

    def _show_stage1_level_info(self) -> None:
        self._show_info_dialog(
            "Info: Stage 1 Pegelabweichung",
            "Die Pegelregel betrachtet avg_decibel relativ zur Baseline.\n\n"
            "Wichtige Kennzahl:\n"
            "- avg_decibel_robust_z\n\n"
            "Interpretation:\n"
            "- Werte nahe 0: typisch\n"
            "- größere positive oder negative Beträge: stärkerer Abstand zur Baseline\n"
            "- bei dauerhaft erhöhten Beträgen wird aus WARNING eine ANOMALY\n\n"
            "Zielbild: ruhiger Verlauf ohne längere Phasen stark erhöhter robuster Z-Werte."
        )

    def _show_stage1_range_info(self) -> None:
        self._show_info_dialog(
            "Info: Stage 1 Spike / Spannweite",
            "Stage 1 nutzt zusätzlich Dynamikregeln.\n\n"
            "- peak_minus_avg: wie stark Spitzen über dem Durchschnitt liegen\n"
            "- peak_minus_min: Spannweite im Messfenster\n\n"
            "Interpretation:\n"
            "- erhöhte peak_minus_avg_robust_z-Werte: impulsartigeres Verhalten / stärkere Peaks\n"
            "- erhöhte peak_minus_min_robust_z-Werte: größere Gesamtschwankung\n\n"
            "Zielbild: keine langen Phasen mit gleichzeitig erhöhter Peak-Dynamik oder Spannweite."
        )

    def _show_stage2_corr_info(self) -> None:
        self._show_info_dialog(
            "Info: Korrelationsmatrix",
            "Die Korrelationsmatrix zeigt lineare Zusammenhänge zwischen den Merkmalen.\n\n"
            "Werte nahe +1 bedeuten: zwei Merkmale steigen typischerweise gemeinsam.\n"
            "Werte nahe -1 bedeuten: wenn das eine Merkmal steigt, fällt das andere eher.\n"
            "Werte nahe 0 bedeuten: kein klarer linearer Zusammenhang.\n\n"
            "Nutzen: Du erkennst, welche Größen zusammenlaufen, z. B. ob ein hoher Pegel auch mit größerer Spannweite oder Peak-Dynamik einhergeht."
        )

    def _show_stage2_view_info(self) -> None:
        if self._stage2_view_mode.get() == "PCA Scatter":
            self._show_info_dialog(
                "Info: PCA Scatter",
                "Der PCA Scatter reduziert mehrere Merkmale auf zwei Hauptachsen (PC1 und PC2).\n\n"
                "Bedeutung:\n"
                "- Dicht beieinander liegende Punkte haben ähnliche Merkmalskombinationen.\n"
                "- Abgesetzte Punktwolken deuten auf andere Betriebsregime hin.\n"
                "- Einzelne entfernte Punkte sprechen für ungewöhnliche Zeitfenster.\n\n"
                "Was du sehen möchtest:\n"
                "- eine oder wenige kompakte Wolken\n"
                "- keine vielen weit verstreuten Ausreißer\n"
                "- nur wenige orange/rote Punkte außerhalb der Hauptwolke\n"
            )
        else:
            self._show_info_dialog(
                "Info: Multivariate Score",
                "Der Multivariate Score beschreibt, wie weit ein Messfenster insgesamt vom typischen Merkmalszentrum entfernt liegt.\n\n"
                "Niedrige Werte sind typischer, höhere Werte ungewöhnlicher.\n"
                "Die Skala ist relativ zum Datensatz und nicht physikalisch.\n\n"
                "Was du sehen möchtest:\n"
                "- überwiegend ruhiger Verlauf\n"
                "- nur wenige kurze Peaks\n"
                "- keine langen Phasen mit erhöhtem Score"
            )

    def _show_stage3_info(self) -> None:
        mode = self._stage3_view_mode.get()
        messages = {
            "Consensus Score": "Der Consensus Score ist der gemeinsame Screening-Score aus Isolation Forest, LOF und One-Class SVM.\n\n0 bis ca. 0,3: eher typisch\n0,3 bis 0,7: Übergangsbereich\nab ca. 0,7: deutlich ungewöhnlich\n\nZielbild: ruhiger Verlauf mit wenigen kurzen Ausschlägen.",
            "Isolation Forest": "Isolation Forest isoliert ungewöhnliche Punkte über zufällige Teilungen.\n\nWichtig: In dieser App ist ein niedrigerer / negativerer Wert typischer. Richtung 0 und darüber wird auffälliger.\n\nZielbild: keine langen Phasen mit erhöhten Werten.",
            "LOF": "LOF vergleicht die lokale Dichte eines Punkts mit seiner Nachbarschaft.\n\nHöher bedeutet: lokal untypischer.\nDie Skala ist datensatzabhängig.\n\nZielbild: nur wenige isolierte Peaks, keine langen erhöhten Phasen.",
            "One-Class SVM": "One-Class SVM lernt eine Normalregion.\n\nNiedrige Werte liegen näher an der Normalregion, höhere Werte weiter außerhalb.\nDie Skala ist modell- und datensatzabhängig.\n\nZielbild: wenige kurze Überschreitungen und keine langen zusammenhängenden Ausreißerphasen.",
        }
        self._show_info_dialog(f"Info: Stage 3 – {mode}", messages.get(mode, ""))

    def _show_stage3_ampel_info(self) -> None:
        events = pd.DataFrame()
        if self._state.stage3_result is not None:
            df = self._state.stage3_result.dataframe
            _, start_timestamp, end_timestamp = self._filter_df_by_timeframe(df, self._stage3_timeframe.get())
            events = filter_events_by_timerange(self._state.stage3_result.events, start_timestamp, end_timestamp)
        counts = self._count_ampel_labels(events)
        overall = self._resolve_visible_ampel(events)
        reason = self._build_stage3_ampel_reason(events)
        total_events = int(len(events))
        total_rows = int(events["row_count"].sum()) if not events.empty and "row_count" in events.columns else 0
        self._show_info_dialog(
            "Info: Stage 3 – Ampelbewertung",
            "Die Ampel ist eine Verdichtung der unüberwachten Stage-3-Ergebnisse.\n\n"
            "Sie berücksichtigt:\n"
            "- Konsens der Modelle\n"
            "- Persistenz\n"
            "- Eventdauer\n"
            "- Abschläge bei Reduced Quality oder Gap-Nähe\n\n"
            "Bedeutung:\n"
            "- normal\n- lokal auffällig\n- konsistent auffällig\n- persistent anomal\n\n"
            f"Aktueller Gesamtstatus: {overall}\n"
            f"Verteilung: normal={counts.get('normal', 0)}, lokal={counts.get('lokal auffällig', 0)}, konsistent={counts.get('konsistent auffällig', 0)}, persistent={counts.get('persistent anomal', 0)}\n"
            f"Events gesamt: {total_events} | zugrunde liegende Warnungszeilen: {total_rows}\n\n"
            f"Warum aktuell diese Farbe? {reason}\n\n"
            "Wichtig: Rot bedeutet nicht automatisch sicherer Defekt, sondern ein priorisierter Prüfhinweis."
        )

    def _show_info_dialog(self, title: str, message: str) -> None:
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.transient(self)
        dialog.geometry("760x560")
        dialog.minsize(560, 360)
        container = ttk.Frame(dialog, padding=12)
        container.pack(fill="both", expand=True)
        text_widget = ScrolledText(container, wrap="word", font=("Arial", 10))
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", message)
        text_widget.configure(state="disabled")
        ttk.Button(container, text="Schließen", command=dialog.destroy).pack(anchor="e", pady=(10, 0))
        dialog.focus_set()
        dialog.grab_set()

    # ------------------------------------------------------------------
    # Small helper texts / formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _set_text_widget(widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")


    def _resolve_weekday_selection(
        self,
        df: pd.DataFrame,
        selected_weekday: str,
        end_timestamp: pd.Timestamp | None,
    ) -> tuple[int | None, str]:
        weekday_map = {
            "Mo": 0,
            "Di": 1,
            "Mi": 2,
            "Do": 3,
            "Fr": 4,
            "Sa": 5,
            "So": 6,
        }

        if selected_weekday in weekday_map:
            return weekday_map[selected_weekday], selected_weekday

        working_df = df.dropna(subset=["timestamp"]).copy()
        if working_df.empty:
            return None, "Auto"

        if end_timestamp is None:
            weekday_index = int(working_df["weekday"].dropna().iloc[-1])
        else:
            weekday_index = int(pd.to_datetime(end_timestamp).weekday())

        reverse_map = {value: key for key, value in weekday_map.items()}
        return weekday_index, reverse_map.get(weekday_index, "Auto")

    def _build_status_text(
        self,
        *,
        timeframe: str,
        baseline_mode: str,
        baseline_granularity: str,
        row_count: int,
        event_count: int,
        baseline_weekday: str = "Auto",
        computed_at: pd.Timestamp | None = None,
        last_run_text: str | None = None,
    ) -> str:
        if last_run_text is not None and str(last_run_text).strip():
            computed_text = str(last_run_text)
        elif computed_at is not None and not pd.isna(computed_at):
            computed_text = self._format_timestamp(computed_at)
        else:
            computed_text = "-"

        return (
            f"Zeitraum: {timeframe}\n"
            f"Baseline: {baseline_mode}\n"
            f"Granularität: {baseline_granularity}\n"
            f"Wochentag: {baseline_weekday}\n"
            f"Anzahl Zeilen: {row_count}\n"
            f"Anzahl Events: {event_count}\n"
            f"Letzte Berechnung: {computed_text}"
        )

    def _build_generic_profile(
        self,
        df: pd.DataFrame,
        metric: str,
        granularity: str,
        selected_weekday: str,
        end_timestamp: pd.Timestamp | None,
    ) -> tuple[pd.DataFrame, str]:
        if metric not in df.columns:
            return pd.DataFrame(columns=["hour_of_day", "median", "q10", "q90", "count"]), "Stunde"

        working_df = df.dropna(subset=[metric, "hour_of_day"]).copy()
        if working_df.empty:
            return pd.DataFrame(columns=["hour_of_day", "median", "q10", "q90", "count"]), "Stunde"

        if granularity != "Stunde + Wochentag" or "weekday" not in working_df.columns:
            return self._build_hourly_profile(working_df, metric), "Stunde"

        weekday_index, weekday_label = self._resolve_weekday_selection(
            working_df,
            selected_weekday,
            end_timestamp,
        )

        if weekday_index is None:
            return self._build_hourly_profile(working_df, metric), "Stunde"

        weekday_df = working_df.loc[working_df["weekday"] == weekday_index].copy()

        min_points = 10
        if weekday_df.empty or len(weekday_df) < min_points:
            return self._build_hourly_profile(working_df, metric), "Stunde (Fallback)"

        return self._build_hourly_profile(weekday_df, metric), f"Stunde + Wochentag ({weekday_label})"

    @staticmethod
    def _get_stage3_axis_label(mode: str) -> str:
        return "Normierter Score" if mode == "Consensus Score" else "Modellscore"

    @staticmethod
    def _get_stage3_ampel_color(label: str) -> str:
        return {
            "normal": "#16a34a",
            "lokal auffällig": "#f59e0b",
            "konsistent auffällig": "#f97316",
            "persistent anomal": "#dc2626",
        }.get(label, "#6b7280")


    def _build_hourly_profile(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        if metric not in df.columns or "hour_of_day" not in df.columns:
            return pd.DataFrame(columns=["hour_of_day", "median", "q10", "q90", "count"])

        working_df = df.dropna(subset=[metric, "hour_of_day"]).copy()
        if working_df.empty:
            return pd.DataFrame(columns=["hour_of_day", "median", "q10", "q90", "count"])

        grouped = working_df.groupby("hour_of_day")[metric]
        profile = pd.DataFrame({"hour_of_day": sorted(working_df["hour_of_day"].dropna().unique())})
        profile = profile.merge(grouped.median().reset_index(name="median"), on="hour_of_day", how="left")
        profile = profile.merge(grouped.quantile(0.10).reset_index(name="q10"), on="hour_of_day", how="left")
        profile = profile.merge(grouped.quantile(0.90).reset_index(name="q90"), on="hour_of_day", how="left")
        profile = profile.merge(grouped.count().reset_index(name="count"), on="hour_of_day", how="left")
        return profile.sort_values("hour_of_day").reset_index(drop=True)

    @staticmethod
    def _plot_hourly_profile(axis, profile: pd.DataFrame, *, metric_label: str, title: str) -> None:
        axis.clear()
        if profile.empty:
            axis.text(0.5, 0.5, "Keine Baseline-Daten verfügbar", ha="center", va="center")
            axis.set_axis_off()
            return

        axis.set_axis_on()
        axis.plot(profile["hour_of_day"], profile["median"], marker="o", label="Median")
        if "q10" in profile.columns and "q90" in profile.columns:
            axis.fill_between(profile["hour_of_day"], profile["q10"], profile["q90"], alpha=0.15, label="10-90%")
        axis.set_title(title)
        axis.set_xlabel("Stunde")
        axis.set_ylabel(metric_label)
        axis.set_xticks(range(0, 24, 2))
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")

    @staticmethod
    def _count_ampel_labels(events: pd.DataFrame) -> dict[str, int]:
        labels = ["normal", "lokal auffällig", "konsistent auffällig", "persistent anomal"]
        if events.empty or "ampel_label" not in events.columns:
            return {label: 0 for label in labels}
        return {label: int(events["ampel_label"].eq(label).sum()) for label in labels}

    @staticmethod
    def _resolve_visible_ampel(events: pd.DataFrame) -> str:
        ranking = {"normal": 0, "lokal auffällig": 1, "konsistent auffällig": 2, "persistent anomal": 3}
        if events.empty or "ampel_label" not in events.columns:
            return "normal"
        max_rank = max(ranking.get(str(label), 0) for label in events["ampel_label"].dropna().astype(str))
        for label, rank in ranking.items():
            if rank == max_rank:
                return label
        return "normal"

    def _build_stage3_ampel_reason(self, events: pd.DataFrame) -> str:
        if events.empty:
            return "Im gewählten Zeitraum gibt es keine Stage-3-Events."
        counts = self._count_ampel_labels(events)
        overall = self._resolve_visible_ampel(events)
        discounted_normal = counts.get("normal", 0)
        if overall == "persistent anomal":
            reason = f"Mindestens ein Event wurde als persistent anomal eingestuft ({counts.get('persistent anomal', 0)} Events)."
        elif overall == "konsistent auffällig":
            reason = f"Es gibt keine roten Events, aber {counts.get('konsistent auffällig', 0)} konsistent auffällige Events."
        elif overall == "lokal auffällig":
            reason = f"Es gibt nur lokale Auffälligkeiten ({counts.get('lokal auffällig', 0)} Events), aber keine konsistenten oder persistenten roten Phasen."
        else:
            reason = "Die vorhandenen Events wurden nach Abschlägen eher als normal eingestuft."
        if discounted_normal > 0:
            reason += f" {discounted_normal} Event(s) wurden durch Reduced Quality oder Gap-Nähe auf normal abgewertet."
        return reason

    @staticmethod
    def _format_timestamp(value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        return pd.to_datetime(value).strftime("%d.%m.%Y %H:%M")

    @staticmethod
    def _parse_label_timestamp(text: str) -> pd.Timestamp:
        parsed = pd.to_datetime(text.strip(), format="%d.%m.%Y %H:%M", errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Zeitstempel ungültig: {text}. Erwartet wird TT.MM.JJJJ HH:MM")
        return parsed
