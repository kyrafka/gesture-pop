from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import joblib
import numpy as np

from app_config import (
    IMAGE_DIR,
    IMAGE_SUFFIXES,
    AppConfig,
    load_config,
    load_gesture_map,
    save_gesture_map,
)
from gesture_features import FeatureResult, LandmarkFeatureExtractor, draw_landmarks
from gesture_runtime import FeatureStabilityTracker
from guided_capture import CaptureTarget, build_capture_targets
from reference_images import (
    ReferenceAnalysis,
    analyze_reference,
    load_reference_records,
    mark_reference_not_training,
    store_reference,
)
from train_gestures import (
    CAPTURE_DIR,
    MANIFEST_FILE,
    MODEL_FILE,
    TrainingSampleRecord,
    append_manifest,
    append_sample,
    create_sample_id,
    load_sample_records,
    load_sample_counts,
    remove_sample_record,
    remove_last_sample_with_id,
    save_capture_frame,
    train_model,
)


ROOT = Path(__file__).parent
BG = "#101419"
PANEL = "#171c22"
PANEL_ALT = "#20262e"
TEXT = "#f4f6f8"
MUTED = "#9ba7b4"
GREEN = "#48c774"
AMBER = "#f2b84b"
RED = "#e25b61"
CYAN = "#55c2da"
BORDER = "#303842"


class GestureStudio:
    def __init__(
        self,
        root: tk.Tk,
        config: AppConfig,
        gesture_map: dict[str, Path],
        extractor: LandmarkFeatureExtractor,
    ) -> None:
        self.root = root
        self.config = config
        self.gesture_map = gesture_map
        self.extractor = extractor
        self.cap: cv2.VideoCapture | None = None
        self.selected_label: str | None = next(iter(gesture_map), None)
        self.sample_counts = load_sample_counts(list(gesture_map))
        self.photo_counts = {label: len(capture_records(label)) for label in gesture_map}
        reference_records = load_reference_records()
        self.reference_counts = {
            label: sum(record.label == label for record in reference_records)
            for label in gesture_map
        }
        self.reference_training_counts = {
            label: sum(record.label == label and record.used_for_training for record in reference_records)
            for label in gesture_map
        }
        self.stability = FeatureStabilityTracker(
            config.capture_stability_frames,
            config.capture_stability_threshold,
        )
        self.latest_result: FeatureResult | None = None
        self.latest_capture_frame: np.ndarray | None = None
        self.ready = False
        self.last_capture_time = float("-inf")
        self.guided_active = False
        self.guided_label: str | None = None
        self.guided_targets: list[CaptureTarget] = []
        self.guided_index = 0
        self.guided_next_capture = float("inf")
        self.closing = False
        self.camera_photo: tk.PhotoImage | None = None
        self.selected_photo: tk.PhotoImage | None = None
        self.last_photo: tk.PhotoImage | None = None
        self.selected_placeholder = solid_photo(240, 150, "#0c1014")
        self.last_placeholder = solid_photo(240, 108, "#0c1014")
        self.gesture_photos: dict[str, tk.PhotoImage] = {}
        self.gesture_buttons: dict[str, tk.Button] = {}
        self.hand_telemetry_canvases: list[tk.Canvas] = []

        self.status_var = tk.StringVar(value="Iniciando camara...")
        self.notice_var = tk.StringVar(value="Agrega una imagen o selecciona un gesto para comenzar.")
        self.selected_var = tk.StringVar(value="Sin gesto seleccionado")
        self.count_var = tk.StringVar(value="0 muestras")
        self.guided_var = tk.StringVar(value="Captura manual")
        self.model_var = tk.StringVar(value=self._model_status())

        self._configure_window()
        self._build_menu()
        self._build_layout()
        self._rebuild_gesture_list()
        self._refresh_selected_panel()
        self._open_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(20, self._update_camera)

    def _configure_window(self) -> None:
        self.root.title("Gesture Studio")
        self.root.geometry("1400x860")
        self.root.minsize(1120, 720)
        self.root.configure(bg=BG)

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Agregar imagen...", command=self.add_image)
        file_menu.add_command(label="Agregar referencia al gesto...", command=self.add_reference)
        file_menu.add_command(label="Reemplazar imagen seleccionada...", command=self.replace_image)
        file_menu.add_command(label="Iniciar captura guiada...", command=self.toggle_guided_capture)
        file_menu.add_command(label="Administrar muestras...", command=self.open_sample_manager)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.close)
        menu.add_cascade(label="Archivo", menu=file_menu)

        camera_menu = tk.Menu(menu, tearoff=False)
        camera_menu.add_command(label="Reiniciar camara", command=self._restart_camera)
        menu.add_cascade(label="Camara", menu=camera_menu)
        self.root.configure(menu=menu)

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg=PANEL, height=62, highlightthickness=1, highlightbackground=BORDER)
        header.pack(fill="x")
        header.pack_propagate(False)
        header.grid_columnconfigure(2, weight=1)
        tk.Label(
            header,
            text="GESTURE STUDIO",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 17),
        ).grid(row=0, column=0, padx=(22, 16), pady=12, sticky="w")
        tk.Label(
            header,
            textvariable=self.model_var,
            bg=PANEL_ALT,
            fg=CYAN,
            font=("Segoe UI", 10),
            padx=12,
            pady=6,
        ).grid(row=0, column=1, sticky="w")
        self._button(
            header,
            "Probar reconocimiento",
            self.launch_recognition,
            bg=CYAN,
            fg="#071014",
        ).grid(row=0, column=3, padx=18, pady=12, sticky="e")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, minsize=290)
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, minsize=290)
        body.grid_rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_camera_panel(body)
        self._build_action_panel(body)

    def _build_left_panel(self, parent: tk.Widget) -> None:
        panel = tk.Frame(parent, bg=PANEL, width=290, highlightthickness=1, highlightbackground=BORDER)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.grid_propagate(False)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(2, weight=1)

        title_row = tk.Frame(panel, bg=PANEL)
        title_row.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        tk.Label(
            title_row,
            text="Gestos e imagenes",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 13),
        ).pack(side="left")
        self._button(title_row, "+ Agregar", self.add_image, compact=True).pack(side="right")

        tk.Label(
            panel,
            text="Selecciona la imagen que recibira las muestras de tu gesto.",
            bg=PANEL,
            fg=MUTED,
            justify="left",
            wraplength=270,
            font=("Segoe UI", 9),
        ).grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))

        list_host = tk.Frame(panel, bg=PANEL)
        list_host.grid(row=2, column=0, sticky="nsew", padx=(12, 4))
        self.gesture_canvas = tk.Canvas(list_host, bg=PANEL, highlightthickness=0, width=276)
        scrollbar = tk.Scrollbar(list_host, orient="vertical", command=self.gesture_canvas.yview)
        self.gesture_list = tk.Frame(self.gesture_canvas, bg=PANEL)
        self.gesture_window = self.gesture_canvas.create_window((0, 0), window=self.gesture_list, anchor="nw")
        self.gesture_canvas.configure(yscrollcommand=scrollbar.set)
        self.gesture_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.gesture_list.bind(
            "<Configure>",
            lambda _event: self.gesture_canvas.configure(scrollregion=self.gesture_canvas.bbox("all")),
        )
        self.gesture_canvas.bind(
            "<Configure>",
            lambda event: self.gesture_canvas.itemconfigure(self.gesture_window, width=event.width),
        )

        footer = tk.Frame(panel, bg=PANEL)
        footer.grid(row=3, column=0, sticky="ew", padx=16, pady=14)
        self._button(footer, "Reemplazar imagen", self.replace_image, compact=True).pack(fill="x")
        self._button(footer, "Agregar referencia", self.add_reference, compact=True, bg=CYAN, fg="#071014").pack(
            fill="x", pady=(6, 0)
        )
        self._button(footer, "Ver referencias", self.open_reference_gallery, compact=True).pack(
            fill="x", pady=(6, 0)
        )

    def _build_camera_panel(self, parent: tk.Widget) -> None:
        panel = tk.Frame(parent, bg=BG)
        panel.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        status = tk.Frame(panel, bg=PANEL_ALT, height=44, highlightthickness=1, highlightbackground=BORDER)
        status.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        status.grid_propagate(False)
        self.status_dot = tk.Canvas(status, width=18, height=18, bg=PANEL_ALT, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(14, 8))
        self._set_status_color(AMBER)
        tk.Label(
            status,
            textvariable=self.status_var,
            bg=PANEL_ALT,
            fg=TEXT,
            font=("Segoe UI Semibold", 10),
        ).pack(side="left")
        tk.Label(
            status,
            text="Cuadro: posicion | A: giro | T: inclinacion",
            bg=PANEL_ALT,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(side="right", padx=14)

        self.camera_canvas = tk.Canvas(
            panel,
            bg="#090c10",
            width=480,
            height=270,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.camera_canvas.grid(row=1, column=0, sticky="nsew")
        self.camera_canvas.bind(
            "<Configure>",
            lambda event: self.camera_canvas.coords("camera_message", event.width // 2, event.height // 2),
        )
        self._show_camera_message("Preparando video...")

        telemetry = tk.Frame(panel, bg=BG, height=88)
        telemetry.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        telemetry.grid_propagate(False)
        telemetry.grid_columnconfigure(0, weight=1, uniform="hands")
        telemetry.grid_columnconfigure(1, weight=1, uniform="hands")
        for index in range(2):
            canvas = tk.Canvas(
                telemetry,
                height=84,
                bg=PANEL_ALT,
                highlightthickness=1,
                highlightbackground=BORDER,
            )
            canvas.grid(row=0, column=index, sticky="nsew", padx=(0, 4) if index == 0 else (4, 0))
            self.hand_telemetry_canvases.append(canvas)
        self._draw_hand_telemetry(None)

        notice = tk.Label(
            panel,
            textvariable=self.notice_var,
            bg=BG,
            fg=MUTED,
            anchor="w",
            justify="left",
            wraplength=700,
            font=("Segoe UI", 9),
        )
        notice.grid(row=3, column=0, sticky="ew", pady=(8, 0))

    def _build_action_panel(self, parent: tk.Widget) -> None:
        outer = tk.Frame(parent, bg=PANEL, width=290, highlightthickness=1, highlightbackground=BORDER)
        outer.grid(row=0, column=2, sticky="nsew")
        outer.grid_propagate(False)

        action_canvas = tk.Canvas(outer, bg=PANEL, width=276, highlightthickness=0)
        scrollbar = tk.Scrollbar(outer, orient="vertical", command=action_canvas.yview)
        panel = tk.Frame(action_canvas, bg=PANEL)
        action_window = action_canvas.create_window((0, 0), window=panel, anchor="nw")
        action_canvas.configure(yscrollcommand=scrollbar.set)
        action_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        panel.bind(
            "<Configure>",
            lambda _event: action_canvas.configure(scrollregion=action_canvas.bbox("all")),
        )
        action_canvas.bind(
            "<Configure>",
            lambda event: action_canvas.itemconfigure(action_window, width=event.width),
        )

        tk.Label(
            panel,
            text="GESTO ACTIVO",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", padx=18, pady=(18, 8))
        self.selected_image_label = tk.Label(
            panel,
            text="Agrega o selecciona una imagen",
            image=self.selected_placeholder,
            compound="center",
            bg="#0c1014",
            fg=MUTED,
            font=("Segoe UI", 9),
        )
        self.selected_image_label.pack(padx=18)
        tk.Label(
            panel,
            textvariable=self.selected_var,
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 14),
            wraplength=245,
        ).pack(padx=18, pady=(12, 2))
        tk.Label(
            panel,
            textvariable=self.count_var,
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10),
            justify="center",
            wraplength=260,
        ).pack(padx=18)

        tk.Label(
            panel,
            text="ESTABILIDAD DE CAPTURA",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 8),
        ).pack(anchor="w", padx=18, pady=(12, 0))
        self.progress_canvas = tk.Canvas(panel, width=246, height=12, bg=PANEL, highlightthickness=0)
        self.progress_canvas.pack(padx=18, pady=(5, 2))
        tk.Label(
            panel,
            textvariable=self.guided_var,
            bg=PANEL,
            fg=CYAN,
            font=("Segoe UI Semibold", 9),
            wraplength=246,
            justify="center",
        ).pack(fill="x", padx=18, pady=(5, 0))

        self.capture_button = self._button(
            panel,
            "Capturar gesto",
            self.capture_sample,
            bg=GREEN,
            fg="#07120b",
            large=True,
        )
        self.capture_button.pack(fill="x", padx=18, pady=(14, 8))
        self._set_capture_ready(False)
        self.guided_button = self._button(
            panel,
            "Iniciar captura guiada",
            self.toggle_guided_capture,
            bg=CYAN,
            fg="#071014",
        )
        self.guided_button.pack(fill="x", padx=18, pady=4)
        self.undo_button = self._button(panel, "Deshacer ultima", self.undo_sample)
        self.undo_button.pack(fill="x", padx=18, pady=4)
        self.gallery_button = self._button(panel, "Galeria de fotos", self.open_capture_gallery)
        self.gallery_button.pack(fill="x", padx=18, pady=4)
        self.samples_button = self._button(panel, "Administrar muestras", self.open_sample_manager)
        self.samples_button.pack(fill="x", padx=18, pady=4)
        self.train_button = self._button(panel, "Entrenar modelo", self.train_current_model, bg=AMBER, fg="#171006")
        self.train_button.pack(fill="x", padx=18, pady=4)

        tk.Frame(panel, bg=BORDER, height=1).pack(fill="x", padx=18, pady=16)
        tk.Label(
            panel,
            text="ULTIMA CAPTURA",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", padx=18, pady=(0, 7))
        self.last_capture_label = tk.Label(
            panel,
            text="Todavia no hay captura nueva",
            image=self.last_placeholder,
            compound="center",
            bg="#0c1014",
            fg=MUTED,
            font=("Segoe UI", 9),
        )
        self.last_capture_label.pack(padx=18)

    def _button(
        self,
        parent: tk.Widget,
        text: str,
        command,
        bg: str = PANEL_ALT,
        fg: str = TEXT,
        compact: bool = False,
        large: bool = False,
    ) -> tk.Button:
        pady = 10 if large else (4 if compact else 7)
        font_size = 11 if large else 9
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=CYAN,
            activeforeground="#071014",
            disabledforeground="#66717d",
            relief="flat",
            bd=0,
            padx=12,
            pady=pady,
            cursor="hand2",
            font=("Segoe UI Semibold", font_size),
        )

    def _set_status_color(self, color: str) -> None:
        self.status_dot.delete("all")
        self.status_dot.create_oval(3, 3, 15, 15, fill=color, outline="")

    def _set_capture_ready(self, ready: bool) -> None:
        if ready:
            self.capture_button.configure(
                state="normal",
                bg=GREEN,
                fg="#07120b",
                cursor="hand2",
            )
        else:
            self.capture_button.configure(
                state="disabled",
                bg="#273039",
                disabledforeground="#77838f",
                cursor="arrow",
            )

    def _draw_hand_telemetry(self, result: FeatureResult | None) -> None:
        poses = result.hand_poses if result is not None else []
        for index, canvas in enumerate(self.hand_telemetry_canvases):
            canvas.delete("all")
            width = max(canvas.winfo_width(), 230)
            height = 84
            color = GREEN if index == 0 else "#50b9ff"
            canvas.create_text(
                12,
                10,
                text=f"MANO {index + 1}",
                fill=color,
                anchor="nw",
                font=("Segoe UI Semibold", 9),
            )
            if index >= len(poses):
                canvas.create_text(
                    12,
                    42,
                    text="No detectada",
                    fill=MUTED,
                    anchor="w",
                    font=("Segoe UI", 9),
                )
                continue

            pose = poses[index]
            map_x1, map_y1, map_x2, map_y2 = 12, 30, 78, 75
            canvas.create_rectangle(map_x1, map_y1, map_x2, map_y2, outline=BORDER)
            canvas.create_line((map_x1 + map_x2) / 2, map_y1, (map_x1 + map_x2) / 2, map_y2, fill="#39434d")
            canvas.create_line(map_x1, (map_y1 + map_y2) / 2, map_x2, (map_y1 + map_y2) / 2, fill="#39434d")
            position_x = map_x1 + pose.center_x * (map_x2 - map_x1)
            position_y = map_y1 + pose.center_y * (map_y2 - map_y1)
            canvas.create_oval(
                position_x - 4,
                position_y - 4,
                position_x + 4,
                position_y + 4,
                fill=color,
                outline="",
            )

            canvas.create_text(
                90,
                30,
                text=f"X {pose.center_x:.0%}   Y {pose.center_y:.0%}",
                fill=TEXT,
                anchor="nw",
                font=("Segoe UI Semibold", 9),
            )
            canvas.create_text(
                90,
                49,
                text=f"Giro {pose.angle_deg:+.0f}deg   Tilt {pose.tilt_deg:+.0f}deg",
                fill=TEXT,
                anchor="nw",
                font=("Segoe UI", 9),
            )
            canvas.create_text(
                90,
                68,
                text=pose.zone,
                fill=MUTED,
                anchor="nw",
                font=("Segoe UI", 8),
            )

            gauge_x = width - 32
            gauge_y = 45
            radius = 22
            canvas.create_oval(
                gauge_x - radius,
                gauge_y - radius,
                gauge_x + radius,
                gauge_y + radius,
                outline="#46515d",
            )
            canvas.create_line(gauge_x, gauge_y - radius + 4, gauge_x, gauge_y + radius - 4, fill="#39434d")
            radians = np.radians(pose.angle_deg)
            end_x = gauge_x + np.sin(radians) * (radius - 5)
            end_y = gauge_y - np.cos(radians) * (radius - 5)
            canvas.create_line(gauge_x, gauge_y, end_x, end_y, fill=color, width=3, arrow="last")

    def _open_camera(self) -> None:
        self.cap = cv2.VideoCapture(self.config.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            self.status_var.set(f"No pude abrir la camara {self.config.camera_index}")
            self._set_status_color(RED)
            self._show_camera_message("Camara no disponible")
            self.notice_var.set("Cierra otras apps que usen la camara o cambia camera_index en app_config.json.")

    def _restart_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.stability.reset()
        self._show_camera_message("Reiniciando camara...")
        self._open_camera()
        self.notice_var.set("Camara reiniciada.")

    def _show_camera_message(self, message: str) -> None:
        width = max(self.camera_canvas.winfo_width(), 320)
        height = max(self.camera_canvas.winfo_height(), 240)
        self.camera_canvas.delete("video")
        self.camera_canvas.delete("camera_message")
        self.camera_canvas.create_text(
            width // 2,
            height // 2,
            text=message,
            fill=MUTED,
            font=("Segoe UI", 12),
            tags="camera_message",
        )

    def _update_camera(self) -> None:
        if self.closing:
            return
        if self.cap is None or not self.cap.isOpened():
            self.root.after(500, self._update_camera)
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status_var.set("No llega video de la camara")
            self._set_status_color(RED)
            self._show_camera_message("No llega video de la camara")
            self.root.after(250, self._update_camera)
            return

        frame = cv2.flip(frame, 1)
        result = self.extractor.extract(frame)
        hand_ready = result is not None and bool(result.hands)
        stable, movement = self.stability.update(result.vector if hand_ready and result else None)
        guided_match = self._guided_target_matches(result)
        guided_waiting = self.guided_active and time.monotonic() < self.guided_next_capture
        self.ready = bool(
            self.selected_label
            and hand_ready
            and stable
            and (not self.guided_active or guided_match)
            and not guided_waiting
        )
        self.latest_result = result

        display = frame.copy()
        draw_landmarks(display, result)
        self.latest_capture_frame = display.copy()
        self._draw_guided_overlay(display, guided_match)
        canvas_width = max(self.camera_canvas.winfo_width(), 320)
        canvas_height = max(self.camera_canvas.winfo_height(), 240)
        self.camera_photo = bgr_to_photo(
            display,
            canvas_width - 4,
            canvas_height - 4,
        )
        self.camera_canvas.delete("camera_message")
        self.camera_canvas.delete("video")
        self.camera_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.camera_photo,
            anchor="center",
            tags="video",
        )
        self._update_detection_status(result, stable, movement, guided_match)
        self._draw_hand_telemetry(result)
        self._update_guided_capture(stable, guided_match)
        self.root.after(20, self._update_camera)

    def _update_detection_status(
        self,
        result: FeatureResult | None,
        stable: bool,
        movement: float,
        guided_match: bool,
    ) -> None:
        if self.selected_label is None:
            self.status_var.set("Selecciona o agrega una imagen")
            self._set_status_color(AMBER)
            self._set_capture_ready(False)
        elif result is None or not result.hands:
            self.status_var.set("Coloca al menos una mano en cuadro")
            self._set_status_color(AMBER)
            self._set_capture_ready(False)
        elif self.guided_active and time.monotonic() < self.guided_next_capture:
            self.status_var.set("Muestra guardada - prepara la siguiente variacion")
            self._set_status_color(CYAN)
            self._set_capture_ready(False)
        elif self.guided_active and not guided_match:
            target = self._current_guided_target()
            self.status_var.set(target.instruction if target else "Prepara el gesto")
            self._set_status_color(AMBER)
            self._set_capture_ready(False)
        elif not stable:
            detail = "calculando" if not np.isfinite(movement) else f"movimiento {movement:.3f}"
            self.status_var.set(f"Manten el gesto quieto - {detail}")
            self._set_status_color(AMBER)
            self._set_capture_ready(False)
        else:
            face = "mano + cara" if result.faces else "mano"
            self.status_var.set(f"Listo para capturar ({face})")
            self._set_status_color(GREEN)
            self._set_capture_ready(True)

        target = self.config.target_samples_per_gesture
        progress = min(self.stability.sample_count / self.config.capture_stability_frames, 1.0)
        self._draw_progress(progress, GREEN if stable else AMBER)
        if self.selected_label:
            count = self.sample_counts.get(self.selected_label, 0)
            photo_count = self.photo_counts.get(self.selected_label, 0)
            reference_count = self.reference_counts.get(self.selected_label, 0)
            trained_references = self.reference_training_counts.get(self.selected_label, 0)
            vector_only = max(count - photo_count - trained_references, 0)
            self.count_var.set(
                f"{count} de {target} muestras para entrenar\n"
                f"{photo_count} con foto | {vector_only} solo vector | {reference_count} referencias"
            )

    def _current_guided_target(self) -> CaptureTarget | None:
        if not self.guided_active or self.guided_index >= len(self.guided_targets):
            return None
        return self.guided_targets[self.guided_index]

    def _guided_target_matches(self, result: FeatureResult | None) -> bool:
        target = self._current_guided_target()
        pose = result.hand_poses[0] if result and result.hand_poses else None
        return bool(target and target.matches(pose))

    def _draw_guided_overlay(self, frame: np.ndarray, matches: bool) -> None:
        target = self._current_guided_target()
        if target is None:
            return
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = target.bounds
        start = (int(x1 * width), int(y1 * height))
        end = (int(x2 * width), int(y2 * height))
        color = (116, 199, 72) if matches else (75, 184, 242)
        cv2.rectangle(frame, start, end, color, 3)
        label = f"GUIA {self.guided_index + 1}/{len(self.guided_targets)}: {target.title}"
        cv2.putText(
            frame,
            label,
            (start[0], max(28, start[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    def toggle_guided_capture(self) -> None:
        if self.guided_active:
            self._stop_guided_capture("Captura guiada detenida.")
            return
        if self.selected_label is None:
            messagebox.showinfo("Selecciona un gesto", "Selecciona primero el gesto que quieres capturar.", parent=self.root)
            return
        current = self.sample_counts.get(self.selected_label, 0)
        remaining = max(self.config.target_samples_per_gesture - current, 1)
        initial = min(12, max(3, remaining))
        total = simpledialog.askinteger(
            "Captura guiada",
            "Cuantas muestras nuevas quieres capturar?",
            initialvalue=initial,
            minvalue=3,
            maxvalue=30,
            parent=self.root,
        )
        if total is None:
            return
        self.guided_active = True
        self.guided_label = self.selected_label
        self.guided_targets = build_capture_targets(total)
        self.guided_index = 0
        self.guided_next_capture = time.monotonic() + 0.6
        self.stability.reset()
        self.guided_button.configure(text="Detener captura guiada", bg=RED, fg=TEXT)
        self.guided_var.set(f"Guiada 1/{total}: {self.guided_targets[0].title}")
        self.notice_var.set(
            "Mantén el mismo gesto y mueve la mano al cuadro indicado. La foto se tomara automaticamente."
        )

    def _update_guided_capture(self, stable: bool, matches: bool) -> None:
        if not self.guided_active or not stable or not matches or not self.ready:
            return
        if self.selected_label != self.guided_label:
            self._stop_guided_capture("La sesion se detuvo porque cambio el gesto activo.")
            return
        if not self.capture_sample():
            return
        completed = self.guided_index + 1
        total = len(self.guided_targets)
        self.guided_index = completed
        self.stability.reset()
        self.ready = False
        if completed >= total:
            label = self.guided_label
            self._stop_guided_capture(f"Sesion guiada terminada: {total} muestras nuevas para '{label}'.")
            return
        self.guided_next_capture = time.monotonic() + max(0.9, self.config.capture_min_interval_seconds)
        target = self.guided_targets[self.guided_index]
        self.guided_var.set(f"Guiada {self.guided_index + 1}/{total}: {target.title}")
        self.notice_var.set(f"Muestra {completed}/{total} guardada. {target.instruction}.")

    def _stop_guided_capture(self, message: str) -> None:
        self.guided_active = False
        self.guided_label = None
        self.guided_targets = []
        self.guided_index = 0
        self.guided_next_capture = float("inf")
        self.stability.reset()
        self.guided_button.configure(text="Iniciar captura guiada", bg=CYAN, fg="#071014")
        self.guided_var.set("Captura manual")
        self.notice_var.set(message)

    def _draw_progress(self, value: float, color: str) -> None:
        self.progress_canvas.delete("all")
        width = 246
        self.progress_canvas.create_rectangle(0, 1, width, 11, fill="#303842", outline="")
        self.progress_canvas.create_rectangle(0, 1, int(width * value), 11, fill=color, outline="")

    def _rebuild_gesture_list(self) -> None:
        for child in self.gesture_list.winfo_children():
            child.destroy()
        self.gesture_photos.clear()
        self.gesture_buttons.clear()

        if not self.gesture_map:
            tk.Label(
                self.gesture_list,
                text="No hay imagenes.\nPulsa Agregar para crear el primer gesto.",
                bg=PANEL,
                fg=MUTED,
                justify="left",
                font=("Segoe UI", 10),
            ).pack(fill="x", padx=8, pady=18)
            return

        for index, (label, path) in enumerate(self.gesture_map.items(), start=1):
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            photo = bgr_to_photo(image, 72, 54) if image is not None else None
            if photo is not None:
                self.gesture_photos[label] = photo
            count = self.sample_counts.get(label, 0)
            photo_count = self.photo_counts.get(label, 0)
            reference_count = self.reference_counts.get(label, 0)
            trained_references = self.reference_training_counts.get(label, 0)
            vector_only = max(count - photo_count - trained_references, 0)
            selected = label == self.selected_label
            button = tk.Button(
                self.gesture_list,
                text=(
                    f"{index}. {label}\n"
                    f"{count} muestras | {photo_count} con foto\n"
                    f"{vector_only} solo vector | {reference_count} ref"
                ),
                image=photo,
                compound="left",
                command=lambda item=label: self.select_gesture(item),
                bg="#29323b" if selected else PANEL_ALT,
                fg=TEXT,
                activebackground="#35414c",
                activeforeground=TEXT,
                relief="flat",
                bd=0,
                anchor="w",
                justify="left",
                padx=8,
                pady=8,
                cursor="hand2",
                font=("Segoe UI Semibold", 9),
            )
            button.pack(fill="x", padx=(0, 8), pady=3)
            self.gesture_buttons[label] = button

    def select_gesture(self, label: str) -> None:
        if label not in self.gesture_map:
            return
        if self.guided_active and label != self.guided_label:
            self._stop_guided_capture("Captura guiada detenida al cambiar de gesto.")
        self.selected_label = label
        self.stability.reset()
        self._rebuild_gesture_list()
        self._refresh_selected_panel()
        vectors = self.sample_counts.get(label, 0)
        photos = self.photo_counts.get(label, 0)
        trained_references = self.reference_training_counts.get(label, 0)
        legacy = max(vectors - photos - trained_references, 0)
        if legacy:
            self.notice_var.set(
                f"'{label}' tiene {vectors} vectores: {photos} con foto y {legacy} antiguos sin foto. "
                "Todos siguen sirviendo para entrenar."
            )
        else:
            self.notice_var.set(f"Ahora tus capturas se guardaran para '{label}'.")

    def _refresh_selected_panel(self) -> None:
        if self.selected_label is None or self.selected_label not in self.gesture_map:
            self.selected_var.set("Sin gesto seleccionado")
            self.count_var.set("0 muestras")
            self.selected_image_label.configure(
                image=self.selected_placeholder,
                text="Agrega una imagen",
                fg=MUTED,
            )
            self.selected_photo = None
            return

        label = self.selected_label
        self.selected_var.set(label)
        count = self.sample_counts.get(label, 0)
        photo_count = self.photo_counts.get(label, 0)
        reference_count = self.reference_counts.get(label, 0)
        trained_references = self.reference_training_counts.get(label, 0)
        vector_only = max(count - photo_count - trained_references, 0)
        self.count_var.set(
            f"{count} de {self.config.target_samples_per_gesture} muestras para entrenar\n"
            f"{photo_count} con foto | {vector_only} solo vector | {reference_count} referencias"
        )
        image = cv2.imread(str(self.gesture_map[label]), cv2.IMREAD_UNCHANGED)
        if image is not None:
            self.selected_photo = bgr_to_photo(image, 240, 150)
            self.selected_image_label.configure(image=self.selected_photo, text="")
        self._load_latest_capture_preview(label)

    def _load_latest_capture_preview(self, label: str) -> None:
        records = capture_records(label)
        if not records:
            self.last_photo = None
            self.last_capture_label.configure(
                image=self.last_placeholder,
                text="No hay fotos revisables\n(los vectores antiguos siguen guardados)",
            )
            return
        path, _captured_at = records[-1]
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return
        self.last_photo = bgr_to_photo(image, 240, 108)
        self.last_capture_label.configure(image=self.last_photo, text="")

    def open_capture_gallery(self) -> None:
        if self.selected_label is None:
            return
        label = self.selected_label
        records = capture_records(label)
        vector_count = self.sample_counts.get(label, 0)
        trained_references = self.reference_training_counts.get(label, 0)
        missing_photos = max(vector_count - len(records) - trained_references, 0)

        window = tk.Toplevel(self.root)
        window.title(f"Capturas - {label}")
        window.geometry("900x620")
        window.minsize(620, 420)
        window.configure(bg=BG)

        header = tk.Frame(window, bg=PANEL, height=70, highlightthickness=1, highlightbackground=BORDER)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text=label,
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 16),
        ).pack(anchor="w", padx=18, pady=(10, 0))
        summary = f"{vector_count} muestras | {len(records)} fotos de camara"
        if missing_photos:
            summary += f" | {missing_photos} vectores antiguos sin foto"
        tk.Label(
            header,
            text=summary,
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=18)

        host = tk.Frame(window, bg=BG)
        host.pack(fill="both", expand=True, padx=12, pady=12)
        canvas = tk.Canvas(host, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(host, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=BG)
        canvas_window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(canvas_window, width=event.width))

        window.gallery_photos = []
        if not records:
            tk.Label(
                content,
                text="Todavia no hay fotos para mostrar.\nLas muestras antiguas solo guardaron el vector numerico.",
                bg=BG,
                fg=MUTED,
                font=("Segoe UI", 11),
                justify="center",
            ).pack(pady=80)
            return

        for column in range(3):
            content.grid_columnconfigure(column, weight=1, uniform="gallery")
        for index, (path, captured_at) in enumerate(reversed(records)):
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            photo = bgr_to_photo(image, 250, 150)
            window.gallery_photos.append(photo)
            tile = tk.Frame(content, bg=PANEL_ALT, highlightthickness=1, highlightbackground=BORDER)
            tile.grid(row=index // 3, column=index % 3, sticky="nsew", padx=6, pady=6)
            tk.Label(tile, image=photo, bg="#0c1014").pack(fill="x", padx=7, pady=(7, 4))
            tk.Label(
                tile,
                text=captured_at or path.stem,
                bg=PANEL_ALT,
                fg=MUTED,
                font=("Segoe UI", 8),
            ).pack(anchor="w", padx=8, pady=(0, 7))

    def open_sample_manager(self) -> None:
        if self.selected_label is None:
            messagebox.showinfo("Selecciona un gesto", "Selecciona primero el gesto que quieres revisar.", parent=self.root)
            return
        label = self.selected_label
        window = tk.Toplevel(self.root)
        window.title(f"Administrar muestras - {label}")
        window.geometry("1040x680")
        window.minsize(820, 560)
        window.configure(bg=BG)

        header = tk.Frame(window, bg=PANEL, height=72, highlightthickness=1, highlightbackground=BORDER)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text=f"MUESTRAS: {label}",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 15),
        ).pack(anchor="w", padx=18, pady=(10, 0))
        summary_var = tk.StringVar()
        tk.Label(header, textvariable=summary_var, bg=PANEL, fg=MUTED, font=("Segoe UI", 9)).pack(
            anchor="w", padx=18
        )

        body = tk.Frame(window, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=12)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(1, weight=1)

        toolbar = tk.Frame(body, bg=BG)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(toolbar, text="Mostrar", bg=BG, fg=MUTED, font=("Segoe UI Semibold", 9)).pack(side="left")
        filter_var = tk.StringVar(value="Todas")
        filter_box = ttk.Combobox(
            toolbar,
            textvariable=filter_var,
            values=("Todas", "Con foto", "Solo vector", "Referencias"),
            state="readonly",
            width=18,
        )
        filter_box.pack(side="left", padx=8)

        table_host = tk.Frame(body, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        table_host.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        table_host.grid_rowconfigure(0, weight=1)
        table_host.grid_columnconfigure(0, weight=1)
        style = ttk.Style(window)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(
            "Samples.Treeview",
            background=PANEL_ALT,
            fieldbackground=PANEL_ALT,
            foreground=TEXT,
            rowheight=32,
            borderwidth=0,
            font=("Segoe UI", 9),
        )
        style.map(
            "Samples.Treeview",
            background=[("selected", "#087fca")],
            foreground=[("selected", TEXT)],
        )
        style.configure(
            "Samples.Treeview.Heading",
            background=PANEL,
            foreground=TEXT,
            relief="flat",
            font=("Segoe UI Semibold", 9),
        )
        tree = ttk.Treeview(
            table_host,
            columns=("origin", "date"),
            show="tree headings",
            style="Samples.Treeview",
            selectmode="browse",
        )
        tree.heading("#0", text="Muestra")
        tree.heading("origin", text="Origen")
        tree.heading("date", text="Fecha")
        tree.column("#0", width=130, minwidth=100)
        tree.column("origin", width=150, minwidth=110)
        tree.column("date", width=170, minwidth=120)
        table_scroll = tk.Scrollbar(table_host, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=table_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        table_scroll.grid(row=0, column=1, sticky="ns")

        preview = tk.Frame(body, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        preview.grid(row=0, column=1, rowspan=2, sticky="nsew")
        tk.Label(
            preview,
            text="MUESTRA SELECCIONADA",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", padx=14, pady=(14, 8))
        placeholder = solid_photo(340, 230, "#0c1014")
        preview_label = tk.Label(
            preview,
            image=placeholder,
            text="Selecciona una muestra",
            compound="center",
            bg="#0c1014",
            fg=MUTED,
            font=("Segoe UI", 10),
        )
        preview_label.pack(fill="x", padx=14)
        details_var = tk.StringVar(value="")
        tk.Label(
            preview,
            textvariable=details_var,
            bg=PANEL,
            fg=TEXT,
            justify="left",
            anchor="nw",
            wraplength=330,
            font=("Segoe UI", 9),
        ).pack(fill="x", padx=16, pady=14)
        delete_button = self._button(preview, "Eliminar muestra", lambda: None, bg=RED, fg=TEXT)
        delete_button.pack(side="bottom", fill="x", padx=14, pady=14)
        delete_button.configure(state="disabled")

        records: list[TrainingSampleRecord] = []
        record_by_item: dict[str, TrainingSampleRecord] = {}
        reference_by_id = {}
        window.preview_photo = placeholder

        def source_for(record: TrainingSampleRecord) -> str:
            if record.sample_id and record.sample_id in reference_by_id:
                return "reference"
            return record.source

        def source_text(source: str) -> str:
            return {
                "camera": "Camara",
                "reference": "Referencia",
                "vector_only": "Solo vector",
                "manifest_only": "Sin foto",
            }.get(source, source)

        def filtered(record: TrainingSampleRecord) -> bool:
            source = source_for(record)
            selected_filter = filter_var.get()
            if selected_filter == "Con foto":
                return source == "camera"
            if selected_filter == "Solo vector":
                return source in {"vector_only", "manifest_only"}
            if selected_filter == "Referencias":
                return source == "reference"
            return True

        def selected_record() -> TrainingSampleRecord | None:
            selected = tree.selection()
            return record_by_item.get(selected[0]) if selected else None

        def show_record(_event=None) -> None:
            record = selected_record()
            if record is None:
                window.preview_photo = placeholder
                preview_label.configure(image=placeholder, text="Selecciona una muestra")
                details_var.set("")
                delete_button.configure(state="disabled")
                return
            source = source_for(record)
            image_path = record.frame_path
            if source == "reference" and record.sample_id:
                image_path = reference_by_id[record.sample_id].annotated_path
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED) if image_path else None
            if image is not None:
                window.preview_photo = bgr_to_photo(image, 340, 230)
                preview_label.configure(image=window.preview_photo, text="")
            else:
                window.preview_photo = placeholder
                preview_label.configure(image=placeholder, text="Vector numerico sin foto")
            sample_id = record.sample_id or "registro antiguo"
            captured_at = record.captured_at or "sin fecha guardada"
            details_var.set(
                f"Muestra {record.ordinal}\n"
                f"Origen: {source_text(source)}\n"
                f"Fecha: {captured_at}\n"
                f"ID: {sample_id}"
            )
            delete_button.configure(state="normal")

        def refresh_manager(*_args) -> None:
            nonlocal records, reference_by_id
            records = load_sample_records(label)
            reference_by_id = {record.reference_id: record for record in load_reference_records(label)}
            for item in tree.get_children():
                tree.delete(item)
            record_by_item.clear()
            photo_count = sum(source_for(record) == "camera" for record in records)
            reference_count = sum(source_for(record) == "reference" for record in records)
            vector_only = len(records) - photo_count - reference_count
            summary_var.set(
                f"{len(records)} muestras | {photo_count} con foto | "
                f"{vector_only} solo vector | {reference_count} referencias de entrenamiento"
            )
            for record in records:
                if not filtered(record):
                    continue
                item = f"sample-{record.csv_row_index}"
                source = source_for(record)
                tree.insert(
                    "",
                    "end",
                    iid=item,
                    text=f"Muestra {record.ordinal}",
                    values=(source_text(source), record.captured_at or "-"),
                )
                record_by_item[item] = record
            children = tree.get_children()
            if children:
                tree.selection_set(children[-1])
                tree.focus(children[-1])
                tree.see(children[-1])
            show_record()

        def delete_selected() -> None:
            record = selected_record()
            if record is None:
                return
            source = source_for(record)
            confirmed = messagebox.askyesno(
                "Eliminar muestra",
                f"Eliminar la muestra {record.ordinal} ({source_text(source)}) de '{label}'?",
                parent=window,
            )
            if not confirmed:
                return
            removed, _removed_path, sample_id = remove_sample_record(record)
            if not removed:
                messagebox.showerror("No pude eliminar", "La lista cambio; actualiza e intenta de nuevo.", parent=window)
                refresh_manager()
                return
            if source == "reference" and sample_id:
                mark_reference_not_training(sample_id)
            self.sample_counts = load_sample_counts(list(self.gesture_map))
            self.photo_counts[label] = len(capture_records(label))
            references = load_reference_records(label)
            self.reference_counts[label] = len(references)
            self.reference_training_counts[label] = sum(record.used_for_training for record in references)
            self._rebuild_gesture_list()
            self._refresh_selected_panel()
            self.model_var.set(self._model_status())
            self.notice_var.set(f"Muestra {record.ordinal} eliminada de '{label}'.")
            refresh_manager()

        delete_button.configure(command=delete_selected)
        tree.bind("<<TreeviewSelect>>", show_record)
        filter_box.bind("<<ComboboxSelected>>", refresh_manager)
        refresh_manager()

    def add_reference(self) -> None:
        if self.selected_label is None:
            messagebox.showinfo(
                "Selecciona un gesto",
                "Selecciona primero la imagen destino del gesto que quieres entrenar.",
                parent=self.root,
            )
            return
        source = self._choose_image()
        if source is None:
            return
        self.notice_var.set("Analizando referencia con MediaPipe...")
        self.root.update_idletasks()
        try:
            analysis = analyze_reference(source, self.extractor)
        except (OSError, ValueError) as exc:
            messagebox.showerror("No pude analizar la referencia", str(exc), parent=self.root)
            return
        self._show_reference_inspector(analysis)

    def _show_reference_inspector(self, analysis: ReferenceAnalysis) -> None:
        label = self.selected_label
        if label is None:
            return

        window = tk.Toplevel(self.root)
        window.title(f"Revisar referencia - {label}")
        window.geometry("1280x720")
        window.minsize(980, 620)
        window.configure(bg=BG)
        window.transient(self.root)

        header = tk.Frame(window, bg=PANEL, height=68, highlightthickness=1, highlightbackground=BORDER)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text=f"REFERENCIA PARA: {label}",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 15),
        ).pack(anchor="w", padx=18, pady=(10, 0))
        tk.Label(
            header,
            text="Confirma que la caja, los puntos y los angulos describen el gesto correcto.",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=18)

        body = tk.Frame(window, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=12)
        body.grid_columnconfigure(0, weight=1, uniform="images")
        body.grid_columnconfigure(1, weight=1, uniform="images")
        body.grid_columnconfigure(2, minsize=320)
        body.grid_rowconfigure(0, weight=1)

        original_panel = tk.Frame(body, bg=PANEL_ALT, highlightthickness=1, highlightbackground=BORDER)
        original_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        detected_panel = tk.Frame(body, bg=PANEL_ALT, highlightthickness=1, highlightbackground=BORDER)
        detected_panel.grid(row=0, column=1, sticky="nsew", padx=6)
        details = tk.Frame(body, bg=PANEL, width=320, highlightthickness=1, highlightbackground=BORDER)
        details.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        details.grid_propagate(False)

        tk.Label(
            original_panel,
            text="ORIGINAL",
            bg=PANEL_ALT,
            fg=MUTED,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", padx=12, pady=(10, 6))
        tk.Label(
            detected_panel,
            text="DETECCION DE VECTORES",
            bg=PANEL_ALT,
            fg=GREEN if analysis.quality.can_accept else RED,
            font=("Segoe UI Semibold", 9),
        ).pack(anchor="w", padx=12, pady=(10, 6))

        original_photo = bgr_to_photo(analysis.original, 430, 560)
        detected_photo = bgr_to_photo(analysis.annotated, 430, 560)
        original_label = tk.Label(original_panel, image=original_photo, bg="#090c10")
        original_label.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        detected_label = tk.Label(detected_panel, image=detected_photo, bg="#090c10")
        detected_label.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        window.reference_photos = [original_photo, detected_photo]

        quality_color = GREEN if analysis.quality.score >= 80 else AMBER if analysis.quality.can_accept else RED
        tk.Label(
            details,
            text=f"CALIDAD {analysis.quality.score}/100",
            bg=quality_color,
            fg="#07120b" if analysis.quality.can_accept else TEXT,
            font=("Segoe UI Semibold", 11),
            padx=12,
            pady=8,
        ).pack(fill="x", padx=14, pady=(14, 10))

        result = analysis.result
        hand_count = len(result.hands) if result else 0
        face_detected = bool(result and result.faces)
        tk.Label(
            details,
            text=f"Manos detectadas: {hand_count}\nCara detectada: {'si' if face_detected else 'no'}",
            bg=PANEL,
            fg=TEXT,
            justify="left",
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w", padx=16, pady=(0, 8))

        if result:
            for pose in result.hand_poses:
                pose_panel = tk.Frame(details, bg=PANEL_ALT, highlightthickness=1, highlightbackground=BORDER)
                pose_panel.pack(fill="x", padx=14, pady=4)
                tk.Label(
                    pose_panel,
                    text=f"MANO {pose.index}  |  {pose.zone}",
                    bg=PANEL_ALT,
                    fg=GREEN if pose.index == 1 else "#50b9ff",
                    font=("Segoe UI Semibold", 9),
                ).pack(anchor="w", padx=9, pady=(7, 2))
                tk.Label(
                    pose_panel,
                    text=(
                        f"X {pose.center_x:.0%}   Y {pose.center_y:.0%}\n"
                        f"Giro {pose.angle_deg:+.1f}deg   Tilt {pose.tilt_deg:+.1f}deg"
                    ),
                    bg=PANEL_ALT,
                    fg=TEXT,
                    justify="left",
                    font=("Segoe UI", 9),
                ).pack(anchor="w", padx=9, pady=(0, 7))

        messages = "\n".join(f"- {message}" for message in analysis.quality.messages)
        tk.Label(
            details,
            text=messages,
            bg=PANEL,
            fg=MUTED,
            justify="left",
            anchor="nw",
            wraplength=280,
            font=("Segoe UI", 9),
        ).pack(fill="x", padx=16, pady=10)

        buttons = tk.Frame(details, bg=PANEL)
        buttons.pack(side="bottom", fill="x", padx=14, pady=14)
        guide_button = self._button(
            buttons,
            "Guardar solo como guia",
            lambda: self._accept_reference(window, analysis, used_for_training=False),
        )
        guide_button.pack(fill="x", pady=3)
        training_button = self._button(
            buttons,
            "Aceptar y agregar al entrenamiento",
            lambda: self._accept_reference(window, analysis, used_for_training=True),
            bg=GREEN,
            fg="#07120b",
        )
        training_button.pack(fill="x", pady=3)
        self._button(buttons, "Cancelar", window.destroy, compact=True).pack(fill="x", pady=(8, 0))
        if not analysis.quality.can_accept:
            guide_button.configure(state="disabled", bg="#273039")
            training_button.configure(state="disabled", bg="#273039")

    def _accept_reference(
        self,
        window: tk.Toplevel,
        analysis: ReferenceAnalysis,
        used_for_training: bool,
    ) -> None:
        label = self.selected_label
        if label is None or analysis.result is None:
            return
        try:
            record = store_reference(label, analysis, used_for_training)
            if used_for_training:
                append_sample(label, analysis.result.vector)
                append_manifest(record.reference_id, label, None)
        except (OSError, ValueError) as exc:
            messagebox.showerror("No pude guardar la referencia", str(exc), parent=window)
            return

        self.reference_counts[label] = self.reference_counts.get(label, 0) + 1
        if used_for_training:
            self.reference_training_counts[label] = self.reference_training_counts.get(label, 0) + 1
            self.sample_counts[label] = self.sample_counts.get(label, 0) + 1
        self._rebuild_gesture_list()
        self._refresh_selected_panel()
        self.model_var.set(self._model_status())
        mode = "guia y muestra de entrenamiento" if used_for_training else "guia visual"
        self.notice_var.set(f"Referencia guardada para '{label}' como {mode}.")
        window.destroy()

    def open_reference_gallery(self) -> None:
        if self.selected_label is None:
            return
        label = self.selected_label
        records = load_reference_records(label)
        window = tk.Toplevel(self.root)
        window.title(f"Referencias - {label}")
        window.geometry("940x640")
        window.minsize(640, 440)
        window.configure(bg=BG)

        header = tk.Frame(window, bg=PANEL, height=68, highlightthickness=1, highlightbackground=BORDER)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text=f"REFERENCIAS: {label}",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 15),
        ).pack(anchor="w", padx=18, pady=(10, 0))
        training_count = sum(record.used_for_training for record in records)
        tk.Label(
            header,
            text=f"{len(records)} referencias | {training_count} agregadas al entrenamiento",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=18)

        host = tk.Frame(window, bg=BG)
        host.pack(fill="both", expand=True, padx=12, pady=12)
        canvas = tk.Canvas(host, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(host, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=BG)
        canvas_window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(canvas_window, width=event.width))
        window.reference_photos = []

        if not records:
            tk.Label(
                content,
                text="Todavia no hay referencias para este gesto.",
                bg=BG,
                fg=MUTED,
                font=("Segoe UI", 11),
            ).pack(pady=80)
            return

        for column in range(3):
            content.grid_columnconfigure(column, weight=1, uniform="references")
        for index, record in enumerate(reversed(records)):
            image = cv2.imread(str(record.annotated_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            photo = bgr_to_photo(image, 270, 170)
            window.reference_photos.append(photo)
            tile = tk.Frame(content, bg=PANEL_ALT, highlightthickness=1, highlightbackground=BORDER)
            tile.grid(row=index // 3, column=index % 3, sticky="nsew", padx=6, pady=6)
            tk.Label(tile, image=photo, bg="#0c1014").pack(fill="x", padx=7, pady=(7, 4))
            mode = "ENTRENAMIENTO" if record.used_for_training else "GUIA"
            color = GREEN if record.used_for_training else CYAN
            tk.Label(
                tile,
                text=f"{mode} | calidad {record.quality_score}/100",
                bg=PANEL_ALT,
                fg=color,
                font=("Segoe UI Semibold", 8),
            ).pack(anchor="w", padx=8)
            tk.Label(
                tile,
                text=record.created_at,
                bg=PANEL_ALT,
                fg=MUTED,
                font=("Segoe UI", 8),
            ).pack(anchor="w", padx=8, pady=(2, 7))

    def add_image(self) -> None:
        source = self._choose_image()
        if source is None:
            return
        label = simpledialog.askstring(
            "Nombre del gesto",
            "Escribe el nombre que tendra este gesto:",
            initialvalue=source.stem,
            parent=self.root,
        )
        if label is None:
            return
        label = label.strip()
        if not valid_label(label):
            messagebox.showerror(
                "Nombre invalido",
                "Usa un nombre corto sin /, \\, : ni caracteres de ruta.",
                parent=self.root,
            )
            return

        if label in self.gesture_map:
            replace = messagebox.askyesno(
                "El gesto ya existe",
                f"'{label}' ya tiene una imagen. Quieres reemplazarla?",
                parent=self.root,
            )
            if not replace:
                return
            self._replace_label_image(label, source)
            return

        destination = unique_image_path(label, source.suffix.lower())
        shutil.copy2(source, destination)
        self.gesture_map[label] = destination
        save_gesture_map(self.gesture_map)
        self.sample_counts = load_sample_counts(list(self.gesture_map))
        self.photo_counts[label] = 0
        self.reference_counts[label] = 0
        self.reference_training_counts[label] = 0
        self.selected_label = label
        self._rebuild_gesture_list()
        self._refresh_selected_panel()
        self.model_var.set(self._model_status())
        self.notice_var.set(f"Imagen agregada. Haz el gesto de '{label}' frente a la camara.")

    def replace_image(self) -> None:
        if self.selected_label is None:
            messagebox.showinfo("Selecciona un gesto", "Selecciona primero la imagen que quieres reemplazar.")
            return
        source = self._choose_image()
        if source is not None:
            self._replace_label_image(self.selected_label, source)

    def _replace_label_image(self, label: str, source: Path) -> None:
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        destination = self.gesture_map[label]
        if not cv2.imwrite(str(destination), image):
            messagebox.showerror("No se pudo guardar", f"No pude reemplazar {destination.name}.")
            return
        save_gesture_map(self.gesture_map)
        self._rebuild_gesture_list()
        self._refresh_selected_panel()
        self.notice_var.set(f"Imagen de '{label}' reemplazada. No necesitas reentrenar.")

    def _choose_image(self) -> Path | None:
        patterns = " ".join(f"*{suffix}" for suffix in sorted(IMAGE_SUFFIXES))
        filename = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imagenes compatibles", patterns), ("Todos los archivos", "*.*")],
            parent=self.root,
        )
        if not filename:
            return None
        path = Path(filename)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            messagebox.showerror(
                "Formato no compatible",
                "Usa PNG, JPG, JPEG, WEBP, BMP, TIF o TIFF.",
                parent=self.root,
            )
            return None
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            messagebox.showerror("Archivo invalido", "El archivo no contiene una imagen que OpenCV pueda leer.")
            return None
        return path

    def capture_sample(self) -> bool:
        if not self.ready or self.latest_result is None or self.latest_capture_frame is None:
            self.notice_var.set("Espera a que la mano este detectada y el indicador se ponga verde.")
            return False
        if self.selected_label is None:
            return False
        now = time.monotonic()
        if now - self.last_capture_time < self.config.capture_min_interval_seconds:
            self.notice_var.set("Espera un instante antes de capturar otra muestra.")
            return False

        label = self.selected_label
        sample_id = create_sample_id()
        append_sample(label, self.latest_result.vector)
        saved_path = save_capture_frame(sample_id, label, self.latest_capture_frame, self.config)
        append_manifest(sample_id, label, saved_path)
        self.sample_counts[label] = self.sample_counts.get(label, 0) + 1
        if saved_path is not None:
            self.photo_counts[label] = self.photo_counts.get(label, 0) + 1
        self.last_capture_time = now
        self.last_photo = bgr_to_photo(self.latest_capture_frame, 240, 108)
        self.last_capture_label.configure(image=self.last_photo, text="")
        self._rebuild_gesture_list()
        self._refresh_selected_panel()
        self.model_var.set(self._model_status())
        self.notice_var.set(
            f"Muestra {self.sample_counts[label]} guardada para '{label}'. Cambia un poco el angulo para la siguiente."
        )
        return True

    def undo_sample(self) -> None:
        if self.selected_label is None:
            return
        label = self.selected_label
        removed, removed_path, sample_id = remove_last_sample_with_id(label)
        if not removed:
            self.notice_var.set(f"'{label}' todavia no tiene muestras para deshacer.")
            return
        self.sample_counts[label] = max(0, self.sample_counts.get(label, 0) - 1)
        if removed_path is not None:
            self.photo_counts[label] = max(0, self.photo_counts.get(label, 0) - 1)
        elif sample_id and mark_reference_not_training(sample_id):
            self.reference_training_counts[label] = max(
                0,
                self.reference_training_counts.get(label, 0) - 1,
            )
        self.last_photo = None
        self.last_capture_label.configure(image=self.last_placeholder, text="Ultima muestra eliminada")
        self._rebuild_gesture_list()
        self._refresh_selected_panel()
        self.model_var.set(self._model_status())
        detail = f" y {removed_path.name}" if removed_path else ""
        self.notice_var.set(f"Elimine el ultimo vector de '{label}'{detail}.")

    def train_current_model(self) -> None:
        labels = list(self.gesture_map)
        if not labels:
            messagebox.showinfo("Sin gestos", "Agrega al menos una imagen antes de entrenar.")
            return
        self.train_button.configure(state="disabled", text="Entrenando...")
        self.root.update_idletasks()
        try:
            message = train_model(labels, self.config)
            self.notice_var.set(message)
            self.model_var.set(self._model_status())
            if message.startswith("Modelo guardado."):
                messagebox.showinfo("Entrenamiento terminado", message, parent=self.root)
            else:
                messagebox.showwarning("Faltan muestras", message, parent=self.root)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Error al entrenar", str(exc), parent=self.root)
        finally:
            self.train_button.configure(state="normal", text="Entrenar modelo")

    def launch_recognition(self) -> None:
        if not MODEL_FILE.exists():
            messagebox.showinfo(
                "Modelo pendiente",
                "Captura muestras de todos los gestos y pulsa Entrenar modelo primero.",
                parent=self.root,
            )
            return
        if not self._model_is_current():
            messagebox.showinfo(
                "Modelo desactualizado",
                "Las etiquetas o muestras cambiaron desde el ultimo entrenamiento. Pulsa Entrenar modelo antes de probar.",
                parent=self.root,
            )
            return
        self._shutdown_resources()
        subprocess.Popen([sys.executable, str(ROOT / "gesture_launcher.py")], cwd=str(ROOT))
        self.root.destroy()

    def _model_status(self) -> str:
        if MODEL_FILE.exists():
            return "Modelo listo" if self._model_is_current() else "Modelo desactualizado"
        total = len(self.gesture_map)
        ready = sum(self.sample_counts.get(label, 0) >= 3 for label in self.gesture_map)
        return f"Modelo pendiente | {ready}/{total} gestos listos"

    def _model_is_current(self) -> bool:
        if not MODEL_FILE.exists():
            return False
        try:
            payload = joblib.load(MODEL_FILE)
        except (OSError, ValueError, KeyError, EOFError):
            return False
        model_labels = list(payload.get("labels", []))
        model_counts = payload.get("sample_counts")
        current_counts = {label: self.sample_counts.get(label, 0) for label in self.gesture_map}
        return model_labels == list(self.gesture_map) and model_counts == current_counts

    def _shutdown_resources(self) -> None:
        if self.closing:
            return
        self.closing = True
        if self.cap is not None:
            self.cap.release()
        self.extractor.close()

    def close(self) -> None:
        self._shutdown_resources()
        self.root.destroy()


def valid_label(label: str) -> bool:
    return bool(label) and not any(character in label for character in '<>:"/\\|?*') and label not in {".", ".."}


def unique_image_path(label: str, suffix: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "gesto"
    candidate = IMAGE_DIR / f"{safe_name}{suffix}"
    index = 2
    while candidate.exists():
        candidate = IMAGE_DIR / f"{safe_name}_{index}{suffix}"
        index += 1
    return candidate


def capture_records(label: str) -> list[tuple[Path, str]]:
    if not MANIFEST_FILE.exists():
        return []

    records: list[tuple[Path, str]] = []
    with MANIFEST_FILE.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("label") != label or not row.get("frame_path"):
                continue
            path = (ROOT / row["frame_path"]).resolve()
            if path.is_relative_to(CAPTURE_DIR.resolve()) and path.is_file():
                records.append((path, row.get("captured_at", "")))
    records.sort(key=lambda item: (item[1], item[0].name))
    return records


def bgr_to_photo(image: np.ndarray, max_width: int, max_height: int) -> tk.PhotoImage:
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = alpha * image[:, :, :3] + (1.0 - alpha) * np.array([18, 18, 18], dtype=np.float32)
        rgb = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)

    height, width = rgb.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    target_width = max(1, int(width * scale))
    target_height = max(1, int(height * scale))
    if (target_width, target_height) != (width, height):
        rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    ppm = f"P6\n{target_width} {target_height}\n255\n".encode("ascii") + rgb.tobytes()
    return tk.PhotoImage(data=ppm, format="PPM")


def solid_photo(width: int, height: int, color: str) -> tk.PhotoImage:
    photo = tk.PhotoImage(width=width, height=height)
    photo.put(color, to=(0, 0, width, height))
    return photo


def smoke_test() -> None:
    config = load_config()
    gesture_map = load_gesture_map()
    invalid = [path for path in gesture_map.values() if cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is None]
    if invalid:
        raise SystemExit(f"Imagenes invalidas: {invalid}")
    extractor = LandmarkFeatureExtractor()
    result = extractor.extract(np.zeros((480, 640, 3), dtype=np.uint8))
    extractor.close()
    print(f"studio=ok camera_index={config.camera_index} gestures={list(gesture_map)} blank_result={result}")


def open_reference_smoke_test(studio: GestureStudio) -> None:
    for label in studio.gesture_map:
        records = capture_records(label)
        if not records:
            continue
        studio.select_gesture(label)
        analysis = analyze_reference(records[-1][0], studio.extractor)
        studio._show_reference_inspector(analysis)
        return
    studio.open_reference_gallery()


def open_manager_smoke_test(studio: GestureStudio) -> None:
    if studio.sample_counts:
        label = max(studio.sample_counts, key=studio.sample_counts.get)
        studio.select_gesture(label)
    studio.open_sample_manager()


def main() -> None:
    if "--smoke-test" in sys.argv:
        smoke_test()
        return

    ui_smoke_test = "--ui-smoke-test" in sys.argv
    manager_smoke_test = "--manager-smoke-test" in sys.argv
    root = tk.Tk()
    try:
        config = load_config()
        gesture_map = load_gesture_map()
        extractor = LandmarkFeatureExtractor()
        studio = GestureStudio(root, config, gesture_map, extractor)
    except (RuntimeError, OSError) as exc:
        messagebox.showerror("No pude iniciar Gesture Studio", str(exc), parent=root)
        root.destroy()
        return
    if ui_smoke_test:
        root.after(900, lambda: open_reference_smoke_test(studio))
        root.after(7000, studio.close)
    elif manager_smoke_test:
        root.after(900, lambda: open_manager_smoke_test(studio))
        root.after(7000, studio.close)
    root.mainloop()
    if ui_smoke_test:
        print("studio_ui=ok")
    elif manager_smoke_test:
        print("studio_manager_ui=ok")


if __name__ == "__main__":
    main()
