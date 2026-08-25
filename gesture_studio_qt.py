from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal
from PySide6.QtGui import QAction, QCloseEvent, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app_config import IMAGE_DIR, IMAGE_SUFFIXES, load_config, load_gesture_map, save_gesture_map
from gesture_features import FeatureResult, LandmarkFeatureExtractor, draw_landmarks, summarize_vector
from gesture_launcher import MODEL_FILE
from gesture_runtime import FeatureStabilityTracker
from guided_capture import CaptureTarget, build_capture_targets
from qt_components import FadeController, FadingStackedWidget, Toast, app_icon, apply_button_icon
from qt_theme import COLORS, application_stylesheet, configure_application_font
from reference_images import analyze_reference, load_reference_records, store_reference
from train_gestures import (
    CAPTURE_DIR,
    append_manifest,
    append_sample,
    create_sample_id,
    load_sample_counts,
    load_sample_records,
    remove_last_sample_with_id,
    remove_sample_record,
    save_capture_frame,
    train_model,
)


ROOT = Path(__file__).parent


class CameraWorker(QObject):
    frame_ready = Signal(object, object, bool, float)
    status_changed = Signal(str, str)
    finished = Signal()

    def __init__(self, camera_index: int, stability_frames: int, stability_threshold: float) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.stability_frames = stability_frames
        self.stability_threshold = stability_threshold
        self.running = True

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        extractor = None
        capture = None
        try:
            self.status_changed.emit("Preparando MediaPipe...", "busy")
            extractor = LandmarkFeatureExtractor()
            capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not capture.isOpened():
                self.status_changed.emit(
                    "No pude abrir la camara. Cierra Teams u otra aplicacion que la este usando.",
                    "error",
                )
                return

            tracker = FeatureStabilityTracker(self.stability_frames, self.stability_threshold)
            self.status_changed.emit("Camara activa", "ok")
            while self.running:
                ok, frame = capture.read()
                if not ok:
                    self.status_changed.emit("La camara dejo de entregar video.", "error")
                    break
                frame = cv2.flip(frame, 1)
                result = extractor.extract(frame)
                vector = result.vector if result is not None and result.hands else None
                stable, movement = tracker.update(vector)
                annotated = frame.copy()
                draw_landmarks(annotated, result)
                self.frame_ready.emit(annotated, result, stable, movement)
                time.sleep(0.005)
        except Exception as exc:
            self.status_changed.emit(f"Error de camara: {exc}", "error")
        finally:
            if capture is not None:
                capture.release()
            if extractor is not None:
                extractor.close()
            self.finished.emit()


class ReferenceDialog(QDialog):
    def __init__(self, analysis, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.analysis = analysis
        self.decision: str | None = None
        self.setWindowTitle(f"Revisar referencia - {label}")
        self.resize(1040, 700)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)
        title = QLabel("Comprueba los vectores antes de aceptar")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        previews = QHBoxLayout()
        previews.setSpacing(12)
        previews.addWidget(self._preview_panel("Imagen original", analysis.original), 1)
        previews.addWidget(self._preview_panel("Deteccion MediaPipe", analysis.annotated), 1)
        root.addLayout(previews, 1)

        quality = QFrame()
        quality.setObjectName("panel")
        quality_layout = QVBoxLayout(quality)
        quality_title = QLabel(f"Calidad {analysis.quality.score}/100")
        quality_title.setObjectName("success" if analysis.quality.score >= 70 else "warning")
        quality_layout.addWidget(quality_title)
        for message in analysis.quality.messages:
            quality_layout.addWidget(QLabel(message))
        root.addWidget(quality)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancelar")
        guide = QPushButton("Guardar como guia")
        train = QPushButton("Aceptar y entrenar")
        train.setObjectName("primary")
        apply_button_icon(cancel, "fa6s.xmark")
        apply_button_icon(guide, "fa6s.bookmark")
        apply_button_icon(train, "fa6s.check", "#ffffff")
        train.setEnabled(analysis.quality.can_accept)
        cancel.clicked.connect(self.reject)
        guide.clicked.connect(lambda: self._finish("guide"))
        train.clicked.connect(lambda: self._finish("train"))
        buttons.addWidget(cancel)
        buttons.addWidget(guide)
        buttons.addWidget(train)
        root.addLayout(buttons)

    def _preview_panel(self, title: str, image: np.ndarray) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        preview = QLabel()
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setMinimumSize(420, 360)
        preview.setPixmap(bgr_to_pixmap(image).scaled(
            470, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        layout.addWidget(heading)
        layout.addWidget(preview, 1)
        return panel

    def _finish(self, decision: str) -> None:
        self.decision = decision
        self.accept()


class GestureStudioQt(QMainWindow):
    def __init__(self, start_camera: bool = True) -> None:
        super().__init__()
        self.config = load_config()
        self.gesture_map = load_gesture_map()
        self.labels = list(self.gesture_map)
        self.selected_label = self.labels[0] if self.labels else None
        self.sample_counts: dict[str, int] = {}
        self.current_frame: np.ndarray | None = None
        self.current_result: FeatureResult | None = None
        self.current_stable = False
        self.current_movement = float("inf")
        self.camera_thread: QThread | None = None
        self.camera_worker: CameraWorker | None = None
        self.guided_targets: list[CaptureTarget] = []
        self.guided_index = 0
        self.guided_last_capture = 0.0
        self.nav_buttons: list[QPushButton] = []
        self.fade_controller = FadeController(self)
        self.toast: Toast | None = None

        self.setWindowTitle("Gesture Pop Studio")
        self.setMinimumSize(1180, 760)
        self.resize(1460, 900)
        self.setStyleSheet(application_stylesheet())
        self._build_ui()
        self._build_menu()
        self._configure_interactions()
        self.refresh_all()
        if start_camera:
            self.start_camera()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_sidebar())

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 18, 24, 18)
        content_layout.setSpacing(14)
        content_layout.addLayout(self._build_topbar())
        self.pages = FadingStackedWidget()
        self.capture_page = self._build_capture_page()
        self.dataset_page = self._build_dataset_page()
        self.training_page = self._build_training_page()
        self.recognition_page = self._build_recognition_page()
        for page in (self.capture_page, self.dataset_page, self.training_page, self.recognition_page):
            self.pages.addWidget(page)
        content_layout.addWidget(self.pages, 1)
        layout.addWidget(content, 1)

        self.statusBar().showMessage("Listo")

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(244)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(8)

        brand_row = QHBoxLayout()
        mark = QLabel()
        mark.setObjectName("brandMark")
        mark.setPixmap(app_icon("fa6s.hand", COLORS["blue"]).pixmap(23, 23))
        brand = QLabel("Gesture Pop")
        brand.setObjectName("brand")
        brand_row.addWidget(mark)
        brand_row.addWidget(brand)
        brand_row.addStretch()
        layout.addLayout(brand_row)
        subtitle = QLabel("ESTUDIO LOCAL")
        subtitle.setObjectName("eyebrow")
        layout.addWidget(subtitle)
        layout.addSpacing(10)

        nav_group = QButtonGroup(self)
        nav_group.setExclusive(True)
        nav_specs = [
            ("Captura", "fa6s.camera"),
            ("Muestras", "fa6s.images"),
            ("Entrenamiento", "fa6s.chart-simple"),
            ("Reconocimiento", "fa6s.play"),
        ]
        for index, (text, icon_name) in enumerate(nav_specs):
            button = QPushButton(text)
            button.setObjectName("navButton")
            button.setCheckable(True)
            apply_button_icon(button, icon_name)
            button.setIconSize(QSize(18, 18))
            button.clicked.connect(lambda checked=False, page=index: self._switch_page(page))
            nav_group.addButton(button)
            self.nav_buttons.append(button)
            layout.addWidget(button)
        self.nav_buttons[0].setChecked(True)

        layout.addSpacing(14)
        label_row = QHBoxLayout()
        label_title = QLabel("GESTOS")
        label_title.setObjectName("eyebrow")
        self.gesture_total_label = QLabel("0")
        self.gesture_total_label.setObjectName("muted")
        label_row.addWidget(label_title)
        label_row.addStretch()
        label_row.addWidget(self.gesture_total_label)
        layout.addLayout(label_row)

        self.gesture_list = QListWidget()
        self.gesture_list.setIconSize(QSize(46, 46))
        self.gesture_list.currentItemChanged.connect(self._on_gesture_selected)
        layout.addWidget(self.gesture_list, 1)

        image_actions = QHBoxLayout()
        add = QPushButton("Agregar")
        apply_button_icon(add, "fa6s.plus")
        add.setToolTip("Agregar una nueva imagen y gesto")
        add.clicked.connect(self.add_gesture)
        replace = QPushButton()
        apply_button_icon(replace, "fa6s.rotate")
        replace.setToolTip("Reemplazar la imagen del gesto seleccionado")
        replace.setFixedWidth(42)
        replace.clicked.connect(self.replace_image)
        image_actions.addWidget(add, 1)
        image_actions.addWidget(replace)
        layout.addLayout(image_actions)
        return sidebar

    def _build_topbar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        title_box = QVBoxLayout()
        self.page_title = QLabel("Captura")
        self.page_title.setObjectName("pageTitle")
        self.page_subtitle = QLabel("Camara y vectores en vivo")
        self.page_subtitle.setObjectName("muted")
        title_box.addWidget(self.page_title)
        title_box.addWidget(self.page_subtitle)
        layout.addLayout(title_box)
        layout.addStretch()
        self.model_badge = QLabel()
        self.model_badge.setObjectName("warning")
        layout.addWidget(self.model_badge)
        camera_button = QPushButton()
        apply_button_icon(camera_button, "fa6s.rotate")
        camera_button.setToolTip("Reiniciar camara")
        camera_button.setFixedWidth(42)
        camera_button.clicked.connect(self.restart_camera)
        layout.addWidget(camera_button)
        return layout

    def _build_capture_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        camera_panel = QFrame()
        camera_panel.setObjectName("panel")
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setContentsMargins(12, 12, 12, 12)
        camera_header = QHBoxLayout()
        self.camera_dot = QLabel()
        self.camera_dot.setPixmap(app_icon("fa6s.circle", COLORS["amber"]).pixmap(9, 9))
        self.camera_dot.setFixedWidth(12)
        self.camera_status = QLabel("Camara detenida")
        self.camera_status.setObjectName("muted")
        camera_header.addWidget(self.camera_dot)
        camera_header.addWidget(self.camera_status)
        camera_header.addStretch()
        self.vector_status = QLabel("Sin vector")
        self.vector_status.setObjectName("muted")
        camera_header.addWidget(self.vector_status)
        camera_layout.addLayout(camera_header)

        self.video_label = QLabel("Iniciando camara...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 420)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("background: #080a0d; border-radius: 5px;")
        camera_layout.addWidget(self.video_label, 1)

        guide_row = QHBoxLayout()
        self.guidance_label = QLabel("Muestra una mano completa y manten el gesto quieto.")
        self.guidance_label.setObjectName("muted")
        self.guidance_label.setWordWrap(True)
        self.stability_progress = QProgressBar()
        self.stability_progress.setRange(0, self.config.capture_stability_frames)
        self.stability_progress.setFixedWidth(150)
        guide_row.addWidget(self.guidance_label, 1)
        guide_row.addWidget(self.stability_progress)
        camera_layout.addLayout(guide_row)

        controls = QHBoxLayout()
        self.capture_button = QPushButton("Capturar gesto")
        self.capture_button.setObjectName("primary")
        apply_button_icon(self.capture_button, "fa6s.camera", "#ffffff")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.capture_sample)
        self.guided_button = QPushButton("Captura guiada")
        apply_button_icon(self.guided_button, "fa6s.route")
        self.guided_button.clicked.connect(self.toggle_guided_capture)
        undo = QPushButton()
        apply_button_icon(undo, "fa6s.rotate-left")
        undo.setToolTip("Deshacer la ultima muestra")
        undo.setFixedWidth(42)
        undo.clicked.connect(self.undo_sample)
        controls.addWidget(self.capture_button)
        controls.addWidget(self.guided_button)
        controls.addWidget(undo)
        controls.addStretch()
        reference = QPushButton("Subir referencia")
        apply_button_icon(reference, "fa6s.image")
        reference.clicked.connect(self.add_reference)
        controls.addWidget(reference)
        camera_layout.addLayout(controls)
        layout.addWidget(camera_panel, 1)

        inspector = QWidget()
        inspector.setFixedWidth(310)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(12)

        target = QFrame()
        target.setObjectName("panel")
        target_layout = QVBoxLayout(target)
        eyebrow = QLabel("GESTO ACTIVO")
        eyebrow.setObjectName("eyebrow")
        self.target_preview = QLabel()
        self.target_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_preview.setMinimumHeight(165)
        self.target_preview.setStyleSheet("background: #0a0d10; border-radius: 4px;")
        self.selected_name = QLabel("Sin gesto")
        self.selected_name.setObjectName("sectionTitle")
        self.selected_count = QLabel("0 muestras")
        self.selected_count.setObjectName("muted")
        self.target_progress = QProgressBar()
        self.target_progress.setRange(0, self.config.target_samples_per_gesture)
        target_layout.addWidget(eyebrow)
        target_layout.addWidget(self.target_preview)
        target_layout.addWidget(self.selected_name)
        target_layout.addWidget(self.selected_count)
        target_layout.addWidget(self.target_progress)
        inspector_layout.addWidget(target)

        telemetry = QFrame()
        telemetry.setObjectName("panel")
        telemetry_layout = QVBoxLayout(telemetry)
        telemetry_title = QLabel("Lectura de vectores")
        telemetry_title.setObjectName("sectionTitle")
        self.hand_count_label = QLabel("Manos  0")
        self.face_label = QLabel("Cara  no detectada")
        self.pose_label = QLabel("Posicion  --\nAngulo  --\nInclinacion  --")
        self.pose_label.setWordWrap(True)
        self.vector_summary_label = QLabel("Esperando landmarks...")
        self.vector_summary_label.setObjectName("muted")
        self.vector_summary_label.setWordWrap(True)
        telemetry_layout.addWidget(telemetry_title)
        telemetry_layout.addWidget(self.hand_count_label)
        telemetry_layout.addWidget(self.face_label)
        telemetry_layout.addWidget(self.pose_label)
        telemetry_layout.addWidget(self.vector_summary_label)
        inspector_layout.addWidget(telemetry)
        inspector_layout.addStretch()
        layout.addWidget(inspector)
        return page

    def _build_dataset_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        self.dataset_heading = QLabel("Muestras")
        self.dataset_heading.setObjectName("sectionTitle")
        self.dataset_summary = QLabel()
        self.dataset_summary.setObjectName("muted")
        header.addWidget(self.dataset_heading)
        header.addWidget(self.dataset_summary)
        header.addStretch()
        refresh = QPushButton()
        apply_button_icon(refresh, "fa6s.rotate")
        refresh.setToolTip("Actualizar galeria")
        refresh.setFixedWidth(42)
        refresh.clicked.connect(self.refresh_dataset)
        header.addWidget(refresh)
        layout.addLayout(header)

        self.dataset_scroll = QScrollArea()
        self.dataset_scroll.setWidgetResizable(True)
        self.dataset_body = QWidget()
        self.dataset_grid = QGridLayout(self.dataset_body)
        self.dataset_grid.setContentsMargins(0, 8, 4, 8)
        self.dataset_grid.setSpacing(12)
        self.dataset_scroll.setWidget(self.dataset_body)
        layout.addWidget(self.dataset_scroll, 1)
        return page

    def _build_training_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        intro = QFrame()
        intro.setObjectName("panel")
        intro_layout = QHBoxLayout(intro)
        text = QVBoxLayout()
        title = QLabel("Preparar el modelo")
        title.setObjectName("pageTitle")
        copy = QLabel("Minimo 3 por gesto. Recomendado 20 con variaciones de luz, distancia y angulo.")
        copy.setObjectName("muted")
        copy.setWordWrap(True)
        text.addWidget(title)
        text.addWidget(copy)
        intro_layout.addLayout(text, 1)
        self.train_button = QPushButton("Entrenar modelo")
        self.train_button.setObjectName("warningButton")
        apply_button_icon(self.train_button, "fa6s.gears", "#181106")
        self.train_button.clicked.connect(self.train_current_model)
        intro_layout.addWidget(self.train_button)
        layout.addWidget(intro)

        self.training_scroll = QScrollArea()
        self.training_scroll.setWidgetResizable(True)
        self.training_body = QWidget()
        self.training_layout = QVBoxLayout(self.training_body)
        self.training_layout.setContentsMargins(0, 8, 4, 8)
        self.training_layout.setSpacing(10)
        self.training_scroll.setWidget(self.training_body)
        layout.addWidget(self.training_scroll, 1)
        return page

    def _build_recognition_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        panel = QFrame()
        panel.setObjectName("panel")
        panel_layout = QVBoxLayout(panel)
        eyebrow = QLabel("MODO EN VIVO")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Haz el gesto. Abre la imagen.")
        title.setObjectName("pageTitle")
        copy = QLabel("Manten el gesto estable para activar la imagen asignada.")
        copy.setObjectName("muted")
        copy.setWordWrap(True)
        self.recognition_model_status = QLabel()
        self.recognition_model_status.setWordWrap(True)
        self.launch_button = QPushButton("Iniciar reconocimiento")
        self.launch_button.setObjectName("primary")
        apply_button_icon(self.launch_button, "fa6s.play", "#ffffff")
        self.launch_button.clicked.connect(self.launch_recognition)
        panel_layout.addWidget(eyebrow)
        panel_layout.addWidget(title)
        panel_layout.addWidget(copy)
        panel_layout.addSpacing(18)
        panel_layout.addWidget(self.recognition_model_status)
        panel_layout.addStretch()
        panel_layout.addWidget(self.launch_button)
        layout.addWidget(panel, 2)

        action_panel = QFrame()
        action_panel.setObjectName("panel")
        action_layout = QVBoxLayout(action_panel)
        action_title = QLabel("Acciones configuradas")
        action_title.setObjectName("sectionTitle")
        action_layout.addWidget(action_title)
        self.action_list = QVBoxLayout()
        action_layout.addLayout(self.action_list)
        action_layout.addStretch()
        layout.addWidget(action_panel, 1)
        return page

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("Archivo")
        add_action = QAction("Agregar gesto", self)
        add_action.setIcon(app_icon("fa6s.plus"))
        add_action.triggered.connect(self.add_gesture)
        file_menu.addAction(add_action)
        reference_action = QAction("Subir referencia", self)
        reference_action.setIcon(app_icon("fa6s.image"))
        reference_action.triggered.connect(self.add_reference)
        file_menu.addAction(reference_action)
        file_menu.addSeparator()
        exit_action = QAction("Salir", self)
        exit_action.setIcon(app_icon("fa6s.xmark"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _switch_page(self, index: int) -> None:
        titles = [
            ("Captura", "Camara y vectores en vivo"),
            ("Muestras", "Fotos y vectores del dataset"),
            ("Entrenamiento", "Cobertura por clase y modelo"),
            ("Reconocimiento", "Gestos y acciones"),
        ]
        self.page_title.setText(titles[index][0])
        self.page_subtitle.setText(titles[index][1])
        if index == 1:
            self.refresh_dataset()
        elif index == 2:
            self.refresh_training()
        elif index == 3:
            self.refresh_recognition()
        self.pages.setCurrentIndex(index)

    def _configure_interactions(self) -> None:
        for button in self.findChildren(QPushButton):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.gesture_list.viewport().setCursor(Qt.CursorShape.PointingHandCursor)

        self.capture_shortcut = QShortcut(QKeySequence("Space"), self)
        self.capture_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.capture_shortcut.activated.connect(self._capture_from_shortcut)
        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.undo_shortcut.activated.connect(self.undo_sample)

    def _capture_from_shortcut(self) -> None:
        if self.pages.currentIndex() == 0 and self.capture_button.isEnabled():
            self.capture_button.click()

    def start_camera(self) -> None:
        if self.camera_thread is not None and self.camera_thread.isRunning():
            return
        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker(
            self.config.camera_index,
            self.config.capture_stability_frames,
            self.config.capture_stability_threshold,
        )
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.frame_ready.connect(self._on_frame)
        self.camera_worker.status_changed.connect(self._on_camera_status)
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_thread.finished.connect(self._camera_stopped)
        self.camera_thread.start()

    def stop_camera(self) -> None:
        if self.camera_worker is not None:
            self.camera_worker.stop()
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.quit()
            self.camera_thread.wait(3500)

    def restart_camera(self) -> None:
        self.stop_camera()
        self.video_label.setText("Reiniciando camara...")
        self.start_camera()

    def _camera_stopped(self) -> None:
        if self.camera_thread is not None:
            self.camera_thread.deleteLater()
        if self.camera_worker is not None:
            self.camera_worker.deleteLater()
        self.camera_thread = None
        self.camera_worker = None

    def _on_camera_status(self, message: str, level: str) -> None:
        colors = {"ok": COLORS["success"], "busy": COLORS["amber"], "error": COLORS["coral"]}
        self.camera_dot.setPixmap(app_icon("fa6s.circle", colors.get(level, COLORS["muted"])).pixmap(9, 9))
        self.camera_status.setText(message)
        self.statusBar().showMessage(message)
        if level == "error":
            self.video_label.setText(message)

    def _on_frame(self, frame: np.ndarray, result: FeatureResult | None, stable: bool, movement: float) -> None:
        self.current_frame = frame.copy()
        self.current_result = result
        self.current_stable = stable
        self.current_movement = movement
        shown = frame.copy()

        target = self._current_guided_target()
        target_matches = self._target_matches(target, result)
        if target is not None:
            self._draw_target(shown, target, target_matches)
            self._update_guided_capture(stable, target_matches)

        pixmap = bgr_to_pixmap(shown)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        self._update_telemetry(result, stable, movement)

    def _update_telemetry(self, result: FeatureResult | None, stable: bool, movement: float) -> None:
        hands = len(result.hands) if result else 0
        faces = len(result.faces) if result else 0
        self.hand_count_label.setText(f"Manos  {hands}")
        self.face_label.setText(f"Cara  {'detectada' if faces else 'no detectada'}")
        self.vector_status.setText("Vector estable" if stable else "Buscando estabilidad")
        stability_count = self.config.capture_stability_frames if stable else max(
            0, self.config.capture_stability_frames - 2
        ) if result and result.hands and np.isfinite(movement) else 0
        self.stability_progress.setValue(stability_count)
        self.capture_button.setEnabled(bool(self.selected_label and result and result.hands and stable))

        if result and result.hand_poses:
            pose = result.hand_poses[0]
            self.pose_label.setText(
                f"Posicion  X {pose.center_x:.0%}  Y {pose.center_y:.0%}\n"
                f"Zona  {pose.zone}\nAngulo  {pose.angle_deg:+.0f} deg\n"
                f"Inclinacion  {pose.tilt_deg:+.0f} deg"
            )
            self.guidance_label.setText(
                "Listo para capturar." if stable else "Manten el gesto quieto hasta completar la barra."
            )
        else:
            self.pose_label.setText("Posicion  --\nAngulo  --\nInclinacion  --")
            self.guidance_label.setText("Muestra una mano completa dentro del cuadro.")
        self.vector_summary_label.setText("\n".join(summarize_vector(result)))

    def _current_guided_target(self) -> CaptureTarget | None:
        if 0 <= self.guided_index < len(self.guided_targets):
            return self.guided_targets[self.guided_index]
        return None

    @staticmethod
    def _target_matches(target: CaptureTarget | None, result: FeatureResult | None) -> bool:
        pose = result.hand_poses[0] if result and result.hand_poses else None
        return bool(target and target.matches(pose))

    @staticmethod
    def _draw_target(frame: np.ndarray, target: CaptureTarget, matches: bool) -> None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = target.bounds
        color = (92, 220, 180) if matches else (85, 189, 246)
        cv2.rectangle(frame, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), color, 3)
        cv2.putText(
            frame,
            target.instruction,
            (int(x1 * width), max(30, int(y1 * height) - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    def toggle_guided_capture(self) -> None:
        if self.guided_targets:
            self._stop_guided_capture("Captura guiada cancelada")
            return
        if not self.selected_label:
            self._notify("Selecciona primero un gesto.", "warning")
            return
        remaining = max(self.config.target_samples_per_gesture - self.sample_counts.get(self.selected_label, 0), 1)
        count, ok = QInputDialog.getInt(self, "Captura guiada", "Cantidad de muestras:", min(6, remaining), 1, 30)
        if not ok:
            return
        self.guided_targets = build_capture_targets(count)
        self.guided_index = 0
        self.guided_last_capture = 0.0
        self.guided_button.setText("Cancelar guia")
        apply_button_icon(self.guided_button, "fa6s.xmark", COLORS["coral"])
        self.guided_button.setObjectName("danger")
        self.guided_button.style().unpolish(self.guided_button)
        self.guided_button.style().polish(self.guided_button)
        self._notify(f"Guia iniciada: {count} posiciones", "ok")

    def _update_guided_capture(self, stable: bool, matches: bool) -> None:
        if not self.guided_targets or not stable or not matches:
            return
        now = time.monotonic()
        if now - self.guided_last_capture < max(0.7, self.config.capture_min_interval_seconds):
            return
        if self.capture_sample(silent=True):
            self.guided_last_capture = now
            self.guided_index += 1
            if self.guided_index >= len(self.guided_targets):
                self._stop_guided_capture("Captura guiada completa")

    def _stop_guided_capture(self, message: str) -> None:
        self.guided_targets = []
        self.guided_index = 0
        self.guided_button.setText("Captura guiada")
        apply_button_icon(self.guided_button, "fa6s.route")
        self.guided_button.setObjectName("")
        self.guided_button.style().unpolish(self.guided_button)
        self.guided_button.style().polish(self.guided_button)
        self._notify(message, "ok")

    def capture_sample(self, checked: bool = False, silent: bool = False) -> bool:
        del checked
        if not self.selected_label or self.current_result is None or not self.current_result.hands:
            if not silent:
                self._notify("Necesito una mano detectada para capturar.", "warning")
            return False
        if not self.current_stable:
            if not silent:
                self._notify("Manten el gesto quieto hasta completar la barra.", "warning")
            return False
        sample_id = create_sample_id()
        append_sample(self.selected_label, self.current_result.vector)
        frame_path = save_capture_frame(sample_id, self.selected_label, self.current_frame, self.config)
        append_manifest(sample_id, self.selected_label, frame_path)
        self.refresh_all()
        if not silent:
            self._notify(f"Muestra guardada para {self.selected_label}", "ok")
        return True

    def undo_sample(self) -> None:
        if not self.selected_label:
            return
        removed, _path, _sample_id = remove_last_sample_with_id(self.selected_label)
        self.refresh_all()
        self._notify("Ultima muestra eliminada" if removed else "No hay muestras para deshacer", "ok" if removed else "warning")

    def add_reference(self) -> None:
        if not self.selected_label:
            self._notify("Selecciona el gesto que recibira la referencia.", "warning")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar referencia",
            str(ROOT),
            "Imagenes (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if not filename:
            return
        self._notify("Analizando referencia...", "busy")
        QApplication.processEvents()
        extractor = None
        try:
            extractor = LandmarkFeatureExtractor()
            analysis = analyze_reference(Path(filename), extractor)
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "Referencia no valida", str(exc))
            return
        finally:
            if extractor is not None:
                extractor.close()

        dialog = ReferenceDialog(analysis, self.selected_label, self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.decision is None:
            return
        use_for_training = dialog.decision == "train"
        store_reference(self.selected_label, analysis, use_for_training)
        if use_for_training and analysis.result is not None:
            append_sample(self.selected_label, analysis.result.vector)
        self.refresh_all()
        self._notify("Referencia agregada al entrenamiento" if use_for_training else "Referencia guardada como guia", "ok")

    def add_gesture(self) -> None:
        source = self._choose_image("Agregar imagen destino")
        if source is None:
            return
        suggested = re.sub(r"[^A-Za-z0-9_-]+", "_", source.stem).strip("_")
        label, ok = QInputDialog.getText(self, "Nuevo gesto", "Nombre del gesto:", text=suggested)
        label = label.strip()
        if not ok or not label:
            return
        if not valid_label(label):
            QMessageBox.warning(self, "Nombre no valido", "Usa letras, numeros, guion o guion bajo.")
            return
        if label in self.gesture_map:
            QMessageBox.warning(self, "Gesto existente", "Ese nombre ya esta configurado.")
            return
        destination = unique_image_path(label, source.suffix.lower())
        shutil.copy2(source, destination)
        self.gesture_map[label] = destination
        save_gesture_map(self.gesture_map)
        self.selected_label = label
        self.refresh_all()
        self._notify(f"Gesto {label} agregado", "ok")

    def replace_image(self) -> None:
        if not self.selected_label:
            return
        source = self._choose_image("Reemplazar imagen destino")
        if source is None:
            return
        destination = unique_image_path(self.selected_label, source.suffix.lower())
        shutil.copy2(source, destination)
        self.gesture_map[self.selected_label] = destination
        save_gesture_map(self.gesture_map)
        self.refresh_all()
        self._notify("Imagen destino actualizada", "ok")

    def _choose_image(self, title: str) -> Path | None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            title,
            str(ROOT),
            "Imagenes (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        return Path(filename) if filename else None

    def train_current_model(self) -> None:
        if not self.labels:
            return
        self.train_button.setEnabled(False)
        self.train_button.setText("Entrenando...")
        QApplication.processEvents()
        try:
            message = train_model(self.labels, self.config)
        except Exception as exc:
            message = f"No pude entrenar: {exc}"
        finally:
            self.train_button.setEnabled(True)
            self.train_button.setText("Entrenar modelo")
        self.refresh_all()
        QMessageBox.information(self, "Entrenamiento", message)

    def launch_recognition(self) -> None:
        if not MODEL_FILE.exists():
            QMessageBox.warning(self, "Falta el modelo", "Entrena el modelo antes de iniciar el reconocimiento.")
            return
        self._notify("Liberando la camara e iniciando reconocimiento...", "busy")
        self.stop_camera()
        try:
            subprocess.Popen([sys.executable, str(ROOT / "gesture_launcher.py")], cwd=str(ROOT))
        except OSError as exc:
            QMessageBox.critical(self, "No pude iniciar", str(exc))
            self.start_camera()
            return
        self._notify("Reconocimiento abierto en una ventana nueva", "ok")

    def refresh_all(self) -> None:
        old_label = self.selected_label
        self.gesture_map = load_gesture_map()
        self.labels = list(self.gesture_map)
        self.sample_counts = load_sample_counts(self.labels)
        if old_label in self.gesture_map:
            self.selected_label = old_label
        elif self.labels:
            self.selected_label = self.labels[0]
        else:
            self.selected_label = None
        self._refresh_gesture_list()
        self._refresh_selected_panel()
        self.refresh_dataset()
        self.refresh_training()
        self.refresh_recognition()
        self._refresh_model_badge()

    def _refresh_gesture_list(self) -> None:
        self.gesture_list.blockSignals(True)
        self.gesture_list.clear()
        selected_row = -1
        for index, label in enumerate(self.labels):
            count = self.sample_counts.get(label, 0)
            item = QListWidgetItem(QIcon(str(self.gesture_map[label])), f"{label}\n{count} muestras")
            item.setData(Qt.ItemDataRole.UserRole, label)
            item.setSizeHint(QSize(210, 62))
            self.gesture_list.addItem(item)
            if label == self.selected_label:
                selected_row = index
        self.gesture_total_label.setText(str(len(self.labels)))
        if selected_row >= 0:
            self.gesture_list.setCurrentRow(selected_row)
        self.gesture_list.blockSignals(False)

    def _on_gesture_selected(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        del previous
        if current is None:
            return
        self.selected_label = current.data(Qt.ItemDataRole.UserRole)
        self._stop_guided_capture("Gesto seleccionado") if self.guided_targets else None
        self._refresh_selected_panel()
        if self.pages.currentIndex() == 1:
            self.refresh_dataset()

    def _refresh_selected_panel(self) -> None:
        label = self.selected_label
        if not label:
            self.selected_name.setText("Sin gesto")
            self.target_preview.clear()
            return
        count = self.sample_counts.get(label, 0)
        self.selected_name.setText(label)
        self.selected_count.setText(f"{count} de {self.config.target_samples_per_gesture} muestras recomendadas")
        self.target_progress.setValue(min(count, self.config.target_samples_per_gesture))
        pixmap = QPixmap(str(self.gesture_map[label]))
        self.target_preview.setPixmap(pixmap.scaled(
            275, 165, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        self.fade_controller.pulse(self.target_preview)

    def refresh_dataset(self) -> None:
        clear_layout(self.dataset_grid)
        label = self.selected_label
        if not label:
            self.dataset_summary.setText("Selecciona un gesto")
            return
        records = list(reversed(load_sample_records(label)))
        references = list(reversed(load_reference_records(label)))
        self.dataset_heading.setText(f"Muestras de {label}")
        visible_photos = sum(record.frame_path is not None for record in records)
        self.dataset_summary.setText(
            f"{len(records)} vectores · {visible_photos} fotos de camara · {len(references)} referencias"
        )
        cards: list[QWidget] = []
        for record in records:
            title = f"Muestra {record.ordinal}"
            source = "Camara" if record.frame_path else "Vector sin foto"
            cards.append(self._sample_card(title, source, record.frame_path, lambda _=False, item=record: self._delete_sample(item)))
        for reference in references:
            source = f"Referencia · calidad {reference.quality_score}"
            if reference.used_for_training:
                source += " · entrenamiento"
            cards.append(self._sample_card("Referencia", source, reference.annotated_path, None))
        if not cards:
            empty = QLabel("Todavia no hay muestras para este gesto. Vuelve a Captura para crear la primera.")
            empty.setObjectName("muted")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.dataset_grid.addWidget(empty, 0, 0, 1, 3)
            return
        for index, card in enumerate(cards):
            self.dataset_grid.addWidget(card, index // 3, index % 3)
        self.dataset_grid.setRowStretch((len(cards) + 2) // 3, 1)

    def _sample_card(self, title: str, source: str, path: Path | None, delete_callback) -> QWidget:
        card = QFrame()
        card.setObjectName("sampleCard")
        card.setMinimumWidth(230)
        layout = QVBoxLayout(card)
        preview = QLabel()
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setFixedHeight(145)
        preview.setStyleSheet("background: #090c0f; border-radius: 4px;")
        if path is not None and path.is_file():
            pixmap = QPixmap(str(path))
            preview.setPixmap(pixmap.scaled(260, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            preview.setText("VECTOR\nSIN FOTO")
            preview.setStyleSheet(f"background: #121820; color: {COLORS['muted']}; border-radius: 4px;")
        row = QHBoxLayout()
        texts = QVBoxLayout()
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        meta = QLabel(source)
        meta.setObjectName("muted")
        meta.setWordWrap(True)
        texts.addWidget(heading)
        texts.addWidget(meta)
        row.addLayout(texts, 1)
        if delete_callback is not None:
            delete_button = QPushButton()
            apply_button_icon(delete_button, "fa6s.trash", COLORS["coral"])
            delete_button.setToolTip("Eliminar esta muestra")
            delete_button.setObjectName("danger")
            delete_button.setFixedSize(38, 36)
            delete_button.clicked.connect(delete_callback)
            row.addWidget(delete_button)
        layout.addWidget(preview)
        layout.addLayout(row)
        return card

    def _delete_sample(self, record) -> None:
        answer = QMessageBox.question(self, "Eliminar muestra", "Esta muestra dejara de participar en el entrenamiento. Continuar?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        removed, _path, _sample_id = remove_sample_record(record)
        self.refresh_all()
        self._notify("Muestra eliminada" if removed else "La muestra ya no existe", "ok" if removed else "warning")

    def refresh_training(self) -> None:
        clear_layout(self.training_layout)
        minimum_ready = True
        for label in self.labels:
            count = self.sample_counts.get(label, 0)
            minimum_ready = minimum_ready and count >= 3
            row = QFrame()
            row.setObjectName("metricPanel")
            layout = QHBoxLayout(row)
            icon = QLabel()
            pixmap = QPixmap(str(self.gesture_map[label]))
            icon.setPixmap(pixmap.scaled(58, 58, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            icon.setFixedSize(64, 64)
            texts = QVBoxLayout()
            title = QLabel(label)
            title.setObjectName("sectionTitle")
            status = QLabel("Lista para entrenar" if count >= 3 else f"Faltan {3 - count} para el minimo")
            status.setObjectName("success" if count >= 3 else "warning")
            bar = QProgressBar()
            bar.setRange(0, self.config.target_samples_per_gesture)
            bar.setValue(min(count, self.config.target_samples_per_gesture))
            texts.addWidget(title)
            texts.addWidget(status)
            texts.addWidget(bar)
            metric = QLabel(str(count))
            metric.setObjectName("metric")
            metric.setToolTip("Vectores disponibles")
            layout.addWidget(icon)
            layout.addLayout(texts, 1)
            layout.addWidget(metric)
            self.training_layout.addWidget(row)
        self.training_layout.addStretch()
        self.train_button.setEnabled(bool(self.labels and minimum_ready))
        self.train_button.setToolTip("" if minimum_ready else "Cada gesto necesita al menos 3 muestras")

    def refresh_recognition(self) -> None:
        clear_layout(self.action_list)
        for label, path in self.gesture_map.items():
            row = QHBoxLayout()
            icon = QLabel()
            icon.setPixmap(QPixmap(str(path)).scaled(44, 44, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            icon.setFixedSize(48, 48)
            text = QLabel(f"{label}\nAbre {path.name}")
            text.setWordWrap(True)
            row.addWidget(icon)
            row.addWidget(text, 1)
            self.action_list.addLayout(row)
        if MODEL_FILE.exists():
            self.recognition_model_status.setText("Modelo disponible. La camara del estudio se liberara al iniciar.")
            self.recognition_model_status.setObjectName("success")
            self.launch_button.setEnabled(True)
            self.launch_button.setToolTip("Abrir el reconocimiento en vivo")
        else:
            self.recognition_model_status.setText("Aun no existe un modelo entrenado.")
            self.recognition_model_status.setObjectName("warning")
            self.launch_button.setEnabled(False)
            self.launch_button.setToolTip("Entrena el modelo antes de iniciar")
        self.recognition_model_status.style().unpolish(self.recognition_model_status)
        self.recognition_model_status.style().polish(self.recognition_model_status)

    def _refresh_model_badge(self) -> None:
        if MODEL_FILE.exists():
            self.model_badge.setText("MODELO LISTO")
            self.model_badge.setObjectName("success")
        else:
            self.model_badge.setText("SIN ENTRENAR")
            self.model_badge.setObjectName("warning")
        self.model_badge.style().unpolish(self.model_badge)
        self.model_badge.style().polish(self.model_badge)

    def _notify(self, message: str, level: str = "ok") -> None:
        self.statusBar().showMessage(message, 5000)
        colors = {"ok": COLORS["success"], "warning": COLORS["amber"], "busy": COLORS["blue"]}
        self.statusBar().setStyleSheet(f"color: {colors.get(level, COLORS['muted'])};")
        if self.toast is not None:
            self.toast.hide()
            self.toast.deleteLater()
        toast = Toast(message, level, self.centralWidget())
        self.toast = toast
        toast.destroyed.connect(lambda _=None, target=toast: self._clear_toast(target))
        self._position_toast()
        toast.reveal()

    def _position_toast(self) -> None:
        if self.toast is None:
            return
        self.toast.adjustSize()
        parent = self.centralWidget()
        x = max(16, parent.width() - self.toast.width() - 24)
        y = max(16, parent.height() - self.toast.height() - 24)
        self.toast.move(x, y)

    def _clear_toast(self, toast: Toast) -> None:
        if self.toast is toast:
            self.toast = None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_toast()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_camera()
        event.accept()


def bgr_to_pixmap(image: np.ndarray) -> QPixmap:
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGBA8888).copy()
        return QPixmap.fromImage(qimage)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    qimage = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        elif child_layout is not None:
            clear_layout(child_layout)


def valid_label(label: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", label))


def unique_image_path(label: str, suffix: str) -> Path:
    IMAGE_DIR.mkdir(exist_ok=True)
    suffix = suffix if suffix in IMAGE_SUFFIXES else ".png"
    candidate = IMAGE_DIR / f"{label}{suffix}"
    index = 2
    while candidate.exists():
        candidate = IMAGE_DIR / f"{label}_{index}{suffix}"
        index += 1
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Gesture Pop Studio con PySide6")
    parser.add_argument("--smoke-test", action="store_true", help="Construye la interfaz sin abrir la camara")
    parser.add_argument("--screenshot", type=Path, help="Guarda una captura de la interfaz durante el smoke test")
    parser.add_argument("--page", type=int, choices=range(4), default=0, help="Pagina inicial para pruebas visuales")
    parser.add_argument("--label", help="Gesto seleccionado durante las pruebas visuales")
    args = parser.parse_args()
    if args.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Gesture Pop Studio")
    app.setStyle("Fusion")
    configure_application_font(app)
    window = GestureStudioQt(start_camera=not args.smoke_test)
    if args.label in window.gesture_map:
        window.selected_label = args.label
        window.refresh_all()
    window.nav_buttons[args.page].setChecked(True)
    window._switch_page(args.page)
    window.show()
    if args.smoke_test:
        from PySide6.QtTest import QTest

        QTest.qWait(240)
        if args.screenshot is not None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(args.screenshot)):
                raise RuntimeError(f"No pude guardar la captura en {args.screenshot}")
        print(f"QT_SMOKE_OK pages={window.pages.count()} gestures={len(window.labels)}")
        window.close()
        return
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
