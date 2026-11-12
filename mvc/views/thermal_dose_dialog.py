"""CEM43 display shown after a Cinema sequence finishes.

The 39°C, CEM43 = 2, and CEM43 = 9 guides are optional and unchecked until
the user turns them on.  Their caption is the van Rhoon et al., Eur Radiol
2013 MR radiofrequency safety note.  It is not a treatment endpoint.
"""

from __future__ import annotations

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mvc.models.thermal_dose import (
    REFERENCE_CEM43_EQUIVALENT_MINUTES,
    REFERENCE_DISCLAIMER,
    REFERENCE_TEMPERATURE_C,
    ThermalDoseError,
    format_dose_summary,
    mask_exclusion_note,
)


def roi_mask_from_points(points, shape):
    """Rasterize an ROI polygon into a boolean mask of ``shape`` (rows, cols)."""

    if points is None or len(points) < 3:
        return None
    height, width = shape
    if height < 1 or width < 1:
        return None
    path = mpath.Path(points)
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    samples = np.column_stack((columns.ravel(), rows.ravel()))
    return path.contains_points(samples).reshape((height, width))


def create_thermal_dose_dialog(parent, result, recalculate=None):
    """Build the CEM43 dialog.  The caller decides when to show it.

    ``recalculate`` is an optional callback with no arguments that returns a
    new ``ThermalDoseResult`` from the current ROI.  It is the "Update ROI
    curves" button.  Any error from that callback is shown and the previous
    result stays on screen.  A missing file or a bad ROI shape is not a
    ``ThermalDoseError``, and those must not escape the button slot.
    """

    dialog = QDialog(parent)
    dialog.setWindowTitle("CEM43 thermal dose")
    dialog.setMinimumSize(1080, 720)
    layout = QVBoxLayout(dialog)

    title = QLabel(
        "Cumulative thermal dose, Sapareto & Dewey 1984: "
        "CEM43 = Σ Δtᵢ · R^(43−Tᵢ).  "
        "R = 0.25 below 43°C and R = 0.5 at or above 43°C.  "
        "Δt is the DICOM AcquisitionTime interval in minutes."
    )
    title.setWordWrap(True)
    layout.addWidget(title)

    body = QHBoxLayout()
    layout.addLayout(body, stretch=1)

    map_figure, map_axis = plt.subplots(figsize=(5.2, 4.4))
    map_canvas = FigureCanvas(map_figure)
    map_panel = QWidget()
    map_layout = QVBoxLayout(map_panel)
    map_layout.setContentsMargins(0, 0, 0, 0)
    map_layout.addWidget(NavigationToolbar(map_canvas, dialog))
    map_layout.addWidget(map_canvas)
    black_label = QLabel("")
    black_label.setWordWrap(True)
    map_layout.addWidget(black_label)
    body.addWidget(map_panel, stretch=1)

    curve_panel = QWidget()
    curve_layout = QVBoxLayout(curve_panel)
    curve_layout.setContentsMargins(0, 0, 0, 0)
    temp_figure, temp_axis = plt.subplots(figsize=(5.4, 2.6))
    temp_canvas = FigureCanvas(temp_figure)
    dose_figure, dose_axis = plt.subplots(figsize=(5.4, 2.6))
    dose_canvas = FigureCanvas(dose_figure)
    curve_layout.addWidget(temp_canvas)
    curve_layout.addWidget(dose_canvas)
    body.addWidget(curve_panel, stretch=1)

    state = {"result": result}
    # with_extremes arrived in matplotlib 3.4.2; 3.4.0 only has set_bad.
    inferno = plt.cm.inferno
    if hasattr(inferno, "with_extremes"):
        colormap = inferno.with_extremes(bad="black")
    else:
        colormap = inferno.copy()
        colormap.set_bad(color="black")
    initial = np.ma.masked_invalid(np.asarray(result.cem43, dtype=float))
    image = map_axis.imshow(initial, cmap=colormap)
    colorbar = map_figure.colorbar(image, ax=map_axis)
    colorbar.set_label("CEM43 (equivalent minutes)")
    map_axis.set_title("CEM43 map")
    map_axis.set_xticks([])
    map_axis.set_yticks([])

    summary = QLabel(format_dose_summary(result))
    summary.setWordWrap(True)
    layout.addWidget(summary)

    guides = QHBoxLayout()
    temperature_box = QCheckBox(
        f"{REFERENCE_TEMPERATURE_C:g}°C reference line"
    )
    dose_boxes = []
    for level in REFERENCE_CEM43_EQUIVALENT_MINUTES:
        dose_boxes.append(QCheckBox(f"CEM43 = {level:g} reference line"))
    guides.addWidget(temperature_box)
    for box in dose_boxes:
        guides.addWidget(box)
    guides.addStretch(1)
    layout.addLayout(guides)

    disclaimer = QLabel(REFERENCE_DISCLAIMER)
    disclaimer.setWordWrap(True)
    layout.addWidget(disclaimer)

    buttons = QHBoxLayout()
    update_button = QPushButton("Update ROI curves")
    update_button.setEnabled(recalculate is not None)
    close_button = QPushButton("Close")
    buttons.addWidget(update_button)
    buttons.addStretch(1)
    buttons.addWidget(close_button)
    layout.addLayout(buttons)

    def draw_map(dose_result):
        masked = np.ma.masked_invalid(np.asarray(dose_result.cem43, dtype=float))
        image.set_data(masked)
        finite = np.asarray(dose_result.cem43, dtype=float)
        finite = finite[np.isfinite(finite)]
        upper = float(np.max(finite)) if finite.size else 1.0
        image.set_clim(0.0, max(upper, 1e-6))
        map_canvas.draw_idle()

    def draw_curves():
        dose_result = state["result"]
        temp_axis.clear()
        dose_axis.clear()
        elapsed = dose_result.elapsed_minutes
        if dose_result.roi_mean_temperature is None:
            temp_axis.text(
                0.5, 0.5, "No ROI yet",
                ha="center", va="center", transform=temp_axis.transAxes,
            )
            dose_axis.text(
                0.5, 0.5, "No ROI yet",
                ha="center", va="center", transform=dose_axis.transAxes,
            )
        else:
            temp_axis.plot(
                elapsed,
                dose_result.roi_mean_temperature,
                "o-",
                color="tab:blue",
                label="ROI mean temperature (valid on every frame)",
            )
            dose_axis.plot(
                elapsed,
                dose_result.roi_cumulative_mean,
                "o-",
                color="tab:red",
                label="ROI mean cumulative dose (same voxels)",
            )
            dose_axis.plot(
                elapsed,
                dose_result.roi_cumulative_max,
                "s--",
                color="tab:orange",
                label="ROI max cumulative dose (same voxels)",
            )
        temp_axis.set_xlabel("Time from first AcquisitionTime (min)")
        temp_axis.set_ylabel("Temperature (°C)")
        temp_axis.set_title("ROI temperature (voxels valid on every frame)")
        temp_axis.grid(True, alpha=0.3)
        dose_axis.set_xlabel("Time from first AcquisitionTime (min)")
        dose_axis.set_ylabel("CEM43 (equivalent minutes)")
        dose_axis.set_title("ROI cumulative dose (same voxels)")
        dose_axis.grid(True, alpha=0.3)
        if temperature_box.isChecked():
            temp_axis.axhline(
                REFERENCE_TEMPERATURE_C,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                label=f"{REFERENCE_TEMPERATURE_C:g}°C reference",
            )
        for box, level in zip(dose_boxes, REFERENCE_CEM43_EQUIVALENT_MINUTES):
            if box.isChecked():
                dose_axis.axhline(
                    level,
                    color="gray",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"CEM43 = {level:g} reference",
                )
        if temp_axis.get_legend_handles_labels()[0]:
            temp_axis.legend(fontsize=8)
        if dose_axis.get_legend_handles_labels()[0]:
            dose_axis.legend(fontsize=8)
        temp_figure.tight_layout()
        dose_figure.tight_layout()
        temp_canvas.draw_idle()
        dose_canvas.draw_idle()

    def apply_result(dose_result):
        state["result"] = dose_result
        summary.setText(format_dose_summary(dose_result))
        note = mask_exclusion_note(dose_result)
        black = (
            "Black voxels had low signal or no valid temperature and were not accumulated."
        )
        if note:
            black = black + " " + note
        black_label.setText(black)
        draw_map(dose_result)
        draw_curves()

    def on_update_roi():
        if recalculate is None:
            return
        try:
            updated = recalculate()
        except ThermalDoseError as exc:
            QMessageBox.warning(dialog, "CEM43 thermal dose", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(
                dialog,
                "CEM43 thermal dose",
                f"CEM43 was not recalculated: {exc}",
            )
            return
        apply_result(updated)

    temperature_box.stateChanged.connect(lambda _state: draw_curves())
    for box in dose_boxes:
        box.stateChanged.connect(lambda _state: draw_curves())
    update_button.clicked.connect(on_update_roi)
    close_button.clicked.connect(dialog.accept)
    apply_result(result)

    def release_figures():
        plt.close(map_figure)
        plt.close(temp_figure)
        plt.close(dose_figure)

    dialog.finished.connect(release_figures)
    return dialog
