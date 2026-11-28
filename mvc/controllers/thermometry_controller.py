import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QProgressDialog, QWidget
from mvc.controllers.background_work import BackgroundCall
from mvc.models.sequence_import import import_sequences
from mvc.models.registration import (
    SimpleITKRegistrar,
    body_part_from_view,
    coarse_liver_available,
    coarse_liver_requested,
    config_for_body_part,
    default_registration_config,
    with_coarse_liver_mask,
)
from datetime import datetime
import csv

class ThermometryController:
    def __init__(self, model, view, registrar=None, registration_config=None):
        """Initialize the controller with model and view references.

        ``registrar`` and ``registration_config`` are optional injection points
        for tests; production code uses the liver-respiratory preset
        (rigid, then a coarse B-spline).  A no-argument RegistrationConfig
        is still the historical single-stage B-spline.
        """
        self.model = model
        self.view = view
        self.registrar = registrar if registrar is not None else SimpleITKRegistrar()
        self.registration_config = (
            registration_config
            if registration_config is not None
            else default_registration_config()
        )
        self._motion_job = None
        self._motion_dialog = None
        self._motion_cancelled = False
        self._wait_cursor = False
        
        # Connect view signals to controller methods
        self.connect_signals()
        
    def connect_signals(self):
        """Connect View signals to Controller methods."""
        # B0 stays a separate optional load.  Baseline and sequence share one signal.
        self.view.load_b0_map_signal.connect(self.load_b0_map)
        
        # Baseline and sequence imports share one signal.  The view sets the role.
        self.view.batch_load_dicom_signal.connect(self._load_from_view)
        
        # Calculation and reset signals.  The direct method stays synchronous
        # for tests and for the no-registration path.  The GUI slot moves only
        # the registration itself off the UI thread.
        self.view.calculate_signal.connect(self.begin_calculate_temperature)
        self.view.reset_signal.connect(self.reset_data)
        
        # Save signals
        self.view.save_temperature_map_signal.connect(self.save_temperature_map)
        self.view.save_temperature_data_signal.connect(self.save_temperature_data)
        
        # ROI signals
        self.view.calculate_roi_stats_signal.connect(self.calculate_roi_stats)
        self.view.save_roi_stats_signal.connect(self.save_roi_stats)
        
        # 添加ROI管理信号连接
        self.view.save_roi_mask_signal.connect(self.save_roi_mask)
        self.view.get_saved_rois_signal.connect(self.get_saved_rois)
        self.view.load_roi_mask_signal.connect(self.load_roi_mask)
        
        # Connect model signals to view methods
        self.model.temperature_updated.connect(self.view.update_temperature_display)
        self.model.roi_stats_updated.connect(self.view.update_roi_stats)
        self.model.error_occurred.connect(self.view.show_error)
        self.model.roi_loaded.connect(self.view.update_roi_display)
        
    def load_real1(self, file_path):
        """Handle loading the 1st real DICOM file."""
        success = self.model.load_real1_dicom(file_path)
        if success:
            self.view.update_status(f"Loaded 1st real DICOM from: {file_path}")
            self.view.update_dicom_info(self.model.dicom_header_info)
        else:
            self.view.show_error("Failed to load 1st real DICOM file")
        
    def load_imag1(self, file_path):
        """Handle loading the 1st imaginary DICOM file."""
        success = self.model.load_imag1_dicom(file_path)
        if success:
            self.view.update_status(f"Loaded 1st imaginary DICOM from: {file_path}")
            self.view.update_dicom_info(self.model.dicom_header_info)
        else:
            self.view.show_error("Failed to load 1st imaginary DICOM file")
        
    def load_real2(self, file_path):
        """Handle loading the 2nd real DICOM file."""
        success = self.model.load_real2_dicom(file_path)
        if success:
            self.view.update_status(f"Loaded 2nd real DICOM from: {file_path}")
            self.view.update_dicom_info(self.model.dicom_header_info)
        else:
            self.view.show_error("Failed to load 2nd real DICOM file")
        
    def load_imag2(self, file_path):
        """Handle loading the 2nd imaginary DICOM file."""
        success = self.model.load_imag2_dicom(file_path)
        if success:
            self.view.update_status(f"Loaded 2nd imaginary DICOM from: {file_path}")
            self.view.update_dicom_info(self.model.dicom_header_info)
        else:
            self.view.show_error("Failed to load 2nd imaginary DICOM file")
        
    def load_b0_map(self, file_path):
        """Handle loading the B0 map DICOM file."""
        success = self.model.load_b0_map(file_path)
        if success:
            self.view.update_status(f"Loaded B0 map from: {file_path}")
            self.view.update_dicom_info(self.model.dicom_header_info)
        else:
            self.view.show_error("Failed to load B0 map DICOM file")
        
    def apply_motion_correction(self):
        """Apply motion correction using elastic registration.

        The registration itself lives in ``mvc.models.registration``.  The
        corrected copy is stored on the model; the acquired ``real2/imag2``
        arrays are never overwritten, so repeated calls and toggling the
        checkbox are non-destructive.
        """
        if self.model.real1 is None or self.model.real2 is None:
            self.view.show_error("Missing required images for motion correction")
            return False

        self.view.update_status("Applying motion correction...")
        try:
            result = self.registrar.register_complex(
                fixed_real=self.model.real1,
                fixed_imag=self.model.imag1,
                moving_real=self.model.real2,
                moving_imag=self.model.imag2,
                config=self._registration_config_for_run(),
                spacing_mm=self._pixel_spacing(),
            )
        except Exception as exc:  # e.g. shape mismatch; keep the UI responsive
            self.model.registration_result = None
            self.view.show_error(f"Error in motion correction: {exc}")
            return False

        return self._accept_registration(result)

    def _registration_status(self, result):
        """One status line, including the diagnostics recorded with the warp.

        A quality warning is part of this same line.  Splitting it into a
        second status call let the following "calculation complete" message
        replace it before it could be read.
        """
        parts = [f"Motion correction applied - DICE coefficient: {result.dice:.3f}"]
        if result.iterations:
            parts.append(f"iterations={result.iterations}")
        if result.stop_condition:
            parts.append(f"stop={result.stop_condition}")
        if np.isfinite(result.ncc_before) and np.isfinite(result.ncc_after):
            parts.append(f"NCC {result.ncc_before:.3f}->{result.ncc_after:.3f}")
        if np.isfinite(result.displacement_rms):
            parts.append(f"displacement RMS {result.displacement_rms:.2f}px")
        if np.isfinite(result.folding_ratio) and result.folding_ratio > 0.0:
            parts.append(f"folding {result.folding_ratio:.3f}")
        if result.status == "warning":
            parts.append(
                "Warning: Registration quality may be insufficient "
                f"(DICE < {self.registration_config.dice_warning_threshold})"
            )
        return "; ".join(parts)
        
    def begin_calculate_temperature(
        self,
        original_temperature,
        apply_gaussian,
        apply_b0_correction,
        apply_motion_correction,
    ):
        """GUI entry.  Registration runs off the UI thread; the numbers do not change.

        Without motion correction this is the synchronous calculation.  With
        motion correction the SimpleITK call runs on a worker.  Cancel discards
        that result and does not calculate a temperature.  A finished
        registration that the user did not cancel uses the same accept and
        fallback rules as ``calculate_temperature``.
        """
        if getattr(self.view, "sequence_busy", False):
            self.view.show_error(
                "A temperature sequence is already running. "
                "Wait for it to finish or cancel it before calculating a temperature."
            )
            return
        missing = (
            self.model.real1 is None
            or self.model.imag1 is None
            or self.model.real2 is None
            or self.model.imag2 is None
        )
        if missing:
            self.view.show_error(
                "Missing required DICOM data. Please load all required DICOM files."
            )
            return
        if not apply_motion_correction:
            self.calculate_temperature(
                original_temperature,
                apply_gaussian,
                apply_b0_correction,
                False,
            )
            return
        self._ensure_motion_job()
        if self._motion_job.running():
            self.view.show_error("Motion correction is already running")
            return

        self._motion_cancelled = False
        self._motion_generation = getattr(self, "_motion_generation", 0) + 1
        generation = self._motion_generation
        snapshot = self._image_snapshot()
        arrays = self._registration_arrays()
        registrar = self.registrar
        # Read the checkbox here, on the UI thread.  The worker must not
        # touch the widget, and the stored preset is left unchanged.
        config = self._registration_config_for_run()
        spacing_mm = self._pixel_spacing()

        def work():
            return registrar.register_complex(
                fixed_real=arrays[0],
                fixed_imag=arrays[1],
                moving_real=arrays[2],
                moving_imag=arrays[3],
                config=config,
                spacing_mm=spacing_mm,
            )

        self.view.update_status("Applying motion correction...")
        self._set_wait_cursor()
        self._set_calc_button_enabled(False)
        self._show_motion_dialog()
        self._motion_job.start(
            work,
            lambda result: self._finish_background_registration(
                result,
                None,
                original_temperature,
                apply_gaussian,
                apply_b0_correction,
                snapshot,
                generation,
            ),
            lambda message: self._finish_background_registration(
                None,
                message,
                original_temperature,
                apply_gaussian,
                apply_b0_correction,
                snapshot,
                generation,
            ),
        )

    def _registration_config_for_run(self):
        """Config for this correction.  The stored preset is not mutated.

        The body part combo picks the preset: abdomen stays on the
        stored config (default liver_respiratory), head and joint
        select rigid-only.  The coarse liver checkbox is only honored
        for abdomen.
        """
        body = body_part_from_view(self.view)
        requested = coarse_liver_requested(self.view) and coarse_liver_available(body)
        base = config_for_body_part(self.registration_config, body)
        return with_coarse_liver_mask(base, requested)

    def calculate_temperature(self, original_temperature, apply_gaussian, apply_b0_correction, apply_motion_correction):
        """Handle temperature calculation request.

        This path stays synchronous.  Direct callers and tests use it, and the
        registration parameters are the same ones the background path uses.
        """
        # Check if we have all required data
        if self.model.real1 is None or self.model.imag1 is None or self.model.real2 is None or self.model.imag2 is None:
            self.view.show_error("Missing required DICOM data. Please load all required DICOM files.")
            return
            
        # Apply motion correction if requested.  A failed correction is
        # reported but the temperature calculation continues on the original
        # images (the historical fallback behavior).
        use_registration = False
        if apply_motion_correction:
            success = self.apply_motion_correction()
            if not success:
                self.view.show_error("Motion correction failed. Proceeding without it.")
            else:
                use_registration = True
        
        self._show_temperature_result(
            original_temperature,
            apply_gaussian,
            apply_b0_correction,
            use_registration,
        )

    def _show_temperature_result(
        self,
        original_temperature,
        apply_gaussian,
        apply_b0_correction,
        use_registration,
    ):
        success = self.model.calculate_temperature(
            original_temperature=original_temperature,
            apply_gaussian_filter=apply_gaussian,
            apply_b0_correction=apply_b0_correction,
            use_registration=use_registration,
        )
        
        if success:
            # Update the view with the temperature map
            self.view.update_temperature_display(self.model.temperature_map)
            if use_registration and self.model.registration_result is not None \
                    and self.model.registration_result.status == "warning":
                warning = self._registration_status(self.model.registration_result)
                self.view.update_status(
                    "Temperature calculation complete. " + warning,
                    timeout_ms=0,
                )
            else:
                self.view.update_status("Temperature calculation complete")
        else:
            self.view.show_error("Temperature calculation failed")

    def _finish_background_registration(
        self,
        result,
        error_message,
        original_temperature,
        apply_gaussian,
        apply_b0_correction,
        snapshot,
        generation,
    ):
        """Apply a worker result on the GUI thread.  Cancel does not warp.

        ``generation`` ignores a result that was cancelled or superseded.
        A stale callback must not enable the button or store a warp over
        the job that replaced it.
        """
        if generation != getattr(self, "_motion_generation", None):
            return
        self._close_motion_dialog()
        self._restore_cursor()
        self._set_calc_button_enabled(True)
        if self._motion_cancelled:
            self.view.update_status("Motion correction cancelled")
            return
        if snapshot != self._image_snapshot():
            self.view.show_error(
                "Images changed during motion correction. The correction was discarded."
            )
            return
        if error_message is not None:
            self.model.registration_result = None
            self.view.show_error(f"Error in motion correction: {error_message}")
            self.view.show_error("Motion correction failed. Proceeding without it.")
            use_registration = False
        else:
            use_registration = self._accept_registration(result)
            if not use_registration:
                self.view.show_error("Motion correction failed. Proceeding without it.")
        self._show_temperature_result(
            original_temperature,
            apply_gaussian,
            apply_b0_correction,
            use_registration,
        )

    def _accept_registration(self, result):
        if result.status == "failed":
            self.model.registration_result = None
            self.view.show_error(f"Error in motion correction: {result.message}")
            return False
        self.model.registration_result = result
        self.view.update_status(self._registration_status(result))
        return True

    def _ensure_motion_job(self):
        if getattr(self, "_motion_job", None) is None:
            self._motion_job = BackgroundCall()
        if not hasattr(self, "_motion_cancelled"):
            self._motion_cancelled = False
            self._motion_dialog = None
            self._wait_cursor = False

    def _pixel_spacing(self):
        return getattr(self.model, "pixel_spacing_mm", None)

    def _image_snapshot(self):
        return (
            id(self.model.real1),
            id(self.model.imag1),
            id(self.model.real2),
            id(self.model.imag2),
        )

    def _registration_arrays(self):
        """Copies, so a later load on the GUI thread cannot change the fit."""
        def copy(array):
            if array is None:
                return None
            return np.array(array, copy=True)

        return (
            copy(self.model.real1),
            copy(self.model.imag1),
            copy(self.model.real2),
            copy(self.model.imag2),
        )

    def _show_motion_dialog(self):
        parent = self.view if isinstance(self.view, QWidget) else None
        dialog = QProgressDialog(
            "Applying motion correction...",
            "Cancel",
            0,
            0,
            parent,
        )
        dialog.setWindowTitle("Motion correction")
        # Non-modal so the main window stays usable while SimpleITK runs.
        # Cancel still drops a result that has not been applied.
        dialog.setWindowModality(Qt.NonModal)
        dialog.setModal(False)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.canceled.connect(self._cancel_motion_correction)
        dialog.show()
        self._motion_dialog = dialog

    def _cancel_motion_correction(self):
        self._motion_cancelled = True
        # The worker may still be inside SimpleITK.  Bump the generation so
        # its late callback cannot apply the warp after the dialog is gone.
        self._motion_generation = getattr(self, "_motion_generation", 0) + 1
        self._close_motion_dialog()
        self._restore_cursor()
        self._set_calc_button_enabled(True)
        if hasattr(self.view, "update_status"):
            self.view.update_status("Cancelling motion correction...")

    def _close_motion_dialog(self):
        dialog = getattr(self, "_motion_dialog", None)
        self._motion_dialog = None
        if dialog is not None:
            dialog.blockSignals(True)
            dialog.close()

    def _set_wait_cursor(self):
        if QApplication.instance() is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self._wait_cursor = True

    def _restore_cursor(self):
        if not getattr(self, "_wait_cursor", False):
            return
        if QApplication.instance() is not None:
            QApplication.restoreOverrideCursor()
        self._wait_cursor = False

    def _set_calc_button_enabled(self, enabled):
        button = getattr(self.view, "calc_button", None)
        if button is not None:
            button.setEnabled(enabled)
        
    def reset_data(self):
        """Handle reset request."""
        self.model.clear_data()
        self.view.reset_ui()
        self.view.update_status("All data reset")
        
    def save_temperature_map(self, file_path):
        """Handle saving temperature map as image."""
        if self.model.temperature_map is None:
            self.view.show_error("No temperature map to save")
            return
            
        success = self.model.save_temperature_jpg(file_path)
        if success:
            self.view.update_status(f"Temperature map saved to {file_path}")
        else:
            self.view.show_error("Failed to save temperature map")
            
    def save_temperature_data(self, file_path):
        """Handle saving temperature data."""
        if self.model.temperature_map is None:
            self.view.show_error("No temperature data to save")
            return
            
        # 根据文件扩展名决定保存格式
        if file_path.lower().endswith('.csv'):
            success = self.save_temperature_as_csv(file_path)
        else:
            success = self.model.save_temperature_data(file_path)
            
        if success:
            self.view.update_status(f"Temperature data saved to {file_path}")
        else:
            self.view.show_error("Failed to save temperature data")
    
    def save_temperature_as_csv(self, file_path):
        """Save temperature data as CSV with mean and variance."""
        try:
            # 计算整体温度统计
            temp_data = self.model.temperature_map
            mean_temp = np.mean(temp_data)
            variance_temp = np.var(temp_data)
            std_dev_temp = np.std(temp_data)
            min_temp = np.min(temp_data)
            max_temp = np.max(temp_data)
            
            # 创建CSV文件
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # 写入时间戳
                writer.writerow(['MR Thermometry Data', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow([]) # 空行
                
                # 写入全图统计数据
                writer.writerow(['全图温度统计'])
                writer.writerow(['Mean ± Variance', f"{mean_temp:.2f} ± {variance_temp:.2f}"])
                writer.writerow(['Standard Deviation', f"{std_dev_temp:.2f}"])
                writer.writerow(['Minimum', f"{min_temp:.2f}"])
                writer.writerow(['Maximum', f"{max_temp:.2f}"])
                writer.writerow([]) # 空行
                
                # 如果有ROI数据，添加ROI统计
                if hasattr(self.view.roi_manager, 'stats') and self.view.roi_manager.stats:
                    roi_stats = self.view.roi_manager.stats
                    writer.writerow(['ROI温度统计'])
                    writer.writerow(['Mean ± Variance', f"{roi_stats['average']:.2f} ± {roi_stats['variance']:.2f}"])
                    writer.writerow(['Standard Deviation', f"{roi_stats['std_dev']:.2f}"])
                    writer.writerow(['Minimum', f"{roi_stats['min']:.2f}"])
                    writer.writerow(['Maximum', f"{roi_stats['max']:.2f}"])
                    writer.writerow(['Pixel Count', f"{roi_stats['count']}"])
                    writer.writerow([]) # 空行
                
                # 写入温度数据的显著值（节省空间）
                writer.writerow(['温度数据采样 (每10个像素)'])
                writer.writerow(['X', 'Y', 'Temperature (°C)'])
                
                # 每10个像素取样一次，防止CSV文件过大
                height, width = temp_data.shape
                for y in range(0, height, 10):
                    for x in range(0, width, 10):
                        writer.writerow([x, y, f"{temp_data[y, x]:.2f}"])
            
            return True
        except Exception as e:
            self.view.show_error(f"Error saving CSV: {str(e)}")
            return False
            
    def calculate_roi_stats(self):
        """Calculate statistics for the current ROI."""
        if self.model.temperature_map is None:
            self.view.show_error("No temperature map available for ROI analysis")
            return
            
        mask = self.view.roi_manager.mask
        if mask is None:
            self.view.show_error("No valid ROI defined")
            return
            
        stats = self.model.get_roi_stats(mask)
        if stats:
            self.view.update_roi_stats(stats)
            self.view.update_status(f"ROI statistics: {stats['average']:.2f}°C ± {stats['std_dev']:.2f}°C")
        else:
            self.view.show_error("Failed to calculate ROI statistics")
            
    def save_roi_stats(self, file_path):
        """Save ROI statistics to file."""
        stats = self.view.roi_manager.stats
        if not stats:
            self.view.show_error("No ROI statistics to save")
            return
            
        try:
            with open(file_path, 'w') as f:
                f.write("MR Thermometry - ROI Statistics\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Average Temperature: {stats['average']:.2f}°C ± {stats['std_dev']:.2f}°C\n")
                f.write(f"Minimum Temperature: {stats['min']:.2f}°C\n")
                f.write(f"Maximum Temperature: {stats['max']:.2f}°C\n")
                f.write(f"Number of Pixels: {stats['count']}\n")
                
            self.view.update_status(f"ROI statistics saved to {file_path}")
            return True
        except Exception as e:
            self.view.show_error(f"Failed to save ROI statistics: {str(e)}")
            return False

    def save_roi_mask(self, name, points):
        """Save ROI to the model."""
        self.model.save_roi(name, points)
        self.view.update_status(f"ROI '{name}' saved successfully")
        
    def get_saved_rois(self):
        """Get list of saved ROIs and display them."""
        roi_names = self.model.get_saved_roi_names()
        self.view.show_roi_list(roi_names)
        
    def load_roi_mask(self, name):
        """Load ROI from the model."""
        success = self.model.load_roi(name)
        if success:
            self.view.update_status(f"ROI '{name}' loaded successfully")
        else:
            self.view.show_error(f"Failed to load ROI '{name}'")

    def _load_from_view(self, folder_paths):
        """GUI slot.  The view names the role and any NIfTI component choice."""
        role = getattr(self.view, "import_role", "sequence") or "sequence"
        pair = getattr(self.view, "import_component_pair", None)
        self.batch_load_dicom(folder_paths, role=role, component_pair=pair)

    def batch_load_dicom(self, folder_paths, role="sequence", component_pair=None):
        """Import a baseline sequence or the comparison / Cinema sequence.

        ``role`` is ``baseline`` or ``sequence``.  Two-frame temperature uses
        the first baseline frame against the first sequence frame.  With no
        baseline, a self-contained sequence uses volume 0 as the reference
        and the next volume as the comparison.  Cinema keeps every sequence
        frame, time-sorted, and its first frame is the reference.
        """
        if getattr(self.view, "sequence_busy", False):
            self.view.show_error(
                "A temperature sequence is already running. "
                "Wait for it to finish or cancel it before loading another sequence."
            )
            return
        if not folder_paths:
            return
        if role not in ("baseline", "sequence"):
            self.view.show_error(f"Unknown import role: {role}")
            return
        try:
            result = import_sequences(folder_paths, component_pair=component_pair)
        except Exception as exc:
            self.view.show_error(f"Failed to import images: {exc}")
            return
        if result.problems:
            self.view.show_error("Could not identify image type:\n" + "\n".join(result.problems))
        if result.skipped:
            self.view.update_status(
                "Skipped unreadable: " + ", ".join(result.skipped)
            )
        if (
            not result.real_sources
            and not result.imag_sources
            and not result.b0_paths
        ):
            if not result.problems:
                detail = ""
                if result.skipped:
                    detail = " Skipped unreadable: " + ", ".join(result.skipped)
                self.view.show_error(
                    "No DICOM or NIfTI images found in the selected path(s)." + detail
                )
            return

        if role == "baseline":
            self.view.baseline_real_files = list(result.real_sources)
            self.view.baseline_imag_files = list(result.imag_sources)
        else:
            self.view.all_real_files = list(result.real_sources)
            self.view.all_imag_files = list(result.imag_sources)
        for path in result.b0_paths:
            self.load_b0_map(path)
        self._apply_two_frame()

        if role == "baseline":
            real_count = len(result.real_sources)
            imag_count = len(result.imag_sources)
            self.view.update_status(
                f"Loaded baseline sequence: {real_count} real frames and "
                f"{imag_count} imaginary frames"
            )
            return

        real_count = len(self.view.all_real_files)
        imag_count = len(self.view.all_imag_files)
        b0_count = len(result.b0_paths)
        if hasattr(self.view, "set_frame_counts"):
            self.view.set_frame_counts(real_count, imag_count, b0_count)
        missing = []
        if real_count <= 0:
            missing.append("real")
        if imag_count <= 0:
            missing.append("imag")
        stored = (
            f"Stored {real_count} real frames and {imag_count} imaginary frames"
        )
        if b0_count:
            stored += f" and {b0_count} B0"
        if missing:
            stored += " (missing: " + ", ".join(missing) + ")"
        else:
            stored += " for Cinema feature"
        self.view.update_status(stored)

    def _apply_two_frame(self):
        """Point the two-frame buffers at the imported first frames.

        A separate baseline wins as the reference.  Otherwise the sequence
        is self-contained: its first frame is the baseline and the next
        frame is the comparison.  Later frames stay on the sequence lists
        for Cinema.
        """
        baseline_real = list(getattr(self.view, "baseline_real_files", None) or [])
        baseline_imag = list(getattr(self.view, "baseline_imag_files", None) or [])
        sequence_real = list(getattr(self.view, "all_real_files", None) or [])
        sequence_imag = list(getattr(self.view, "all_imag_files", None) or [])
        if baseline_real and baseline_imag and sequence_real and sequence_imag:
            self.load_real1(baseline_real[0])
            self.load_imag1(baseline_imag[0])
            self.load_real2(sequence_real[0])
            self.load_imag2(sequence_imag[0])
            return
        if not baseline_real and not baseline_imag:
            if sequence_real:
                self.load_real1(sequence_real[0])
            if sequence_imag:
                self.load_imag1(sequence_imag[0])
            if len(sequence_real) >= 2:
                self.load_real2(sequence_real[1])
            if len(sequence_imag) >= 2:
                self.load_imag2(sequence_imag[1])
            return
        if baseline_real:
            self.load_real1(baseline_real[0])
        if baseline_imag:
            self.load_imag1(baseline_imag[0])
