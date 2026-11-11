 # mvc/controllers/advanced_features_controller.py

import os
import threading
import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, QProgressBar
from PyQt5.QtCore import Qt, QTimer
from mvc.controllers.background_work import BackgroundCall
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.animation as animation

class AdvancedFeaturesController:
    """Controller for advanced features like Cinema and Thermostat."""
    
    def __init__(self, view, cinema_model, thermostat_model, registrar=None,
                 registration_config=None):
        """Initialize with references to view and models.

        ``registrar`` and ``registration_config`` are optional.  The GUI
        passes the thermometry controller's registrar and its
        liver-respiratory config, so a checked box registers every later
        frame to the first frame with that preset.  If the caller omits
        the registrar, ``CinemaModel`` creates one and, when no config was
        passed either, uses the same preset.  The sequence still does not
        register unless the checkbox is checked.
        """
        self.view = view
        self.cinema_model = cinema_model
        self.thermostat_model = thermostat_model
        self.registrar = registrar
        self.registration_config = registration_config
        self._cinema_job = None
        self._cinema_stop = None
        
        # Connect signals
        self.connect_signals()
    
    def connect_signals(self):
        """Connect view signals to controller methods."""
        self.view.generate_cinema_signal.connect(self.generate_cinema)
        self.view.generate_thermostat_signal.connect(self.generate_thermostat)
        
        # Connect model signals
        self.cinema_model.calculation_progress.connect(self.update_cinema_progress)
        self.cinema_model.calculation_complete.connect(self.show_cinema_dialog)
        self.cinema_model.error_occurred.connect(self.view.show_error)
        
        self.thermostat_model.calculation_complete.connect(self.show_thermostat_dialog)
        self.thermostat_model.error_occurred.connect(self.view.show_error)
    
    def generate_cinema(self):
        """Generate a temperature map movie from sequence data."""
        try:
            # Check if we have batch-loaded files
            if not hasattr(self.view, 'all_real_files') or not hasattr(self.view, 'all_imag_files'):
                self.view.show_error("No sequence data available. Please use 'Batch Load Folders' first.")
                return
                
            real_files = self.view.all_real_files
            imag_files = self.view.all_imag_files
            
            if len(real_files) < 2 or len(imag_files) < 2:
                self.view.show_error("Need at least 2 real and 2 imaginary files for Cinema feature.")
                return
                
            # Get processing options from view
            original_temperature = self.view.original_temp_input.value()
            apply_gaussian = self.view.gaussian_filter_checkbox.isChecked()
            apply_b0_correction = self.view.b0_correction_checkbox.isChecked()
            apply_motion_correction = self.view.motion_correction_checkbox.isChecked()
            
            if self._cinema_job is not None and self._cinema_job.running():
                self.view.show_error("A temperature sequence is already running")
                return

            # Create progress dialog.  The sequence itself runs on a worker,
            # so this dialog can paint and its Cancel button can be clicked.
            # Cancel stops before the next frame; it does not change a frame
            # that was already registered.
            self.progress_dialog = QDialog(self.view)
            self.progress_dialog.setWindowTitle("Generating Temperature Sequence")
            self.progress_dialog.setFixedSize(400, 140)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            
            layout = QVBoxLayout(self.progress_dialog)
            
            self.progress_label = QLabel("Processing temperature maps...")
            layout.addWidget(self.progress_label)
            
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, len(real_files) - 1)
            self.progress_bar.setValue(0)
            layout.addWidget(self.progress_bar)

            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(self._request_cinema_cancel)
            layout.addWidget(cancel_button)
            self.progress_dialog.rejected.connect(self._request_cinema_cancel)
            
            # Show progress dialog
            self.progress_dialog.show()
            
            # Calculate temperature maps for the sequence
            self.view.update_status("Generating temperature map sequence...")
            self._cinema_stop = threading.Event()
            if self._cinema_job is None:
                self._cinema_job = BackgroundCall()

            def work():
                return self.cinema_model.calculate_temperature_sequence(
                    real_files,
                    imag_files,
                    original_temperature,
                    apply_gaussian,
                    apply_b0_correction,
                    apply_motion_correction=apply_motion_correction,
                    registrar=self.registrar,
                    registration_config=self.registration_config,
                    stop_event=self._cinema_stop,
                )

            self._cinema_job.start(
                work,
                self._cinema_finished,
                self._cinema_failed,
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.view.show_error(f"Error generating cinema: {str(e)}")
            self._close_cinema_progress()

    def _request_cinema_cancel(self):
        """Stop before the next frame.  The current frame is left as it is."""
        if self._cinema_stop is not None:
            self._cinema_stop.set()
        if getattr(self, "progress_label", None) is not None:
            self.progress_label.setText("Cancelling after the current frame...")

    def _cinema_finished(self, sequence):
        """Close the progress dialog if the sequence did not open the player.

        ``calculation_complete`` already opens the player when the model
        returns a list.  An early ``None`` does not emit that signal.
        """
        if sequence is None:
            self._close_cinema_progress()

    def _cinema_failed(self, message):
        self._close_cinema_progress()
        self.view.show_error(f"Error generating cinema: {message}")

    def _close_cinema_progress(self):
        dialog = getattr(self, "progress_dialog", None)
        if dialog is not None:
            dialog.blockSignals(True)
            dialog.close()
            self.progress_dialog = None
    
    def update_cinema_progress(self, current, total):
        """Update progress dialog."""
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.setValue(current)
            self.progress_label.setText(f"Processing temperature maps... ({current+1}/{total})")
    
    def show_cinema_dialog(self, temperature_sequence):
        """Show cinema dialog with the calculated temperature sequence."""
        # Close progress dialog if it exists.  Block signals so the window
        # close is not treated as another cancel request.
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.blockSignals(True)
            self.progress_dialog.close()
            self.progress_dialog = None
        
        # Store the temperature sequence in the view for later use by thermostat
        self.view.temperature_sequence = temperature_sequence
        
        # Check if we have valid data
        if not temperature_sequence or all(t is None for t in temperature_sequence):
            self.view.show_error("Failed to generate temperature maps for the sequence.")
            return
        
        # Create dialog for cinema display
        cinema_dialog = QDialog(self.view)
        cinema_dialog.setWindowTitle("Temperature Map Cinema")
        cinema_dialog.setMinimumSize(800, 600)
        
        # Create layout
        layout = QVBoxLayout(cinema_dialog)
        
        # Create matplotlib figure for animation
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Add navigation toolbar
        toolbar = NavigationToolbar(canvas, cinema_dialog)
        layout.addWidget(toolbar)
        
        # Add slider controls
        slider_layout = QHBoxLayout()
        
        # Frame slider
        frame_slider = QSlider(Qt.Horizontal)
        frame_slider.setMinimum(0)
        frame_slider.setMaximum(len(temperature_sequence) - 1)
        frame_slider.setValue(0)
        frame_slider.setTickPosition(QSlider.TicksBelow)
        frame_slider.setTickInterval(1)
        
        # Frame counter label
        frame_label = QLabel(f"Frame: 1/{len(temperature_sequence)}")
        
        slider_layout.addWidget(QLabel("Frame:"))
        slider_layout.addWidget(frame_slider)
        slider_layout.addWidget(frame_label)
        
        layout.addLayout(slider_layout)
        
        # Add control buttons
        button_layout = QHBoxLayout()
        
        play_button = QPushButton("Play")
        stop_button = QPushButton("Stop")
        save_button = QPushButton("Save Video")
        
        button_layout.addWidget(play_button)
        button_layout.addWidget(stop_button)
        button_layout.addWidget(save_button)
        
        layout.addLayout(button_layout)
        
        # Get window/level settings from main view
        window_center = self.view.window_center_input.value()
        window_width = self.view.window_width_input.value()
        vmin = window_center - window_width / 2
        vmax = window_center + window_width / 2
        
        # Initialize the plot with first frame
        im = ax.imshow(temperature_sequence[0], cmap='jet', vmin=vmin, vmax=vmax)
        colorbar = fig.colorbar(im, ax=ax)
        colorbar.set_label('Temperature (°C)')
        ax.set_title(f'Temperature Map - Frame 1/{len(temperature_sequence)}')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Animation control variables
        animation_running = False
        current_frame = 0
        
        # Function to update frame
        def update_frame(frame_idx):
            if frame_idx < len(temperature_sequence):
                im.set_array(temperature_sequence[frame_idx])
                ax.set_title(f'Temperature Map - Frame {frame_idx+1}/{len(temperature_sequence)}')
                frame_slider.setValue(frame_idx)
                frame_label.setText(f"Frame: {frame_idx+1}/{len(temperature_sequence)}")
                canvas.draw_idle()
                return [im]
            return []
        
        # Timer for manual animation
        timer = QTimer()
        timer.setInterval(500)  # 500ms interval (2 fps)
        
        def timer_update():
            nonlocal current_frame
            current_frame = (current_frame + 1) % len(temperature_sequence)
            update_frame(current_frame)
        
        timer.timeout.connect(timer_update)
        
        # Slider change event
        def on_slider_change():
            nonlocal current_frame
            current_frame = frame_slider.value()
            update_frame(current_frame)
        
        # Connect slider
        frame_slider.valueChanged.connect(on_slider_change)
        
        # Play/Stop functions
        def start_animation():
            nonlocal animation_running
            if not animation_running:
                animation_running = True
                timer.start()
                play_button.setText("Pause")
        
        def stop_animation():
            nonlocal animation_running
            if animation_running:
                animation_running = False
                timer.stop()
                play_button.setText("Play")
        
        # Toggle play/pause
        def toggle_play():
            nonlocal animation_running
            if animation_running:
                stop_animation()
            else:
                start_animation()
        
        # Save video function
        def save_video():
            file_path, _ = QFileDialog.getSaveFileName(
                cinema_dialog, "Save Video", "", "MP4 Files (*.mp4);;GIF Files (*.gif)"
            )
            
            if file_path:
                # Stop any running animation
                stop_animation()
                
                # Get current window/level settings
                window_center = self.view.window_center_input.value()
                window_width = self.view.window_width_input.value()
                vmin = window_center - window_width / 2
                vmax = window_center + window_width / 2
                
                # Save video using model
                success = self.cinema_model.save_sequence_video(
                    file_path, fps=2, vmin=vmin, vmax=vmax
                )
                
                if success:
                    self.view.update_status(f"Video saved to {file_path}")
                else:
                    self.view.show_error("Failed to save video")
        
        # Connect buttons
        play_button.clicked.connect(toggle_play)
        stop_button.clicked.connect(stop_animation)
        save_button.clicked.connect(save_video)
        
        # Memory management - clear sequence data when dialog is closed
        def on_cinema_dialog_close():
            # Clear sequence data to free memory when not needed
            self.cinema_model.clear_sequence()
            
        cinema_dialog.finished.connect(on_cinema_dialog_close)
        
        # Show the dialog
        cinema_dialog.exec_()
    
    def generate_thermostat(self):
        """Generate time-temperature curve for the selected ROI."""
        try:
            # Check if we have temperature sequence data
            if not hasattr(self.view, 'temperature_sequence') or not self.view.temperature_sequence:
                self.view.show_error("No temperature sequence available. Please run Cinema first.")
                return
                
            # Check if we have a valid ROI
            if not self.view.roi_manager.roi_points or len(self.view.roi_manager.roi_points) < 3:
                self.view.show_error("No valid ROI defined. Please draw an ROI first.")
                return
                
            # Make sure ROI mask is calculated
            self.view.roi_manager.calculate_mask()
            if self.view.roi_manager.mask is None:
                self.view.show_error("Failed to create ROI mask.")
                return
                
            # Calculate temperature curve for ROI
            self.view.update_status("Calculating ROI temperature curve...")
            self.thermostat_model.calculate_roi_temperature_curve(
                self.view.temperature_sequence,
                self.view.roi_manager.mask
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.view.show_error(f"Error generating thermostat: {str(e)}")
    
    def show_thermostat_dialog(self, roi_temp_data):
        """Show thermostat dialog with the calculated temperature curve."""
        if not roi_temp_data:
            self.view.show_error("Failed to calculate temperature curve.")
            return
            
        # Create dialog
        thermostat_dialog = QDialog(self.view)
        thermostat_dialog.setWindowTitle("ROI Temperature-Time Curve")
        thermostat_dialog.setMinimumSize(800, 600)
        
        # Create layout
        layout = QVBoxLayout(thermostat_dialog)
        
        # Create matplotlib figure
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Add navigation toolbar
        toolbar = NavigationToolbar(canvas, thermostat_dialog)
        layout.addWidget(toolbar)
        
        # Add save button
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save Data")
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)
        
        # Plot temperature curve with error bars
        time_points = roi_temp_data['time_points']
        mean_temps = roi_temp_data['mean_temps']
        std_devs = roi_temp_data['std_devs']
        min_temps = roi_temp_data['min_temps']
        max_temps = roi_temp_data['max_temps']
        
        # Plot mean temperature with error bars
        ax.errorbar(time_points, mean_temps, yerr=std_devs, fmt='o-', 
                   capsize=5, label='Mean ± Std Dev', color='blue')
        
        # Plot min/max as shaded area
        ax.fill_between(time_points, min_temps, max_temps, alpha=0.2, 
                       color='blue', label='Min-Max Range')
        
        # Set labels and title
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('ROI Temperature vs. Time')
        ax.grid(True)
        ax.legend()
        
        # Function to save data
        def save_data():
            file_path, _ = QFileDialog.getSaveFileName(
                thermostat_dialog, "Save Temperature Data", "", "CSV Files (*.csv)"
            )
            
            if file_path:
                success = self.thermostat_model.save_temperature_curve_data(file_path)
                if success:
                    self.view.update_status(f"Temperature curve data saved to {file_path}")
                else:
                    self.view.show_error("Failed to save temperature curve data")
        
        # Connect save button
        save_button.clicked.connect(save_data)
        
        # Memory management - clear data when dialog is closed
        def on_thermostat_dialog_close():
            # Clear curve data to free memory when not needed
            self.thermostat_model.clear_data()
            
        thermostat_dialog.finished.connect(on_thermostat_dialog_close)
        
        # Show the dialog
        thermostat_dialog.exec_()
 