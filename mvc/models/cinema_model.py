# mvc/models/cinema_model.py

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os
import cv2
from datetime import datetime
import gc

class CinemaModel(QObject):
    """Model for generating temperature map sequence movies."""
    
    # Define signals
    calculation_progress = pyqtSignal(int, int)  # Signal to emit progress (current, total)
    calculation_complete = pyqtSignal(list)      # Signal to emit when sequence calculation is complete
    error_occurred = pyqtSignal(str)             # Signal to emit when an error occurs
    
    def __init__(self, thermometry_model):
        """Initialize with reference to the main thermometry model."""
        super().__init__()
        self.thermometry_model = thermometry_model
        self.temperature_sequence = []
        
    def calculate_temperature_sequence(self, real_files, imag_files, original_temperature=37.0, 
                                      apply_gaussian=False, apply_b0_correction=False):
        """Calculate temperature maps for a sequence of real/imaginary file pairs."""
        if len(real_files) < 2 or len(imag_files) < 2:
            self.error_occurred.emit("Need at least 2 real and 2 imaginary files for sequence calculation")
            return None
        
        # Clear previous sequence
        self.temperature_sequence = []
        
        # Calculate total number of pairs
        total_pairs = len(real_files) - 1
        
        # Process each consecutive pair of real/imaginary files
        for i in range(total_pairs):
            # Emit progress
            self.calculation_progress.emit(i, total_pairs)
            
            try:
                # Load first pair
                self.thermometry_model.load_real1_dicom(real_files[i])
                self.thermometry_model.load_imag1_dicom(imag_files[i])
                
                # Load second pair
                self.thermometry_model.load_real2_dicom(real_files[i+1])
                self.thermometry_model.load_imag2_dicom(imag_files[i+1])
                
                # Calculate temperature
                success = self.thermometry_model.calculate_temperature(
                    original_temperature=original_temperature,
                    apply_gaussian_filter=apply_gaussian,
                    apply_b0_correction=apply_b0_correction
                )
                
                if success and self.thermometry_model.temperature_map is not None:
                    self.temperature_sequence.append(self.thermometry_model.temperature_map.copy())
                else:
                    # If calculation fails, add None to maintain sequence
                    self.temperature_sequence.append(None)
                    self.error_occurred.emit(f"Failed to calculate temperature for pair {i+1}")
            
            except Exception as e:
                self.error_occurred.emit(f"Error processing pair {i+1}: {str(e)}")
                self.temperature_sequence.append(None)
        
        # Emit completion signal with the sequence
        self.calculation_complete.emit(self.temperature_sequence)
        return self.temperature_sequence
    
    def save_sequence_video(self, file_path, fps=2, vmin=None, vmax=None):
        """Save temperature sequence as a video file."""
        if not self.temperature_sequence or len(self.temperature_sequence) == 0:
            self.error_occurred.emit("No temperature sequence to save")
            return False
        
        try:
            # Determine file format
            if file_path.lower().endswith('.mp4'):
                # Use OpenCV for MP4
                return self._save_mp4_video(file_path, fps, vmin, vmax)
            elif file_path.lower().endswith('.gif'):
                # Use matplotlib animation for GIF
                return self._save_gif_video(file_path, fps, vmin, vmax)
            else:
                self.error_occurred.emit("Unsupported video format. Use .mp4 or .gif")
                return False
        
        except Exception as e:
            self.error_occurred.emit(f"Error saving video: {str(e)}")
            return False
    
    def _save_mp4_video(self, file_path, fps=2, vmin=None, vmax=None):
        """Save temperature sequence as MP4 using OpenCV."""
        # Get dimensions from first frame
        first_frame = self.temperature_sequence[0]
        if first_frame is None:
            self.error_occurred.emit("First frame is invalid")
            return False
        
        height, width = first_frame.shape
        
        # Determine color mapping range
        if vmin is None or vmax is None:
            # Calculate appropriate min/max across all frames
            all_temps = np.concatenate([temp.flatten() for temp in self.temperature_sequence if temp is not None])
            vmin, vmax = np.percentile(all_temps, [2, 98])
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
        
        # Process each frame
        for i, temp_map in enumerate(self.temperature_sequence):
            if temp_map is None:
                continue
                
            # Normalize temperature to 0-1 range for colormap
            norm_temp = np.clip((temp_map - vmin) / (vmax - vmin), 0, 1)
            
            # Apply jet colormap (similar to matplotlib's jet)
            # Convert to BGR for OpenCV
            colored_frame = cv2.applyColorMap((norm_temp * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Add frame number text
            cv2.putText(colored_frame, f"Frame {i+1}/{len(self.temperature_sequence)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add timestamp
            time_str = datetime.now().strftime("%Y-%m-%d")
            cv2.putText(colored_frame, time_str, 
                       (width - 200, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame to video
            video.write(colored_frame)
        
        # Release video writer
        video.release()
        return True
    
    def _save_gif_video(self, file_path, fps=2, vmin=None, vmax=None):
        """Save temperature sequence as GIF using matplotlib animation."""
        # Create figure for animation
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Determine color mapping range
        if vmin is None or vmax is None:
            # Calculate appropriate min/max across all frames
            all_temps = np.concatenate([temp.flatten() for temp in self.temperature_sequence if temp is not None])
            vmin, vmax = np.percentile(all_temps, [2, 98])
        
        # Initialize plot with first frame
        im = ax.imshow(self.temperature_sequence[0], cmap='jet', vmin=vmin, vmax=vmax)
        colorbar = fig.colorbar(im, ax=ax)
        colorbar.set_label('Temperature (°C)')
        title = ax.set_title('Temperature Map - Frame 1')
        
        # Function to update frame
        def update_frame(frame_idx):
            if frame_idx < len(self.temperature_sequence) and self.temperature_sequence[frame_idx] is not None:
                im.set_array(self.temperature_sequence[frame_idx])
                title.set_text(f'Temperature Map - Frame {frame_idx+1}')
            return [im, title]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update_frame, frames=len(self.temperature_sequence),
            interval=1000/fps, blit=True
        )
        
        # Save animation
        anim.save(file_path, writer='pillow', fps=fps)
        plt.close(fig)
        
        return True 
    
    def clear_sequence(self):
        """Clear temperature sequence to free memory."""
        self.temperature_sequence = []
        gc.collect()  # Force garbage collection
