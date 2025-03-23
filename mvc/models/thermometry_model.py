import numpy as np
import pydicom
import os
import json
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import math
from scipy.ndimage import zoom
from PyQt5.QtCore import QObject, pyqtSignal

class ThermometryModel(QObject):
    # Define signals
    temperature_updated = pyqtSignal(object)  # Signal to emit when temperature map is updated
    roi_stats_updated = pyqtSignal(object)    # Signal to emit when ROI stats are updated
    error_occurred = pyqtSignal(str)          # Signal to emit when an error occurs
    roi_loaded = pyqtSignal(object)           # Signal to emit when ROI is loaded
    
    def __init__(self):
        super().__init__()
        self.real1 = None
        self.imag1 = None
        self.real2 = None
        self.imag2 = None
        self.te = None  # Echo time in milliseconds
        self.b0 = None  # Magnetic field strength (T)
        self.dem_b0_map = None
        self.flip_angle = None
        self.temperature_change = None
        self.dicom_header_info = ""
        self.phase1 = None
        self.phase2 = None
        self.phase_diff = None
        self.gyromagnetic_ratio = 42.58  # MHz/T
        self.alpha = -0.01  # ppm/°C, PRF temperature sensitivity coefficient
        self.roi_points = None  # Store ROI points
        self.roi_name = None    # Store ROI name
        
        # 创建ROI保存目录
        self.roi_save_dir = os.path.join(os.path.expanduser("~"), ".mr_thermometry", "rois")
        os.makedirs(self.roi_save_dir, exist_ok=True)

    def _extract_dicom_parameters(self, ds):
        """Extract relevant parameters from DICOM header.
        
        Parameters extracted:
        - EchoTime (TE): Time between excitation and echo in milliseconds
        - MagneticFieldStrength (B0): Static magnetic field strength in Tesla
        - FlipAngle: RF excitation flip angle in degrees
        """
        try:
            # Extract Echo Time (TE) in milliseconds
            if hasattr(ds, 'EchoTime'):
                self.te = float(ds.EchoTime)
                print(f"Extracted TE: {self.te} ms")
            else:
                print("Warning: EchoTime not found in DICOM header")
            
            # Extract Magnetic Field Strength (B0) in Tesla
            if hasattr(ds, 'MagneticFieldStrength'):
                self.b0 = float(ds.MagneticFieldStrength)
                print(f"Extracted B0: {self.b0} T")
            else:
                print("Warning: MagneticFieldStrength not found in DICOM header")
            
            # Extract Flip Angle in degrees
            if hasattr(ds, 'FlipAngle'):
                self.flip_angle = float(ds.FlipAngle)
                print(f"Extracted Flip Angle: {self.flip_angle}°")
            else:
                print("Warning: FlipAngle not found in DICOM header")
                
            # Check if we have the minimum required parameters
            if self.te is None or self.b0 is None:
                error_msg = "Missing required DICOM parameters: "
                if self.te is None:
                    error_msg += "EchoTime "
                if self.b0 is None:
                    error_msg += "MagneticFieldStrength"
                self.error_occurred.emit(error_msg)
                return False
                
            return True
        except Exception as e:
            error_msg = f"Error extracting DICOM parameters: {e}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return False

    def _update_header_info(self, dicom_dataset, image_type):
        """Update DICOM header information with detailed parameter information."""
        info = f"\n--- {image_type} DICOM Info ---\n"
        
        # Basic patient and study information
        basic_info = ['PatientName', 'PatientID', 'StudyDate', 'SeriesDescription']
        for tag in basic_info:
            if hasattr(dicom_dataset, tag):
                info += f"{tag}: {getattr(dicom_dataset, tag)}\n"
        
        # MR-specific parameters
        mr_params = {
            'EchoTime': ('TE', 'ms'),
            'MagneticFieldStrength': ('B0', 'T'),
            'FlipAngle': ('Flip Angle', '°'),
            'RepetitionTime': ('TR', 'ms'),
            'SliceThickness': ('Slice Thickness', 'mm'),
            'PixelSpacing': ('Pixel Spacing', 'mm'),
            'Rows': ('Image Rows', ''),
            'Columns': ('Image Columns', '')
        }
        
        info += "\nMR Parameters:\n"
        for tag, (name, unit) in mr_params.items():
            if hasattr(dicom_dataset, tag):
                value = getattr(dicom_dataset, tag)
                if isinstance(value, (list, tuple)):
                    value = f"{value[0]} x {value[1]}" if len(value) == 2 else str(value)
                info += f"{name}: {value} {unit}\n"
        
        self.dicom_header_info += info

    def load_real1_dicom(self, file_path):
        """Load first real DICOM image."""
        try:
            ds = pydicom.dcmread(file_path)
            self.real1 = ds.pixel_array.astype(float)
            self._extract_dicom_parameters(ds)
            self._update_header_info(ds, "Real 1")
            return True
        except Exception as e:
            print(f"Error loading Real 1 DICOM: {e}")
            return False

    def load_imag1_dicom(self, file_path):
        """Load first imaginary DICOM image."""
        try:
            ds = pydicom.dcmread(file_path)
            self.imag1 = ds.pixel_array.astype(float)
            self._extract_dicom_parameters(ds)
            self._update_header_info(ds, "Imaginary 1")
            return True
        except Exception as e:
            print(f"Error loading Imaginary 1 DICOM: {e}")
            return False

    def load_real2_dicom(self, file_path):
        """Load second real DICOM image."""
        try:
            ds = pydicom.dcmread(file_path)
            self.real2 = ds.pixel_array.astype(float)
            self._extract_dicom_parameters(ds)
            self._update_header_info(ds, "Real 2")
            return True
        except Exception as e:
            print(f"Error loading Real 2 DICOM: {e}")
            return False

    def load_imag2_dicom(self, file_path):
        """Load second imaginary DICOM image."""
        try:
            ds = pydicom.dcmread(file_path)
            self.imag2 = ds.pixel_array.astype(float)
            self._extract_dicom_parameters(ds)
            self._update_header_info(ds, "Imaginary 2")
            return True
        except Exception as e:
            print(f"Error loading Imaginary 2 DICOM: {e}")
            return False

    def load_b0_map(self, file_path):
        """Load B0 map DICOM image."""
        try:
            ds = pydicom.dcmread(file_path)
            self.dem_b0_map = ds.pixel_array.astype(float)
            self._update_header_info(ds, "B0 Map")
            return True
        except Exception as e:
            print(f"Error loading B0 Map DICOM: {e}")
            return False

    # def calculate_temperature(self, original_temperature, apply_gaussian_filter=False, apply_b0_correction=False, apply_motion_correction=False, sigma=1.0):
    #     """Calculate temperature changes using PRF thermometry."""
    #     if self.real1 is None or self.imag1 is None or self.real2 is None or self.imag2 is None:
    #         self.error_occurred.emit("Missing required DICOM data")
    #         return False
        
    #     if self.te is None or self.b0 is None:
    #         self.error_occurred.emit("Missing required DICOM parameters (TE or B0)")
    #         return False

    #     try:
    #         # Apply motion correction if requested
    #         if apply_motion_correction:
    #             print("Applying motion correction...")
    #             # Correct real and imaginary parts separately
    #             self.real2 = self.apply_motion_correction(self.real2, self.real1)
    #             self.imag2 = self.apply_motion_correction(self.imag2, self.imag1)
            
    #         # Normalize complex data
    #         complex1 = self.real1 + 1j * self.imag1
    #         complex2 = self.real2 + 1j * self.imag2
            
    #         # Calculate magnitude for normalization
    #         magnitude1 = np.abs(complex1)
    #         magnitude2 = np.abs(complex2)
            
    #         # Normalize complex data to reduce noise
    #         complex1_norm = complex1 / (magnitude1 + np.finfo(float).eps)
    #         complex2_norm = complex2 / (magnitude2 + np.finfo(float).eps)
            
    #         # Calculate phase difference
    #         self.phase_diff = np.angle(complex2_norm * np.conj(complex1_norm))
            
    #         # Apply Gaussian filter if requested (optimized for noise reduction)
    #         if apply_gaussian_filter:
    #             # Use adaptive sigma based on noise level
    #             noise_level = np.std(self.phase_diff)
    #             adaptive_sigma = sigma * (1 + noise_level)
    #             self.phase_diff = gaussian(self.phase_diff, sigma=adaptive_sigma, preserve_range=True)
            
    #         # Apply B0 correction if requested and available
    #         if apply_b0_correction and self.dem_b0_map is not None:
    #             if self.dem_b0_map.shape != self.phase_diff.shape:
    #                 print(f"Resizing B0 map from {self.dem_b0_map.shape} to {self.phase_diff.shape}")
    #                 zoom_factors = (self.phase_diff.shape[0] / self.dem_b0_map.shape[0], 
    #                               self.phase_diff.shape[1] / self.dem_b0_map.shape[1])
    #                 # Use cubic interpolation for better accuracy
    #                 resized_b0_map = zoom(self.dem_b0_map, zoom_factors, order=3)
                    
    #                 # Calculate normalization factor based on phase difference statistics
    #                 # This helps align the B0 map scale with the phase difference scale
    #                 phase_diff_range = np.percentile(self.phase_diff, 95) - np.percentile(self.phase_diff, 5)
    #                 b0_range = np.percentile(resized_b0_map, 95) - np.percentile(resized_b0_map, 5)
    #                 norm_factor = phase_diff_range / (b0_range + np.finfo(float).eps)
                    
    #                 # Apply normalized B0 correction
    #                 self.phase_diff -= resized_b0_map * norm_factor
    #             else:
    #                 # Calculate normalization factor based on phase difference statistics
    #                 phase_diff_range = np.percentile(self.phase_diff, 95) - np.percentile(self.phase_diff, 5)
    #                 b0_range = np.percentile(self.dem_b0_map, 95) - np.percentile(self.dem_b0_map, 5)
    #                 norm_factor = phase_diff_range / (b0_range + np.finfo(float).eps)
                    
    #                 # Apply normalized B0 correction
    #                 self.phase_diff -= self.dem_b0_map * norm_factor
            
    #         # Convert units for temperature calculation
    #         te_seconds = self.te / 1000.0  # Convert ms to s
    #         gamma_hz_per_tesla = self.gyromagnetic_ratio * 1e6  # Convert MHz/T to Hz/T
    #         alpha_per_celsius = self.alpha * 1e-6  # Convert ppm/°C to 1/°C
            
    #         # Calculate temperature change
    #         # ΔT = Δφ / (2π * γ * α * B0 * TE)
    #         denominator = 2 * np.pi * gamma_hz_per_tesla * abs(alpha_per_celsius) * self.b0 * te_seconds
    #         self.temperature_change = self.phase_diff / denominator
            
    #         # Calculate absolute temperature from temperature change (add baseline)
    #         self.temperature_map = original_temperature + self.temperature_change
            
    #         # Emit signal to notify view
    #         self.temperature_updated.emit(self.temperature_map)
            
    #         return True
    #     except Exception as e:
    #         error_msg = f"Error calculating temperature: {e}"
    #         print(error_msg)
    #         self.error_occurred.emit(error_msg)
    #         return False 
    
    def calculate_temperature(self, original_temperature, apply_gaussian_filter=False, apply_b0_correction=False, apply_motion_correction=False, sigma=1.0):
        """Calculate temperature changes using PRF thermometry with improved filtering."""
        if self.real1 is None or self.imag1 is None or self.real2 is None or self.imag2 is None:
            self.error_occurred.emit("Missing required DICOM data")
            return False
        
        if self.te is None or self.b0 is None:
            self.error_occurred.emit("Missing required DICOM parameters (TE or B0)")
            return False

        try:
            # Apply motion correction if requested
            if apply_motion_correction:
                print("Applying motion correction...")
                # Correct real and imaginary parts separately
                self.real2 = self.apply_motion_correction(self.real2, self.real1)
                self.imag2 = self.apply_motion_correction(self.imag2, self.imag1)
            
            # Normalize complex data
            complex1 = self.real1 + 1j * self.imag1
            complex2 = self.real2 + 1j * self.imag2
            
            # Calculate magnitude for normalization
            magnitude1 = np.abs(complex1)
            magnitude2 = np.abs(complex2)
            
            # Create binary masks for regions with signal
            signal_mask1 = magnitude1 > 0.1 * np.max(magnitude1)
            signal_mask2 = magnitude2 > 0.1 * np.max(magnitude2)
            valid_signal_mask = signal_mask1 & signal_mask2
            
            # Normalize complex data to reduce noise
            complex1_norm = complex1 / (magnitude1 + np.finfo(float).eps)
            complex2_norm = complex2 / (magnitude2 + np.finfo(float).eps)
            
            # Calculate phase difference
            self.phase_diff = np.angle(complex2_norm * np.conj(complex1_norm))
            
            # Apply advanced Gaussian filtering if requested
            if apply_gaussian_filter:
                self.phase_diff = self._apply_advanced_gaussian_filter(
                    self.phase_diff, valid_signal_mask, sigma=sigma)
            
            # Apply B0 correction if requested and available
            if apply_b0_correction and self.dem_b0_map is not None:
                self.phase_diff = self._apply_improved_b0_correction(
                    self.phase_diff, valid_signal_mask)
            
            # Convert units for temperature calculation
            te_seconds = self.te / 1000.0  # Convert ms to s
            gamma_hz_per_tesla = self.gyromagnetic_ratio * 1e6  # Convert MHz/T to Hz/T
            alpha_per_celsius = self.alpha * 1e-6  # Convert ppm/°C to 1/°C
            
            # Calculate temperature change
            # ΔT = Δφ / (2π * γ * α * B0 * TE)
            denominator = 2 * np.pi * gamma_hz_per_tesla * abs(alpha_per_celsius) * self.b0 * te_seconds
            self.temperature_change = self.phase_diff / denominator
            
            # Apply post-processing to remove outliers
            self._remove_temperature_outliers(valid_signal_mask)
            
            # Calculate absolute temperature from temperature change (add baseline)
            self.temperature_map = original_temperature + self.temperature_change
            
            # Emit signal to notify view
            self.temperature_updated.emit(self.temperature_map)
            
            return True
        except Exception as e:
            error_msg = f"Error calculating temperature: {e}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def save_temperature_jpg(self, file_path):
        """Save temperature map as JPG image."""
        if self.temperature_change is None:
            print("No temperature map to save")
            return False
        
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(self.temperature_map, cmap='jet')
            plt.colorbar(label='Temperature (°C)')
            plt.title('Temperature Map')
            plt.savefig(file_path)
            plt.close()
            return True
        except Exception as e:
            print(f"Error saving temperature map: {e}")
            return False
    
    def save_temperature_data(self, file_path):
        """Save temperature map as raw data."""
        if self.temperature_map is None:
            print("No temperature map to save")
            return False
        
        try:
            np.save(file_path, self.temperature_map)
            return True
        except Exception as e:
            print(f"Error saving temperature data: {e}")
            return False
    
    def get_roi_stats(self, mask):
        """Calculate statistics for the given ROI mask."""
        if self.temperature_map is None or mask is None:
            return None
            
        try:
            # Apply mask to temperature map
            roi_temps = self.temperature_map[mask]
            
            if roi_temps.size == 0:
                return None
                
            # Calculate statistics
            stats = {
                'average': np.mean(roi_temps),
                'std_dev': np.std(roi_temps),
                'variance': np.var(roi_temps),
                'min': np.min(roi_temps),
                'max': np.max(roi_temps),
                'count': roi_temps.size
            }
            
            # Emit signal with the statistics
            self.roi_stats_updated.emit(stats)
            
            return stats
        except Exception as e:
            print(f"Error calculating ROI statistics: {e}")
            return None
            
    def get_pixel_value(self, x, y):
        """Get temperature value at a specific pixel coordinate."""
        if self.temperature_map is None:
            return None
            
        try:
            # Ensure coordinates are within bounds
            if (0 <= y < self.temperature_map.shape[0] and 
                0 <= x < self.temperature_map.shape[1]):
                return self.temperature_map[y, x]
            else:
                return None
        except Exception as e:
            print(f"Error getting pixel value: {e}")
            return None
    
    def clear_data(self):
        """Reset all loaded data."""
        self.real1 = None
        self.imag1 = None
        self.real2 = None
        self.imag2 = None
        self.dem_b0_map = None
        self.temperature_change = None
        self.temperature_map = None
        self.phase1 = None
        self.phase2 = None
        self.phase_diff = None
        self.dicom_header_info = ""

    def save_roi(self, name, points):
        """Save ROI to disk."""
        try:
            # Create ROI save directory if it doesn't exist
            os.makedirs(self.roi_save_dir, exist_ok=True)
            
            # Save ROI points to JSON file
            roi_data = {
                'name': name,
                'points': points
            }
            
            # 生成文件名
            filename = os.path.join(self.roi_save_dir, f"{name}.json")
            
            with open(filename, 'w') as f:
                json.dump(roi_data, f)
                
            self.roi_name = name
            self.roi_points = points
            return True
        except Exception as e:
            print(f"Error saving ROI: {e}")
            self.error_occurred.emit(f"Error saving ROI: {str(e)}")
            return False
            
    def load_roi(self, name):
        """Load ROI from disk."""
        try:
            # Generate filename
            filename = os.path.join(self.roi_save_dir, f"{name}.json")
            
            # Check if file exists
            if not os.path.exists(filename):
                self.error_occurred.emit(f"ROI file not found: {name}")
                return False
                
            # Load ROI data
            with open(filename, 'r') as f:
                roi_data = json.load(f)
                
            self.roi_name = roi_data['name']
            self.roi_points = roi_data['points']
            
            # Emit signal to update view
            self.roi_loaded.emit(self.roi_points)
            return True
        except Exception as e:
            print(f"Error loading ROI: {e}")
            self.error_occurred.emit(f"Error loading ROI: {str(e)}")
            return False
            
    def get_saved_roi_names(self):
        """Get list of saved ROI names."""
        try:
            # 确保目录存在
            os.makedirs(self.roi_save_dir, exist_ok=True)
            
            # 获取所有JSON文件
            roi_files = [f for f in os.listdir(self.roi_save_dir) if f.endswith('.json')]
            
            # 提取ROI名称（去掉.json后缀）
            roi_names = [os.path.splitext(f)[0] for f in roi_files]
            
            return roi_names
        except Exception as e:
            print(f"Error getting saved ROI names: {e}")
            self.error_occurred.emit(f"Error getting saved ROI names: {str(e)}")
            return []

    def apply_motion_correction(self, moving_image, fixed_image):
        """Apply B-spline registration for motion correction.
        
        Uses free-form deformation model with B-spline interpolation.
        """
        try:
            from scipy.interpolate import RegularGridInterpolator
            import numpy as np
            
            # Create control point grid (coarse grid for B-spline control points)
            grid_spacing = 10  # Control point spacing
            y = np.arange(0, moving_image.shape[0], grid_spacing)
            x = np.arange(0, moving_image.shape[1], grid_spacing)
            
            # Initialize displacement fields
            dx = np.zeros((len(y), len(x)))
            dy = np.zeros((len(y), len(x)))
            
            # Create full resolution grid
            y_full = np.arange(moving_image.shape[0])
            x_full = np.arange(moving_image.shape[1])
            X_full, Y_full = np.meshgrid(x_full, y_full)
            
            # Optimization parameters
            max_iter = 100
            learning_rate = 0.1
            convergence_threshold = 1e-6
            
            # Iterative optimization
            prev_error = float('inf')
            for iteration in range(max_iter):
                # Interpolate displacement fields to full resolution
                dx_interpolator = RegularGridInterpolator((y, x), dx, bounds_error=False, fill_value=0)
                dy_interpolator = RegularGridInterpolator((y, x), dy, bounds_error=False, fill_value=0)
                
                # Get full resolution displacement fields
                points = np.column_stack((Y_full.flatten(), X_full.flatten()))
                dx_full = dx_interpolator(points).reshape(moving_image.shape)
                dy_full = dy_interpolator(points).reshape(moving_image.shape)
                
                # Apply deformation
                warped_y = Y_full + dy_full
                warped_x = X_full + dx_full
                
                # Interpolate moving image at new positions
                valid_mask = (warped_y >= 0) & (warped_y < moving_image.shape[0]) & \
                           (warped_x >= 0) & (warped_x < moving_image.shape[1])
                
                warped_image = np.zeros_like(moving_image)
                points = np.column_stack((warped_y[valid_mask], warped_x[valid_mask]))
                coords = np.column_stack((Y_full[valid_mask], X_full[valid_mask]))
                
                # Create interpolator for moving image
                moving_interpolator = RegularGridInterpolator(
                    (np.arange(moving_image.shape[0]), np.arange(moving_image.shape[1])),
                    moving_image, bounds_error=False, fill_value=0)
                
                # Apply interpolation
                warped_image[valid_mask] = moving_interpolator(points)
                
                # Calculate error
                error = np.mean((warped_image - fixed_image) ** 2)
                
                # Check convergence
                if abs(error - prev_error) < convergence_threshold:
                    break
                
                # Update displacement fields
                error_gradient = 2 * (warped_image - fixed_image)
                dx -= learning_rate * np.mean(error_gradient[valid_mask] * 
                                           (moving_image[1:, :] - moving_image[:-1, :]))
                dy -= learning_rate * np.mean(error_gradient[valid_mask] * 
                                           (moving_image[:, 1:] - moving_image[:, :-1]))
                
                prev_error = error
            
            return warped_image
            
        except Exception as e:
            error_msg = f"Error in motion correction: {e}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return moving_image  # Return original image if correction fails
    def _apply_advanced_gaussian_filter(self, phase_diff, mask, sigma=1.0):
        """Apply multi-scale Gaussian filtering with edge preservation."""
        try:
            from scipy.ndimage import gaussian_filter
            
            # Make a copy of the phase difference
            filtered_phase = np.copy(phase_diff)
            
            # Step 1: Apply noise estimation in local regions
            patch_size = 7
            noise_map = np.zeros_like(phase_diff)
            
            # Calculate local noise estimates
            for i in range(0, phase_diff.shape[0], patch_size):
                for j in range(0, phase_diff.shape[1], patch_size):
                    i_end = min(i + patch_size, phase_diff.shape[0])
                    j_end = min(j + patch_size, phase_diff.shape[1])
                    
                    patch = phase_diff[i:i_end, j:j_end]
                    patch_mask = mask[i:i_end, j:j_end]
                    
                    if np.sum(patch_mask) > patch_size:  # Enough valid pixels
                        # Use robust estimator (MAD) for noise
                        patch_values = patch[patch_mask]
                        median_val = np.median(patch_values)
                        mad = np.median(np.abs(patch_values - median_val))
                        # MAD to standard deviation conversion factor is 1.4826
                        noise_estimate = 1.4826 * mad
                        noise_map[i:i_end, j:j_end] = noise_estimate
                    else:
                        # Default moderate noise estimate for regions with insufficient data
                        noise_map[i:i_end, j:j_end] = 0.05
            
            # Step 2: Apply multi-scale Gaussian filtering
            # Scale 1: Fine details - small sigma for low-noise regions
            sigma_min = max(0.5, sigma * 0.5)
            # Scale 2: Medium details
            sigma_med = max(1.0, sigma)
            # Scale 3: Coarse filtering for high-noise regions
            sigma_max = max(1.5, sigma * 1.5)
            
            # Normalize noise map to range [0, 1] for adaptive filtering
            if np.max(noise_map) > np.min(noise_map):
                norm_noise = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))
            else:
                norm_noise = np.zeros_like(noise_map)
            
            # Apply different levels of Gaussian filtering
            fine_filter = gaussian_filter(phase_diff, sigma=sigma_min, mode='nearest')
            medium_filter = gaussian_filter(phase_diff, sigma=sigma_med, mode='nearest')
            coarse_filter = gaussian_filter(phase_diff, sigma=sigma_max, mode='nearest')
            
            # Combine filters based on local noise level
            for i in range(phase_diff.shape[0]):
                for j in range(phase_diff.shape[1]):
                    if mask[i, j]:
                        noise_level = norm_noise[i, j]
                        
                        # Low noise: mostly fine filter
                        # High noise: mostly coarse filter
                        if noise_level < 0.3:
                            # Low noise regions - minimal filtering
                            weight_fine = 0.7
                            weight_medium = 0.3
                            weight_coarse = 0.0
                        elif noise_level < 0.7:
                            # Medium noise regions - balanced filtering
                            weight_fine = 0.3
                            weight_medium = 0.6
                            weight_coarse = 0.1
                        else:
                            # High noise regions - aggressive filtering
                            weight_fine = 0.1
                            weight_medium = 0.3
                            weight_coarse = 0.6
                        
                        # Weighted combination
                        filtered_phase[i, j] = (
                            weight_fine * fine_filter[i, j] +
                            weight_medium * medium_filter[i, j] +
                            weight_coarse * coarse_filter[i, j]
                        )
            
            # Step 3: Apply edge-preserving regularization
            # Find edges in the magnitude image
            magnitude = np.abs(self.real1 + 1j * self.imag1)
            from scipy.ndimage import sobel
            edge_h = sobel(magnitude, axis=0)
            edge_v = sobel(magnitude, axis=1)
            edge_magnitude = np.sqrt(edge_h**2 + edge_v**2)
            
            # Normalize edge magnitude to [0, 1]
            if np.max(edge_magnitude) > 0:
                edge_magnitude = edge_magnitude / np.max(edge_magnitude)
            
            # Apply less filtering near edges
            for i in range(phase_diff.shape[0]):
                for j in range(phase_diff.shape[1]):
                    if mask[i, j] and edge_magnitude[i, j] > 0.3:  # Strong edge
                        # Reduce filtering effect at edges by blending with original
                        edge_weight = min(1.0, edge_magnitude[i, j] * 2)  # Scale up edge importance
                        filtered_phase[i, j] = (
                            edge_weight * phase_diff[i, j] + 
                            (1 - edge_weight) * filtered_phase[i, j]
                        )
            
            return filtered_phase
            
        except Exception as e:
            print(f"Error in advanced Gaussian filtering: {e}")
            # Fall back to basic Gaussian filter if advanced filtering fails
            from skimage.filters import gaussian
            return gaussian(phase_diff, sigma=sigma, preserve_range=True)

    def _apply_improved_b0_correction(self, phase_diff, mask):
        """Apply improved B0 field correction with spatial regularization."""
        try:
            # Ensure B0 map exists
            if self.dem_b0_map is None:
                return phase_diff
                
            # Resize B0 map if needed
            if self.dem_b0_map.shape != phase_diff.shape:
                from scipy.ndimage import zoom
                zoom_factors = (phase_diff.shape[0] / self.dem_b0_map.shape[0],
                            phase_diff.shape[1] / self.dem_b0_map.shape[1])
                resized_b0_map = zoom(self.dem_b0_map, zoom_factors, order=3)
            else:
                resized_b0_map = self.dem_b0_map.copy()
                
            # Step 1: Denoise the B0 map
            from scipy.ndimage import gaussian_filter
            b0_smoothed = gaussian_filter(resized_b0_map, sigma=2.0)
            
            # Step 2: Calculate robust percentile range for B0 and phase
            # Use only pixels in the mask for better statistics
            masked_phase = phase_diff[mask]
            masked_b0 = b0_smoothed[mask]
            
            if len(masked_phase) == 0 or len(masked_b0) == 0:
                return phase_diff  # No valid pixels
            
            # Use robust range (10-90 percentile instead of 5-95)
            phase_low, phase_high = np.percentile(masked_phase, [10, 90])
            b0_low, b0_high = np.percentile(masked_b0, [10, 90])
            
            phase_range = phase_high - phase_low
            b0_range = b0_high - b0_low
            
            # Avoid division by zero
            if b0_range < 1e-6:
                return phase_diff
                
            # Step 3: Polynomial fitting for better B0-phase relationship
            # Reshape for polynomial fitting
            x = masked_b0.flatten()
            y = masked_phase.flatten()
            
            # Use 2nd-degree polynomial fit for more accurate modeling
            from numpy.polynomial.polynomial import polyfit
            
            # If there are too many points, sample to speed up calculation
            max_points = 10000
            if len(x) > max_points:
                indices = np.random.choice(len(x), max_points, replace=False)
                x_sample = x[indices]
                y_sample = y[indices]
            else:
                x_sample = x
                y_sample = y
                
            # Fit polynomial of degree 2
            coeffs = polyfit(x_sample, y_sample, 2)
            
            # Step 4: Apply correction based on polynomial relationship
            # Create polynomial correction map
            correction_map = np.zeros_like(phase_diff)
            for i in range(len(coeffs)):
                correction_map += coeffs[i] * (b0_smoothed ** i)
            
            # Step 5: Apply spatial regularization to the correction
            correction_smooth = gaussian_filter(correction_map, sigma=1.5)
            
            # Step 6: Apply weighted correction
            # Initialize corrected phase map
            corrected_phase = np.copy(phase_diff)
            
            # Apply different correction weights based on signal quality
            magnitude = np.abs(self.real1 + 1j * self.imag1)
            normalized_magnitude = magnitude / np.max(magnitude + np.finfo(float).eps)
            
            # Scale correction by signal strength - less correction in low signal areas
            for i in range(phase_diff.shape[0]):
                for j in range(phase_diff.shape[1]):
                    if mask[i, j]:
                        # Higher weight in high signal regions
                        correction_weight = min(1.0, normalized_magnitude[i, j] * 1.5)
                        corrected_phase[i, j] -= correction_smooth[i, j] * correction_weight
            
            return corrected_phase
            
        except Exception as e:
            print(f"Error in improved B0 correction: {e}")
            return phase_diff  # Return uncorrected phase if correction fails

    def _remove_temperature_outliers(self, mask, percentile_threshold=99.5):
        """Remove extreme temperature outliers while preserving anatomical features."""
        try:
            if self.temperature_change is None:
                return
                
            # Only consider pixels within the mask
            valid_temps = self.temperature_change[mask]
            
            if len(valid_temps) == 0:
                return
                
            # Get valid temperature range (more robust than simple min/max)
            lower_bound, upper_bound = np.percentile(valid_temps, [0.5, percentile_threshold])
            
            # Calculate IQR-based threshold for outlier detection
            q1, q3 = np.percentile(valid_temps, [25, 75])
            iqr = q3 - q1
            iqr_lower = q1 - 1.5 * iqr
            iqr_upper = q3 + 1.5 * iqr
            
            # Use the more conservative bound
            min_valid = max(lower_bound, iqr_lower)
            max_valid = min(upper_bound, iqr_upper)
            
            # Create outlier mask
            outlier_mask = (self.temperature_change < min_valid) | (self.temperature_change > max_valid)
            outlier_mask = outlier_mask & mask  # Only within signal mask
            
            # Replace outliers with local median
            window_size = 5
            padded_temp = np.pad(self.temperature_change, window_size//2, mode='reflect')
            
            for i, j in zip(*np.where(outlier_mask)):
                # Extract neighborhood
                neighborhood = padded_temp[
                    i:i+window_size,
                    j:j+window_size
                ]
                # Compute median of valid neighbors
                self.temperature_change[i, j] = np.median(neighborhood)
                
        except Exception as e:
            print(f"Error removing temperature outliers: {e}")
