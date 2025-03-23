# mvc/models/thermostat_model.py

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import gc

class ThermostatModel(QObject):
    """Model for generating time-temperature curves from ROI in temperature sequences."""
    
    # Define signals
    calculation_complete = pyqtSignal(dict)  # Signal to emit when calculation is complete
    error_occurred = pyqtSignal(str)         # Signal to emit when an error occurs
    
    def __init__(self):
        """Initialize the thermostat model."""
        super().__init__()
        self.temperature_curve_data = None
    
    def calculate_roi_temperature_curve(self, temperature_sequence, roi_mask):
        """Calculate temperature statistics over time for a given ROI."""
        if not temperature_sequence or roi_mask is None:
            self.error_occurred.emit("Missing temperature sequence or ROI mask")
            return None
        
        try:
            time_points = list(range(len(temperature_sequence)))
            mean_temps = []
            std_devs = []
            min_temps = []
            max_temps = []
            
            for temp_map in temperature_sequence:
                if temp_map is not None:
                    # Apply mask to temperature map
                    roi_temps = temp_map[roi_mask]
                    
                    if roi_temps.size > 0:
                        mean_temps.append(np.mean(roi_temps))
                        std_devs.append(np.std(roi_temps))
                        min_temps.append(np.min(roi_temps))
                        max_temps.append(np.max(roi_temps))
                    else:
                        mean_temps.append(np.nan)
                        std_devs.append(np.nan)
                        min_temps.append(np.nan)
                        max_temps.append(np.nan)
                else:
                    mean_temps.append(np.nan)
                    std_devs.append(np.nan)
                    min_temps.append(np.nan)
                    max_temps.append(np.nan)
            
            # Store results
            self.temperature_curve_data = {
                'time_points': time_points,
                'mean_temps': mean_temps,
                'std_devs': std_devs,
                'min_temps': min_temps,
                'max_temps': max_temps
            }
            
            # Emit signal with results
            self.calculation_complete.emit(self.temperature_curve_data)
            
            return self.temperature_curve_data
            
        except Exception as e:
            self.error_occurred.emit(f"Error calculating temperature curve: {str(e)}")
            return None
    
    def save_temperature_curve_data(self, file_path):
        """Save temperature curve data to CSV file."""
        if self.temperature_curve_data is None:
            self.error_occurred.emit("No temperature curve data to save")
            return False
        
        try:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['MR Thermometry - ROI Temperature Curve'])
                writer.writerow(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow([])
                
                # Write column headers
                writer.writerow(['Frame', 'Mean Temp (°C)', 'Std Dev', 'Min Temp', 'Max Temp'])
                
                # Write data rows
                for i in range(len(self.temperature_curve_data['time_points'])):
                    writer.writerow([
                        i+1,  # Frame number (1-based)
                        f"{self.temperature_curve_data['mean_temps'][i]:.2f}",
                        f"{self.temperature_curve_data['std_devs'][i]:.2f}",
                        f"{self.temperature_curve_data['min_temps'][i]:.2f}",
                        f"{self.temperature_curve_data['max_temps'][i]:.2f}"
                    ])
                
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error saving temperature curve data: {str(e)}")
            return False
            
    def clear_data(self):
        """Clear temperature curve data to free memory."""
        self.temperature_curve_data = None
        gc.collect()  # Force garbage collection