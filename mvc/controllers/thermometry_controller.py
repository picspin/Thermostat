import os
import numpy as np
import SimpleITK as sitk
import pydicom
from datetime import datetime
import csv

class ThermometryController:
    def __init__(self, model, view):
        """Initialize the controller with model and view references."""
        self.model = model
        self.view = view
        
        # Connect view signals to controller methods
        self.connect_signals()
        
    def connect_signals(self):
        """Connect View signals to Controller methods."""
        # DICOM loading signals
        self.view.load_real1_signal.connect(self.load_real1)
        self.view.load_imag1_signal.connect(self.load_imag1)
        self.view.load_real2_signal.connect(self.load_real2)
        self.view.load_imag2_signal.connect(self.load_imag2)
        self.view.load_b0_map_signal.connect(self.load_b0_map)
        
        # 添加批量加载信号连接
        self.view.batch_load_dicom_signal.connect(self.batch_load_dicom)
        
        # Calculation and reset signals
        self.view.calculate_signal.connect(self.calculate_temperature)
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
        """Apply motion correction using elastic registration."""
        try:
            # Check if we have the necessary images
            if self.model.real1 is None or self.model.real2 is None:
                self.view.show_error("Missing required images for motion correction")
                return False
                
            self.view.update_status("Applying motion correction...")
            
            # Convert numpy arrays to SimpleITK images
            fixed_image = sitk.GetImageFromArray(self.model.real1)
            moving_image = sitk.GetImageFromArray(self.model.real2)
            
            # Set up registration method
            registration_method = sitk.ImageRegistrationMethod()
            
            # Set similarity metric - Mutual Information works well for inter-modal registration
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.1)
            
            # Set optimizer - Gradient Descent
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=1.0, 
                numberOfIterations=100, 
                convergenceMinimumValue=1e-6, 
                convergenceWindowSize=10
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()
            
            # Setup for the multi-resolution framework
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            
            # Set the interpolator
            registration_method.SetInterpolator(sitk.sitkLinear)
            
            # Set the transform - BSpline for elastic registration
            transform_domain_mesh_size = [8] * fixed_image.GetDimension()
            transform = sitk.BSplineTransformInitializer(
                fixed_image, 
                transform_domain_mesh_size
            )
            registration_method.SetInitialTransform(transform)
            
            # Execute registration
            final_transform = registration_method.Execute(fixed_image, moving_image)
            
            # Apply transform to real2 and imag2
            real2_registered = sitk.GetArrayFromImage(
                sitk.Resample(
                    moving_image, 
                    fixed_image, 
                    final_transform, 
                    sitk.sitkLinear, 
                    0.0, 
                    moving_image.GetPixelID()
                )
            )
            
            # Apply same transform to imaginary component
            if self.model.imag2 is not None:
                moving_imag = sitk.GetImageFromArray(self.model.imag2)
                imag2_registered = sitk.GetArrayFromImage(
                    sitk.Resample(
                        moving_imag, 
                        fixed_image, 
                        final_transform, 
                        sitk.sitkLinear, 
                        0.0, 
                        moving_imag.GetPixelID()
                    )
                )
                self.model.imag2 = imag2_registered
            
            # Update the model with registered images
            self.model.real2 = real2_registered
            
            # Calculate DICE coefficient to evaluate registration quality
            fixed_binary = sitk.BinaryThreshold(fixed_image, lowerThreshold=np.mean(self.model.real1), upperThreshold=np.max(self.model.real1))
            moving_binary = sitk.BinaryThreshold(sitk.GetImageFromArray(real2_registered), lowerThreshold=np.mean(real2_registered), upperThreshold=np.max(real2_registered))
            
            overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
            overlap_measures_filter.Execute(fixed_binary, moving_binary)
            dice_coefficient = overlap_measures_filter.GetDiceCoefficient()
            
            self.view.update_status(f"Motion correction applied - DICE coefficient: {dice_coefficient:.3f}")
            
            if dice_coefficient < 0.9:
                self.view.update_status("Warning: Registration quality may be insufficient (DICE < 0.9)")
                
            return True
            
        except Exception as e:
            self.view.show_error(f"Error in motion correction: {str(e)}")
            return False
        
    def calculate_temperature(self, original_temperature, apply_gaussian, apply_b0_correction, apply_motion_correction):
        """Handle temperature calculation request."""
        # Check if we have all required data
        if self.model.real1 is None or self.model.imag1 is None or self.model.real2 is None or self.model.imag2 is None:
            self.view.show_error("Missing required DICOM data. Please load all required DICOM files.")
            return
            
        # Apply motion correction if requested
        if apply_motion_correction:
            success = self.apply_motion_correction()
            if not success:
                self.view.show_error("Motion correction failed. Proceeding without it.")
        
        # Calculate temperature
        success = self.model.calculate_temperature(
            original_temperature=original_temperature,
            apply_gaussian_filter=apply_gaussian,
            apply_b0_correction=apply_b0_correction
        )
        
        if success:
            # Update the view with the temperature map
            self.view.update_temperature_display(self.model.temperature_map)
            self.view.update_status("Temperature calculation complete")
        else:
            self.view.show_error("Temperature calculation failed")
        
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

    def batch_load_dicom(self, folder_paths):
        """Process DICOM files from multiple folders."""
        if not folder_paths:
            return
            
        # 扫描文件夹中的所有DICOM文件
        dicom_files = []
        
        for folder_path in folder_paths:
            # 递归查找文件夹中的所有DICOM文件
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith('.dcm'):
                        dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            self.view.show_error("No DICOM files found in the selected folder(s)")
            return
            
        self.view.update_status(f"Found {len(dicom_files)} DICOM files. Analyzing...")
        
        # 临时存储所有DICOM的采集时间和类型信息
        dicom_acquisitions = []
        
        # 分析所有DICOM文件
        for file_path in dicom_files:
            try:
                ds = pydicom.dcmread(file_path)
                
                # 初始化图像类型和采集时间
                image_type = None
                is_real = False
                is_imaginary = False
                is_b0_map = False
                
                # 直接使用标签(0008,9208)检查ImageType以确定实部/虚部
                # 注意: pydicom需要以16进制访问标签
                try:
                    # 尝试直接访问ImageType标签(0008,9208)
                    image_type_value = str(ds.get((0x0008, 0x9208), ""))
                    if image_type_value:
                        self.view.update_status(f"Found ImageType (0008,9208): {image_type_value}")
                        if 'REAL' in image_type_value.upper():
                            image_type = 'real'
                            is_real = True
                        elif 'IMAGINARY' in image_type_value.upper() or 'IMAG' in image_type_value.upper():
                            image_type = 'imaginary'
                            is_imaginary = True
                except Exception as e:
                    self.view.update_status(f"Error accessing ImageType tag (0008,9208): {e}")
                
                # 如果无法通过(0008,9208)标签确定，尝试通过普通ImageType属性
                if not image_type:
                    if hasattr(ds, 'ImageType') and isinstance(ds.ImageType, (list, tuple)):
                        image_type_str = ' '.join(ds.ImageType).upper()
                        self.view.update_status(f"Using ImageType attribute: {image_type_str}")
                        if 'REAL' in image_type_str:
                            image_type = 'real'
                            is_real = True
                        elif 'IMAGINARY' in image_type_str or 'IMAG' in image_type_str:
                            image_type = 'imaginary'
                            is_imaginary = True
                
                # 直接使用标签(0008,002A)检查AcquisitionDateTime确定采集顺序
                acq_time = None
                try:
                    # 尝试直接访问AcquisitionDateTime标签(0008,002A)
                    acq_datetime = str(ds.get((0x0008, 0x002A), ""))
                    if acq_datetime:
                        acq_time = acq_datetime
                        self.view.update_status(f"Found AcquisitionDateTime (0008,002A): {acq_time}")
                except Exception as e:
                    self.view.update_status(f"Error accessing AcquisitionDateTime tag (0008,002A): {e}")
                
                # 如果无法获取(0008,002A)，尝试其他方式获取时间信息
                if not acq_time:
                    if hasattr(ds, 'AcquisitionDateTime'):
                        acq_time = ds.AcquisitionDateTime
                        self.view.update_status(f"Using AcquisitionDateTime attribute: {acq_time}")
                    elif hasattr(ds, 'SeriesTime'):
                        acq_time = ds.SeriesTime
                        self.view.update_status(f"Using SeriesTime attribute: {acq_time}")
                    elif hasattr(ds, 'SeriesDate') and hasattr(ds, 'SeriesTime'):
                        acq_time = ds.SeriesDate + ds.SeriesTime
                        self.view.update_status(f"Using SeriesDate+SeriesTime: {acq_time}")
                
                # 检查是否是B0 map（通过SeriesDescription中的关键词）
                if hasattr(ds, 'SeriesDescription') and isinstance(ds.SeriesDescription, str):
                    series_desc = ds.SeriesDescription.upper()
                    if 'B0' in series_desc or 'FIELD' in series_desc or 'MAP' in series_desc:
                        is_b0_map = True
                        image_type = 'b0_map'
                        self.view.update_status(f"Found B0 map: {os.path.basename(file_path)}")
                
                # 如果仍无法确定，尝试通过文件名识别
                if not image_type and not is_b0_map:
                    file_name = os.path.basename(file_path).upper()
                    if 'REAL' in file_name:
                        image_type = 'real'
                        is_real = True
                        self.view.update_status(f"Identified as real from filename: {os.path.basename(file_path)}")
                    elif 'IMAG' in file_name:
                        image_type = 'imaginary'
                        is_imaginary = True
                        self.view.update_status(f"Identified as imaginary from filename: {os.path.basename(file_path)}")
                    elif 'B0' in file_name or 'FIELD' in file_name or 'MAP' in file_name:
                        image_type = 'b0_map'
                        is_b0_map = True
                        self.view.update_status(f"Identified as B0 map from filename: {os.path.basename(file_path)}")
                
                # 只有当我们成功识别了类型时才添加文件
                if image_type:
                    dicom_acquisitions.append({
                        'path': file_path,
                        'acq_time': acq_time,
                        'image_type': image_type,
                        'is_real': is_real,
                        'is_imaginary': is_imaginary,
                        'is_b0_map': is_b0_map,
                        'filename': os.path.basename(file_path)
                    })
                    self.view.update_status(f"Added file: {os.path.basename(file_path)} as {image_type}")
                else:
                    self.view.update_status(f"Warning: Could not identify type of {os.path.basename(file_path)}")
                
            except Exception as e:
                self.view.show_error(f"Error reading DICOM file {os.path.basename(file_path)}: {str(e)}")
                continue
        
        # 如果没有成功读取任何DICOM文件
        if not dicom_acquisitions:
            self.view.show_error("No valid DICOM files found or cannot identify file types")
            return
        
        # 按时间排序（对非B0文件）
        real_files = [d for d in dicom_acquisitions if d['is_real']]
        imag_files = [d for d in dicom_acquisitions if d['is_imaginary']]
        b0_files = [d for d in dicom_acquisitions if d['is_b0_map']]
        
        self.view.update_status(f"Before sorting - Found {len(real_files)} real files, {len(imag_files)} imaginary files, {len(b0_files)} B0 map files")
        
        # 单独排序实部和虚部文件根据采集时间，确保较小的值是第1个
        def sort_by_acq_time(files):
            # 首先检查是否所有文件都有采集时间
            if all(f['acq_time'] for f in files):
                return sorted(files, key=lambda x: str(x['acq_time']))
            else:
                self.view.update_status("Warning: Not all files have acquisition time. Ordering may not be accurate.")
                # 对于没有时间的文件，使用一个大值作为默认值
                return sorted(files, key=lambda x: str(x['acq_time'] or '999999'))
        
        # 分别排序实部和虚部文件
        if len(real_files) >= 2:
            real_files = sort_by_acq_time(real_files)
            self.view.update_status("Real files sorted by acquisition time")
            for i, file in enumerate(real_files):
                self.view.update_status(f"Sorted real file {i+1}: {file['filename']}, time: {file['acq_time']}")
        
        if len(imag_files) >= 2:
            imag_files = sort_by_acq_time(imag_files)
            self.view.update_status("Imaginary files sorted by acquisition time")
            for i, file in enumerate(imag_files):
                self.view.update_status(f"Sorted imaginary file {i+1}: {file['filename']}, time: {file['acq_time']}")
        
        # 确定第一组和第二组
        if len(real_files) >= 2:
            self.load_real1(real_files[0]['path'])
            self.load_real2(real_files[1]['path'])
            self.view.update_status(f"Loaded 1st real: {real_files[0]['filename']} (time: {real_files[0]['acq_time']})")
            self.view.update_status(f"Loaded 2nd real: {real_files[1]['filename']} (time: {real_files[1]['acq_time']})")
        elif len(real_files) == 1:
            self.load_real1(real_files[0]['path'])
            self.view.update_status(f"Warning: Only one real image found: {real_files[0]['filename']}. Need two for temperature calculation.")
        else:
            self.view.update_status("Warning: No real images found. Cannot calculate temperature.")
        
        if len(imag_files) >= 2:
            self.load_imag1(imag_files[0]['path'])
            self.load_imag2(imag_files[1]['path'])
            self.view.update_status(f"Loaded 1st imaginary: {imag_files[0]['filename']} (time: {imag_files[0]['acq_time']})")
            self.view.update_status(f"Loaded 2nd imaginary: {imag_files[1]['filename']} (time: {imag_files[1]['acq_time']})")
        elif len(imag_files) == 1:
            self.load_imag1(imag_files[0]['path'])
            self.view.update_status(f"Warning: Only one imaginary image found: {imag_files[0]['filename']}. Need two for temperature calculation.")
        else:
            self.view.update_status("Warning: No imaginary images found. Cannot calculate temperature.")
        
        if b0_files:
            self.load_b0_map(b0_files[0]['path'])
            self.view.update_status(f"Loaded B0 map: {b0_files[0]['filename']}")
        
        # 显示加载结果摘要
        loaded_files = []
        if self.model.real1 is not None: loaded_files.append("1st Real")
        if self.model.imag1 is not None: loaded_files.append("1st Imaginary")
        if self.model.real2 is not None: loaded_files.append("2nd Real")
        if self.model.imag2 is not None: loaded_files.append("2nd Imaginary")
        if self.model.dem_b0_map is not None: loaded_files.append("B0 Map")
        
        summary = "Loaded DICOM files: " + ", ".join(loaded_files)
        self.view.update_status(summary)

        # Store all real and imaginary files for Cinema feature
        self.view.all_real_files = [file['path'] for file in real_files]
        self.view.all_imag_files = [file['path'] for file in imag_files]

        # Log the stored files
        self.view.update_status(f"Stored {len(self.view.all_real_files)} real files and {len(self.view.all_imag_files)} imaginary files for Cinema feature")
