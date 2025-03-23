from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QFileDialog, QLabel,
    QVBoxLayout, QWidget, QCheckBox, QErrorMessage, QLineEdit,
    QHBoxLayout, QSplitter, QTextEdit, QStatusBar, QGroupBox,
    QFormLayout, QDoubleSpinBox, QPushButton, QMenu, QAction, QInputDialog,
    QDialog, QListWidget, QAbstractItemView, QDialogButtonBox,
    QProgressBar, QSlider
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QCursor, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import os
from datetime import datetime

class ROIManager:
    """Manages ROI drawing and stats calculation."""
    def __init__(self, parent_view):
        self.parent_view = parent_view
        self.roi_points = []
        self.roi_polygon = None
        self.drawing = False
        self.mask = None
        self.stats = None
        
    def start_drawing(self, event):
        """Start drawing ROI at the given event position."""
        if event.button == 3:  # Right mouse button
            self.drawing = True
            self.roi_points = [(event.xdata, event.ydata)]
            self.update_roi()
            
    def continue_drawing(self, event):
        """Continue drawing ROI by adding new point."""
        if self.drawing and event.xdata is not None and event.ydata is not None:
            self.roi_points.append((event.xdata, event.ydata))
            self.update_roi()
            
    def finish_drawing(self, event):
        """Finish drawing ROI and calculate stats."""
        if self.drawing:
            self.drawing = False
            if len(self.roi_points) > 2:  # Ensure we have a valid polygon
                self.calculate_mask()
                self.parent_view.calculate_roi_stats_signal.emit()
            
    def update_roi(self):
        """Update ROI display."""
        # Remove previous polygon if it exists
        if self.roi_polygon is not None:
            self.roi_polygon.remove()
            
        # Draw new polygon if we have at least 2 points
        if len(self.roi_points) >= 2:
            self.roi_polygon = Polygon(self.roi_points, fill=False, edgecolor='r', linewidth=2)
            self.parent_view.ax.add_patch(self.roi_polygon)
            
        self.parent_view.canvas.draw()
        
    def calculate_mask(self):
        """Calculate binary mask for the ROI."""
        if not self.roi_points or len(self.roi_points) < 3:
            self.mask = None
            return
            
        # Get current temperature map dimensions
        if self.parent_view.temperature_map is not None:
            height, width = self.parent_view.temperature_map.shape
            
            # Create a mask array
            import matplotlib.path as mpath
            path = mpath.Path(self.roi_points)
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            mask = path.contains_points(points)
            self.mask = mask.reshape(height, width)
        else:
            self.mask = None
            
    def clear(self):
        """Clear ROI data."""
        self.roi_points = []
        if self.roi_polygon is not None:
            try:
                self.roi_polygon.remove()
            except NotImplementedError:
                # 处理matplotlib的已知问题
                pass
            self.roi_polygon = None
        self.drawing = False
        self.mask = None
        self.stats = None
        self.parent_view.canvas.draw()


class ThermometryView(QMainWindow):
    # Define signals for UI events
    load_real1_signal = pyqtSignal(str)
    load_imag1_signal = pyqtSignal(str)
    load_real2_signal = pyqtSignal(str)
    load_imag2_signal = pyqtSignal(str)
    load_b0_map_signal = pyqtSignal(str)
    calculate_signal = pyqtSignal(float, bool, bool, bool)  # Added motion correction
    reset_signal = pyqtSignal()
    save_temperature_map_signal = pyqtSignal(str)
    save_temperature_data_signal = pyqtSignal(str)
    mouse_move_signal = pyqtSignal(float, float)
    calculate_roi_stats_signal = pyqtSignal()
    save_roi_stats_signal = pyqtSignal(str)
    save_roi_mask_signal = pyqtSignal(str, object)  # 新增: 保存ROI mask
    get_saved_rois_signal = pyqtSignal()  # 新增: 获取保存的ROI列表
    load_roi_mask_signal = pyqtSignal(str)  # 新增: 加载ROI mask
    batch_load_dicom_signal = pyqtSignal(list)  # 新增: 批量加载DICOM文件
    generate_cinema_signal = pyqtSignal()  # 新增: 生成温度图电影
    generate_thermostat_signal = pyqtSignal()  # 新增: 生成温度-时间曲线

    def __init__(self):
        super().__init__()
        self.initUI()
        self.colorbar = None
        self.temperature_map = None
        self.roi_manager = ROIManager(self)
        
        # 初始化窗宽窗位
        self.window_center = 37.0
        self.window_width = 10.0
        
        # 设置程序图标 - 使用绝对路径
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            print(f"Icon loaded from: {icon_path}")
        else:
            print(f"Icon not found at: {icon_path}")
        
        # 存储用于Cinema和Thermostat的数据
        self.all_real_files = []
        self.all_imag_files = []
        self.temperature_sequence = []
        
        self.setup_connections()
        self.create_menu_bar()

    def initUI(self):
        self.setWindowTitle('MR Thermometry GUI')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create status bar for cursor position
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.cursor_label = QLabel("Position: N/A")
        self.value_label = QLabel("Value: N/A")
        self.statusBar.addPermanentWidget(self.cursor_label)
        self.statusBar.addPermanentWidget(self.value_label)
        
        # --- Central Widget and Main Splitter ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- Create Left Panel (Controls) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # DICOM Loading Group - 移除Batch Load按钮，只保留单独加载按钮
        dicom_group = QGroupBox("DICOM Loading")
        dicom_layout = QVBoxLayout()
        
        # 保留原有的单独加载按钮（为了向后兼容性）
        load_real1_layout = QHBoxLayout()
        self.load_real1_button = QPushButton('Load 1st Real DICOM', self)
        self.load_real1_button.clicked.connect(self.on_load_real1)
        load_real1_layout.addWidget(self.load_real1_button)
        self.load_imag1_button = QPushButton('Load 1st Imaginary DICOM', self)
        self.load_imag1_button.clicked.connect(self.on_load_imag1)
        load_real1_layout.addWidget(self.load_imag1_button)
        dicom_layout.addLayout(load_real1_layout)
        
        # Load 2nd Real/Imag Buttons
        load_real2_layout = QHBoxLayout()
        self.load_real2_button = QPushButton('Load 2nd Real DICOM', self)
        self.load_real2_button.clicked.connect(self.on_load_real2)
        load_real2_layout.addWidget(self.load_real2_button)
        self.load_imag2_button = QPushButton('Load 2nd Imaginary DICOM', self)
        self.load_imag2_button.clicked.connect(self.on_load_imag2)
        load_real2_layout.addWidget(self.load_imag2_button)
        dicom_layout.addLayout(load_real2_layout)
        
        # B0 Map Button
        self.load_b0_button = QPushButton('Load B0 Map DICOM', self)
        self.load_b0_button.clicked.connect(self.on_load_b0_map)
        dicom_layout.addWidget(self.load_b0_button)
        
        dicom_group.setLayout(dicom_layout)
        left_layout.addWidget(dicom_group)
        
        # Processing Options Group
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()
        
        # Temperature Input
        temp_layout = QFormLayout()
        self.original_temp_input = QDoubleSpinBox(self)
        self.original_temp_input.setRange(20, 45)
        self.original_temp_input.setValue(37.0)
        self.original_temp_input.setSingleStep(0.1)
        self.original_temp_input.setDecimals(1)
        temp_layout.addRow("Baseline Temperature (°C):", self.original_temp_input)
        options_layout.addLayout(temp_layout)
        
        # 添加窗宽窗位控制
        window_layout = QFormLayout()
        self.window_center_input = QDoubleSpinBox(self)
        self.window_center_input.setRange(-100, 100)
        self.window_center_input.setValue(37.0)
        self.window_center_input.setSingleStep(0.5)
        self.window_center_input.valueChanged.connect(self.on_window_level_changed)
        window_layout.addRow("Window Center (°C):", self.window_center_input)
        
        self.window_width_input = QDoubleSpinBox(self)
        self.window_width_input.setRange(0.1, 100)
        self.window_width_input.setValue(10.0)
        self.window_width_input.setSingleStep(0.5)
        self.window_width_input.valueChanged.connect(self.on_window_level_changed)
        window_layout.addRow("Window Width (°C):", self.window_width_input)
        
        # 添加自动窗宽窗位按钮
        auto_window_button = QPushButton("Auto Window", self)
        auto_window_button.clicked.connect(self.on_auto_window_level)
        window_layout.addRow("", auto_window_button)
        
        options_layout.addLayout(window_layout)
        
        # Processing checkboxes
        self.gaussian_filter_checkbox = QCheckBox('Apply Gaussian Filter', self)
        options_layout.addWidget(self.gaussian_filter_checkbox)
        
        self.b0_correction_checkbox = QCheckBox('Apply B0 Correction', self)
        options_layout.addWidget(self.b0_correction_checkbox)
        
        # Motion correction checkbox - New feature
        self.motion_correction_checkbox = QCheckBox('Apply Motion Correction', self)
        self.motion_correction_checkbox.setToolTip('Uses elastic registration with 1st real image as reference')
        options_layout.addWidget(self.motion_correction_checkbox)
        
        options_group.setLayout(options_layout)
        left_layout.addWidget(options_group)
        
        # Calculation Button
        self.calc_button = QPushButton('Calculate Temperature Change', self)
        self.calc_button.clicked.connect(self.on_calculate)
        left_layout.addWidget(self.calc_button)
        
        # ROI Group
        roi_group = QGroupBox("ROI Analysis")
        roi_layout = QVBoxLayout()
        
        self.roi_instructions = QLabel("Right-click and drag to draw ROI")
        roi_layout.addWidget(self.roi_instructions)
        
        self.roi_results = QTextEdit()
        self.roi_results.setReadOnly(True)
        self.roi_results.setMaximumHeight(100)
        roi_layout.addWidget(self.roi_results)
        
        roi_buttons_layout = QHBoxLayout()
        self.clear_roi_button = QPushButton("Clear ROI")
        self.clear_roi_button.clicked.connect(self.on_clear_roi)
        roi_buttons_layout.addWidget(self.clear_roi_button)
        
        self.save_roi_button = QPushButton("Save ROI Stats")
        self.save_roi_button.clicked.connect(self.on_save_roi_stats)
        roi_buttons_layout.addWidget(self.save_roi_button)
        
        roi_layout.addLayout(roi_buttons_layout)
        roi_group.setLayout(roi_layout)
        left_layout.addWidget(roi_group)
        
        # Save/Reset Group
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        self.save_button = QPushButton('Save Temperature Map')
        self.save_button.clicked.connect(self.on_save_temperature_map)
        actions_layout.addWidget(self.save_button)
        
        self.save_data_button = QPushButton('Save Temperature Data')
        self.save_data_button.clicked.connect(self.on_save_temperature_data)
        actions_layout.addWidget(self.save_data_button)
        
        self.clear_button = QPushButton('Reset All')
        self.clear_button.clicked.connect(self.on_reset)
        actions_layout.addWidget(self.clear_button)
        
        actions_group.setLayout(actions_layout)
        left_layout.addWidget(actions_group)
        
        # Add Cinema and Thermostat buttons
        advanced_group = QGroupBox("Advanced Features")
        advanced_layout = QVBoxLayout()

        self.cinema_button = QPushButton('Generate Cinema', self)
        self.cinema_button.setToolTip('Generate temperature map sequence movie')
        advanced_layout.addWidget(self.cinema_button)

        self.thermostat_button = QPushButton('Generate Thermostat', self)
        self.thermostat_button.setToolTip('Generate time-temperature curve for ROI')
        advanced_layout.addWidget(self.thermostat_button)

        advanced_group.setLayout(advanced_layout)
        left_layout.addWidget(advanced_group)

        # Add stretch to push everything up
        left_layout.addStretch()
        
        # --- Create Right Panel (Display) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Display Area
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumWidth(600)
        
        # Connect mouse events for ROI drawing and cursor tracking
        self.canvas.mpl_connect('button_press_event', self.on_canvas_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        
        # DICOM Header Info Text Area
        self.dicom_info = QTextEdit()
        self.dicom_info.setReadOnly(True)
        self.dicom_info.setMaximumHeight(150)
        right_layout.addWidget(QLabel("DICOM Information:"))
        right_layout.addWidget(self.dicom_info)

        # 添加日志显示区
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        self.log_area.setVisible(False)  # 默认隐藏，只在需要时显示
        right_layout.addWidget(QLabel("Processing Log:"))
        right_layout.addWidget(self.log_area)
        
        # 创建垂直布局，用于包含分隔器和作者信息
        main_vertical_layout = QVBoxLayout()
        
        # 添加分隔器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])  # Set initial sizes
        main_vertical_layout.addWidget(splitter)
        
        # 添加作者信息
        author_label = QLabel("Authored by Xiaolei Zhu, zxl1412@gmail.com")
        author_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        author_label.setStyleSheet("color: #666; font-size: 9pt;")  # 设置样式
        main_vertical_layout.addWidget(author_label)
        
        # 将垂直布局添加到主布局
        main_layout.addLayout(main_vertical_layout)
        
    def setup_connections(self):
        """Set up internal signal connections."""
        self.mouse_move_signal.connect(self.update_cursor_info)
        # Connect Cinema and Thermostat buttons
        self.cinema_button.clicked.connect(self.generate_cinema_signal.emit)    # 新增: 生成温度图电影按钮
        self.thermostat_button.clicked.connect(self.generate_thermostat_signal.emit)    # 新增: 生成温度-时间曲线按钮
        
    # UI Event Handlers
    def on_load_real1(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select 1st Real DICOM", "", "DICOM Files (*.dcm)")
        if file:
            self.load_real1_signal.emit(file)
            
    def on_load_imag1(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select 1st Imaginary DICOM", "", "DICOM Files (*.dcm)")
        if file:
            self.load_imag1_signal.emit(file)
            
    def on_load_real2(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select 2nd Real DICOM", "", "DICOM Files (*.dcm)")
        if file:
            self.load_real2_signal.emit(file)
            
    def on_load_imag2(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select 2nd Imaginary DICOM", "", "DICOM Files (*.dcm)")
        if file:
            self.load_imag2_signal.emit(file)
            
    def on_load_b0_map(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select B0 Map DICOM", "", "DICOM Files (*.dcm)")
        if file:
            self.load_b0_map_signal.emit(file)
            self.b0_correction_checkbox.setChecked(True)
                
    def on_calculate(self):
        try:
            original_temperature = self.original_temp_input.value()
            apply_gaussian = self.gaussian_filter_checkbox.isChecked()
            apply_b0_correction = self.b0_correction_checkbox.isChecked()
            apply_motion_correction = self.motion_correction_checkbox.isChecked()
            self.calculate_signal.emit(original_temperature, apply_gaussian, apply_b0_correction, apply_motion_correction)
        except ValueError:
            self.show_error("Please enter a valid number for temperature.")
            
    def on_reset(self):
        self.reset_signal.emit()
        self.reset_ui()
        
    def on_save_temperature_map(self):
        file, _ = QFileDialog.getSaveFileName(self, "Save Temperature Map", "", "JPEG Files (*.jpg);;PNG Files (*.png)")
        if file:
            self.save_temperature_map_signal.emit(file)
            
    def on_save_temperature_data(self):
        file, _ = QFileDialog.getSaveFileName(self, "Save Temperature Data", "", 
                                              "CSV Files (*.csv);;NumPy Files (*.npy)")
        if file:
            self.save_temperature_data_signal.emit(file)
            
    def on_clear_roi(self):
        self.roi_manager.clear()
        self.roi_results.clear()
        
    def on_save_roi_stats(self):
        if self.roi_manager.stats is not None:
            file, _ = QFileDialog.getSaveFileName(self, "Save ROI Statistics", "", "Text Files (*.txt);;CSV Files (*.csv)")
            if file:
                self.save_roi_stats_signal.emit(file)
                
    # Canvas Event Handlers
    def on_canvas_press(self, event):
        if event.inaxes == self.ax:
            self.roi_manager.start_drawing(event)
            
    def on_canvas_motion(self, event):
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.mouse_move_signal.emit(x, y)
            if self.roi_manager.drawing:
                self.roi_manager.continue_drawing(event)
                
    def on_canvas_release(self, event):
        if event.inaxes == self.ax and event.button == 3:  # Right mouse button
            self.roi_manager.finish_drawing(event)
            
    # Display Update Methods
    def update_cursor_info(self, x, y):
        """Update cursor position and value display."""
        x_int, y_int = int(round(x)), int(round(y))
        self.cursor_label.setText(f"Position: ({x_int}, {y_int})")
        
        if self.temperature_map is not None:
            # Ensure coordinates are within bounds
            if (0 <= y_int < self.temperature_map.shape[0] and 
                0 <= x_int < self.temperature_map.shape[1]):
                value = self.temperature_map[y_int, x_int]
                self.value_label.setText(f"Temperature: {value:.2f}°C")
            else:
                self.value_label.setText("Temperature: Out of bounds")
        else:
            self.value_label.setText("Temperature: N/A")
            
    def update_temperature_display(self, temperature_map):
        """Update the temperature map display."""
        self.temperature_map = temperature_map
        
        # Clear the current figure completely
        self.figure.clear()
        
        # Create new axes
        self.ax = self.figure.add_subplot(111)
        
        try:
            # 使用窗宽窗位设置
            window_center = self.window_center_input.value()
            window_width = self.window_width_input.value()
            vmin = window_center - window_width / 2
            vmax = window_center + window_width / 2
            
            # Display temperature map with proper scaling
            im = self.ax.imshow(temperature_map, cmap='jet', vmin=vmin, vmax=vmax)
            
            # Add new colorbar with appropriate ticks
            self.colorbar = self.figure.colorbar(im, ax=self.ax)
            self.colorbar.set_label('Temperature (°C)')
            
            # Set title and labels
            self.ax.set_title('Temperature Map')
            
            # 移除坐标轴刻度，只保留图像
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            # Add timestamp
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.figure.text(0.01, 0.01, f"Generated: {time_str}", fontsize=8)
            
            # Redraw canvas
            self.canvas.draw()
            
            # Update status
            mean_temp = np.mean(temperature_map)
            self.update_status(f"Temperature calculation complete. Mean temperature: {mean_temp:.2f}°C")
            
        except Exception as e:
            print(f"Error updating temperature display: {e}")
            self.show_error("Error updating temperature display")
        
    def update_dicom_info(self, info_text):
        """Update DICOM information display."""
        self.dicom_info.setText(info_text)
        
    def update_roi_display(self, points):
        """Update the ROI display with the given points."""
        if points is not None:
            # Clear existing ROI if any
            self.roi_manager.clear()
            
            # Set the new points
            self.roi_manager.roi_points = points
            
            # Update the display
            self.roi_manager.update_roi()
            
            # Calculate and display ROI stats if temperature map is available
            if self.temperature_map is not None:
                self.calculate_roi_stats_signal.emit()
        else:
            self.roi_manager.clear()
            
    def update_roi_stats(self, stats):
        """Update ROI statistics display."""
        self.roi_manager.stats = stats
        if stats is not None:
            text = (f"ROI Statistics:\n"
                   f"Average: {stats['average']:.2f}°C ± {stats['std_dev']:.2f}°C\n"
                   f"Min: {stats['min']:.2f}°C, Max: {stats['max']:.2f}°C\n"
                   f"Pixels: {stats['count']}")
            self.roi_results.setText(text)
        else:
            self.roi_results.clear()
        
    def update_status(self, message):
        """Update the status bar with a message."""
        self.statusBar.showMessage(message, 5000)  # Show for 5 seconds
        
        # 如果是关于文件识别或加载的日志信息，添加到日志区域
        if any(keyword in message for keyword in ['file', 'Found', 'Loaded', 'Warning', 'sorted', 'time']):
            # 显示日志区域
            self.log_area.setVisible(True)
            
            # 添加时间戳和消息到日志
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}"
            self.log_area.append(log_message)
            
            # 滚动到底部
            scrollbar = self.log_area.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        
    def show_error(self, message):
        """Show an error message."""
        error_dialog = QErrorMessage(self)
        error_dialog.showMessage(message)
        
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu('File')
        
        load_menu = file_menu.addMenu('Load')
        load_real1_action = QAction('Load 1st Real DICOM', self)
        load_real1_action.triggered.connect(self.on_load_real1)
        load_imag1_action = QAction('Load 1st Imaginary DICOM', self)
        load_imag1_action.triggered.connect(self.on_load_imag1)
        load_real2_action = QAction('Load 2nd Real DICOM', self)
        load_real2_action.triggered.connect(self.on_load_real2)
        load_imag2_action = QAction('Load 2nd Imaginary DICOM', self)
        load_imag2_action.triggered.connect(self.on_load_imag2)
        load_b0_action = QAction('Load B0 Map DICOM', self)
        load_b0_action.triggered.connect(self.on_load_b0_map)
        
        load_menu.addAction(load_real1_action)
        load_menu.addAction(load_imag1_action)
        load_menu.addAction(load_real2_action)
        load_menu.addAction(load_imag2_action)
        load_menu.addAction(load_b0_action)
        
        # 创建独立的Batch菜单
        batch_menu = menubar.addMenu('Batch')
        load_folders_action = QAction('Load Folders', self)
        load_folders_action.triggered.connect(self.on_batch_load_folders)
        batch_menu.addAction(load_folders_action)
        
        # 添加Cinema功能
        cinema_action = QAction('Cinema', self)
        cinema_action.triggered.connect(self.on_cinema)
        cinema_action.setToolTip('Generate temperature map sequence movie')
        batch_menu.addAction(cinema_action)
        
        # 添加Thermostat功能
        thermostat_action = QAction('Thermostat', self)
        thermostat_action.triggered.connect(self.on_thermostat)
        thermostat_action.setToolTip('Generate time-temperature curve for ROI')
        batch_menu.addAction(thermostat_action)
        
        save_menu = file_menu.addMenu('Save')
        save_img_action = QAction('Save Temperature Map', self)
        save_img_action.triggered.connect(self.on_save_temperature_map)
        save_data_action = QAction('Save Temperature Data', self)
        save_data_action.triggered.connect(self.on_save_temperature_data)
        save_roi_action = QAction('Save ROI Statistics', self)
        save_roi_action.triggered.connect(self.on_save_roi_stats)
        
        save_menu.addAction(save_img_action)
        save_menu.addAction(save_data_action)
        save_menu.addAction(save_roi_action)
        
        file_menu.addSeparator()
        close_action = QAction('Close', self)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        
        # Tools Menu
        tools_menu = menubar.addMenu('Tools')
        
        roi_action = QAction('Draw ROI', self)
        roi_action.triggered.connect(lambda: self.roi_instructions.setText("Right-click and drag to draw ROI"))
        tools_menu.addAction(roi_action)
        
        clear_roi_action = QAction('Clear ROI', self)
        clear_roi_action.triggered.connect(self.on_clear_roi)
        tools_menu.addAction(clear_roi_action)
        
        # 添加ROI保存/加载子菜单
        roi_menu = tools_menu.addMenu('ROI Management')
        
        save_roi_mask_action = QAction('Save ROI Mask', self)
        save_roi_mask_action.triggered.connect(self.on_save_roi_mask)
        roi_menu.addAction(save_roi_mask_action)
        
        load_roi_mask_action = QAction('Load ROI Mask', self)
        load_roi_mask_action.triggered.connect(self.on_load_roi_mask)
        roi_menu.addAction(load_roi_mask_action)
        
        tools_menu.addSeparator()
        
        reset_action = QAction('Reset All', self)
        reset_action.triggered.connect(self.on_reset)
        tools_menu.addAction(reset_action)

    def reset_ui(self):
        """Reset the UI elements."""
        self.original_temp_input.setValue(37.0)
        self.gaussian_filter_checkbox.setChecked(False)
        self.b0_correction_checkbox.setChecked(False)
        self.motion_correction_checkbox.setChecked(False)
        self.dicom_info.clear()
        self.roi_results.clear()
        self.temperature_map = None
        
        # Clear plot
        self.ax.clear()
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except:
                pass
            self.colorbar = None
        self.canvas.draw()
        
        # Clear ROI - 修改这部分以避免错误
        try:
            self.roi_manager.clear()
        except Exception as e:
            print(f"Warning: Could not clear ROI: {e}")
            # 手动重置ROI管理器
            self.roi_manager = ROIManager(self)
        
        # Clear cursor info
        self.cursor_label.setText("Position: N/A")
        self.value_label.setText("Value: N/A")
        
        # Update status
        self.update_status("Ready")

    def on_save_roi_mask(self):
        """Save current ROI as a mask file."""
        if not self.roi_manager.roi_points or len(self.roi_manager.roi_points) < 3:
            self.show_error("No valid ROI to save")
            return
            
        name, ok = QInputDialog.getText(self, "Save ROI", "Enter ROI name:")
        if ok and name:
            self.save_roi_mask_signal.emit(name, self.roi_manager.roi_points)
            self.update_status(f"ROI saved as '{name}'")
    
    def on_load_roi_mask(self):
        """Load a saved ROI mask."""
        # Get list of saved ROIs
        self.get_saved_rois_signal.emit()
        
    def show_roi_list(self, roi_names):
        """Display a dialog with saved ROI names and let user select one to load."""
        if not roi_names:
            self.show_error("No saved ROIs found")
            return
            
        name, ok = QInputDialog.getItem(self, "Load ROI", 
                                        "Select a saved ROI:", 
                                        roi_names, 0, False)
        if ok and name:
            self.load_roi_mask_signal.emit(name)
            
    def on_batch_load_folders(self):
        """使用自定义对话框批量加载多个文件夹"""
        dialog = FolderSelectionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            selected_folders = dialog.get_selected_folders()
            
            if selected_folders:
                self.batch_load_dicom_signal.emit(selected_folders)
                self.update_status(f"Scanning {len(selected_folders)} folders for DICOM files...")
            else:
                self.show_error("No folders selected")

    def on_window_level_changed(self):
        """Handle window/level change for temperature map display."""
        if self.temperature_map is not None:
            self.update_temperature_display(self.temperature_map)
            
    def on_auto_window_level(self):
        """Automatically set window/level based on temperature map statistics."""
        if self.temperature_map is not None:
            # Calculate appropriate window center and width based on data
            vmin, vmax = np.percentile(self.temperature_map, [2, 98])
            window_center = (vmax + vmin) / 2
            window_width = vmax - vmin
            
            # Ensure minimum width of 10°C for better visualization
            if window_width < 10:
                window_width = 10
                
            # Update window/level spinboxes
            self.window_center_input.setValue(window_center)
            self.window_width_input.setValue(window_width)
            
            # Update display
            self.update_temperature_display(self.temperature_map)
            self.update_status(f"Auto window: Center={window_center:.1f}°C, Width={window_width:.1f}°C")
        else:
            self.show_error("No temperature map available")

    def on_batch_load_dicom(self):
        """Redirect to folder loading."""
        self.on_batch_load_folders()

    def on_cinema(self):
        """生成温度图变化电影。"""
        # 检查是否有足够的数据
        if not hasattr(self, 'all_real_files') or not hasattr(self, 'all_imag_files'):
            self.show_error("No sequence data available. Please use 'Batch Load Folders' first.")
            return
            
        # 发送信号到控制器
        self.generate_cinema_signal.emit()
        
    def on_thermostat(self):
        """生成ROI温度-时间曲线。"""
        # 检查是否有温度图和ROI
        if not hasattr(self, 'temperature_sequence') or not self.roi_manager.roi_points:
            if not hasattr(self, 'temperature_sequence'):
                self.show_error("No temperature sequence available. Please run 'Cinema' first.")
            elif not self.roi_manager.roi_points:
                self.show_error("No ROI defined. Please draw an ROI first.")
            return
            
        # 发送信号到控制器
        self.generate_thermostat_signal.emit()

class FolderSelectionDialog(QDialog):
    """自定义文件夹选择对话框，支持多选文件夹"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select DICOM Folders")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        
        layout = QVBoxLayout()
        
        # 添加说明标签
        instructions_layout = QVBoxLayout()
        label1 = QLabel("1. Click 'Add Folder' to add DICOM folders to the list")
        label2 = QLabel("2. Hold Ctrl key to select multiple folders in the list")
        label3 = QLabel("3. Click 'OK' to process selected folders")
        label4 = QLabel("Note: All folders will be scanned recursively for DICOM files")
        
        # 使说明文本更加突出
        for label in [label1, label2, label3, label4]:
            label.setStyleSheet("font-weight: bold; color: #0066cc;")
        
        instructions_layout.addWidget(label1)
        instructions_layout.addWidget(label2)
        instructions_layout.addWidget(label3)
        instructions_layout.addWidget(label4)
        layout.addLayout(instructions_layout)
        
        # 增加间距
        layout.addSpacing(10)
        
        # 文件夹列表与按钮的水平布局
        folders_layout = QHBoxLayout()
        
        # 创建文件夹列表控件
        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QAbstractItemView.MultiSelection)
        folders_layout.addWidget(self.folder_list, 3)  # 文件夹列表占3/4宽度
        
        # 按钮的垂直布局
        buttons_layout = QVBoxLayout()
        
        # 添加文件夹按钮
        add_button = QPushButton("Add Folder")
        add_button.clicked.connect(self.add_folder)
        buttons_layout.addWidget(add_button)
        
        # 添加删除文件夹按钮
        remove_button = QPushButton("Remove Folder")
        remove_button.clicked.connect(self.remove_folder)
        buttons_layout.addWidget(remove_button)
        
        # 添加全选按钮
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self.select_all_folders)
        buttons_layout.addWidget(select_all_button)
        
        # 添加清除选择按钮
        clear_selection_button = QPushButton("Clear Selection")
        clear_selection_button.clicked.connect(self.clear_selection)
        buttons_layout.addWidget(clear_selection_button)
        
        # 添加按钮布局到文件夹布局
        folders_layout.addLayout(buttons_layout, 1)  # 按钮占1/4宽度
        
        # 添加文件夹布局到主布局
        layout.addLayout(folders_layout)
        
        # 状态标签
        self.status_label = QLabel("Ready to select folders")
        layout.addWidget(self.status_label)
        
        # 确认和取消按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
    def add_folder(self):
        """添加单个文件夹到列表"""
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder:
            # 检查是否已在列表中
            items = [self.folder_list.item(i).text() for i in range(self.folder_list.count())]
            if folder not in items:
                self.folder_list.addItem(folder)
                self.status_label.setText(f"Added folder: {folder}")
                # 自动选择新添加的文件夹
                self.folder_list.item(self.folder_list.count() - 1).setSelected(True)
    
    def remove_folder(self):
        """从列表中删除选定的文件夹"""
        selected_items = self.folder_list.selectedItems()
        if not selected_items:
            self.status_label.setText("No folders selected to remove")
            return
            
        for item in selected_items:
            row = self.folder_list.row(item)
            self.folder_list.takeItem(row)
            
        self.status_label.setText(f"Removed {len(selected_items)} folder(s)")
    
    def select_all_folders(self):
        """选择列表中的所有文件夹"""
        for i in range(self.folder_list.count()):
            self.folder_list.item(i).setSelected(True)
        
        self.status_label.setText(f"Selected all {self.folder_list.count()} folder(s)")
    
    def clear_selection(self):
        """清除所有选择"""
        for i in range(self.folder_list.count()):
            self.folder_list.item(i).setSelected(False)
            
        self.status_label.setText("Cleared all selections")
    
    def validate_and_accept(self):
        """验证选择并接受对话框"""
        selected_folders = self.get_selected_folders()
        if not selected_folders:
            self.status_label.setText("Please select at least one folder")
            return
            
        self.accept()
    
    def get_selected_folders(self):
        """获取所有选择的文件夹"""
        return [self.folder_list.item(i).text() for i in range(self.folder_list.count()) 
                if self.folder_list.item(i).isSelected()]
