from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QFileDialog, QLabel,
    QVBoxLayout, QWidget, QCheckBox, QComboBox, QErrorMessage, QLineEdit,
    QHBoxLayout, QSplitter, QTextEdit, QStatusBar, QGroupBox,
    QFormLayout, QDoubleSpinBox, QPushButton, QMenu, QAction, QInputDialog,
    QDialog, QListWidget, QAbstractItemView, QDialogButtonBox,
    QProgressBar, QSlider, QScrollArea, QSizePolicy, QFrame
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
from mvc.models.registration import (
    coarse_liver_checkbox_transition,
    registration_site_from_label,
)

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
    # Define signals for UI events.  Real/imag frames arrive through the
    # baseline and sequence imports, not four single-file signals.
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
        
        # Cinema uses the sequence lists.  Two-frame temperature prefers the
        # baseline lists when the user imported a separate reference.
        self.all_real_files = []
        self.all_imag_files = []
        self.baseline_real_files = []
        self.baseline_imag_files = []
        self.import_role = "sequence"
        self.import_component_pair = None
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
        self.frame_count_label = QLabel("Sequence: not loaded")
        self.statusBar.addPermanentWidget(self.cursor_label)
        self.statusBar.addPermanentWidget(self.value_label)
        self.statusBar.addPermanentWidget(self.frame_count_label)
        
        # --- Central Widget and Main Splitter ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --- Create Left Panel (Controls) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Two imports replace the four single-file real/imag buttons.
        # Window level stays in its own group below this one.
        dicom_group = QGroupBox("Two-frame temperature")
        dicom_layout = QVBoxLayout()
        two_frame_hint = QLabel(
            "Load a baseline sequence and a sequence. Temperature compares "
            "their first frames. B0 is optional."
        )
        two_frame_hint.setWordWrap(True)
        dicom_layout.addWidget(two_frame_hint)

        self.load_baseline_button = QPushButton('Load Baseline Sequence', self)
        self.load_baseline_button.setToolTip(
            'Reference series (the old first frame). DICOM folders, '
            'extensionless DICOM, and .nii/.nii.gz.'
        )
        self.load_baseline_button.clicked.connect(self.on_load_baseline)
        dicom_layout.addWidget(self.load_baseline_button)

        self.load_sequence_button = QPushButton('Load Sequence', self)
        self.load_sequence_button.setToolTip(
            'Series to compare or play. Two-frame uses its first frame. '
            'Cinema and CEM43 use every frame in time order; the first is '
            'the reference. Same files as the baseline import.'
        )
        self.load_sequence_button.clicked.connect(self.on_load_sequence)
        dicom_layout.addWidget(self.load_sequence_button)

        self.load_b0_button = QPushButton('Load B0 Map DICOM', self)
        self.load_b0_button.clicked.connect(self.on_load_b0_map)
        dicom_layout.addWidget(self.load_b0_button)
        
        dicom_group.setLayout(dicom_layout)
        left_layout.addWidget(dicom_group)

        # Window level sits in its own group, just under the two-frame loads,
        # so a short window does not fold these rows inside Processing Options.
        window_group = QGroupBox("Window", left_panel)
        window_layout = QVBoxLayout()
        center_label = QLabel("Window Center (°C):", window_group)
        center_label.setWordWrap(False)
        self.window_center_input = QDoubleSpinBox(self)
        self.window_center_input.setRange(-100, 100)
        self.window_center_input.setValue(37.0)
        self.window_center_input.setSingleStep(0.5)
        self.window_center_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window_center_input.setMinimumHeight(24)
        self.window_center_input.valueChanged.connect(self.on_window_level_changed)
        center_label.setBuddy(self.window_center_input)
        window_layout.addWidget(center_label)
        window_layout.addWidget(self.window_center_input)

        width_label = QLabel("Window Width (°C):", window_group)
        width_label.setWordWrap(False)
        self.window_width_input = QDoubleSpinBox(self)
        self.window_width_input.setRange(0.1, 100)
        self.window_width_input.setValue(10.0)
        self.window_width_input.setSingleStep(0.5)
        self.window_width_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window_width_input.setMinimumHeight(24)
        self.window_width_input.valueChanged.connect(self.on_window_level_changed)
        width_label.setBuddy(self.window_width_input)
        window_layout.addWidget(width_label)
        window_layout.addWidget(self.window_width_input)

        self.auto_window_button = QPushButton("Auto Window", self)
        self.auto_window_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.auto_window_button.setMinimumHeight(28)
        self.auto_window_button.clicked.connect(self.on_auto_window_level)
        window_layout.addWidget(self.auto_window_button)
        window_group.setLayout(window_layout)
        left_layout.addWidget(window_group)
        
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
        self.original_temp_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.original_temp_input.setMinimumHeight(24)
        temp_layout.addRow("Baseline Temperature (°C):", self.original_temp_input)
        options_layout.addLayout(temp_layout)
        
        # Processing checkboxes
        self.gaussian_filter_checkbox = QCheckBox('Apply Gaussian Filter', self)
        options_layout.addWidget(self.gaussian_filter_checkbox)
        
        self.b0_correction_checkbox = QCheckBox('Apply B0 Correction', self)
        options_layout.addWidget(self.b0_correction_checkbox)
        
        # Motion correction checkbox - New feature
        self.motion_correction_checkbox = QCheckBox('Apply Motion Correction', self)
        self.motion_correction_checkbox.setToolTip(
            'Uses elastic registration with the 1st real image as reference. '
            'For a Cinema sequence, every later frame is registered to the first frame. '
            'The mask stays the Otsu tissue mask unless "Use coarse liver ROI" is also checked.'
        )
        options_layout.addWidget(self.motion_correction_checkbox)

        # Body part selection.  Default is "Abdomen" which keeps the existing
        # liver_respiratory() preset.  Head and Joint switch to rigid-only,
        # no 40 mm breathing grid, no coarse liver ROI.
        site_layout = QHBoxLayout()
        site_layout.addWidget(QLabel("Body part:", self))
        self.registration_site_combo = QComboBox(self)
        self.registration_site_combo.addItems(["Abdomen", "Head", "Joint"])
        self.registration_site_combo.setToolTip(
            'Abdomen: rigid + 40 mm B-spline (liver respiratory). '
            'Head / Joint: rigid only, no coarse liver ROI. '
            'Default is Abdomen.'
        )
        self.registration_site_combo.currentTextChanged.connect(
            self._on_registration_site_changed
        )
        site_layout.addWidget(self.registration_site_combo)
        site_layout.addStretch()
        options_layout.addLayout(site_layout)

        # Off by default.  Motion correction keeps the Otsu tissue mask
        # until this is checked; checking it alone does not register.
        # Only available when the body part is Abdomen.
        self.coarse_liver_checkbox = QCheckBox('Use coarse liver ROI', self)
        self.coarse_liver_checkbox.setToolTip(
            'Off by default. When motion correction is also on, the metric '
            'uses a coarse liver interior instead of the Otsu tissue mask: '
            'tissue above the corner air estimate, with about 20 mm of body '
            'wall removed. The ROI is taken from the magnitude image, not '
            'the real component. This is not a liver segmentation. It does '
            'nothing unless motion correction is checked. '
            'Only available for Abdomen body part.'
        )
        options_layout.addWidget(self.coarse_liver_checkbox)
        # Abdomen check state saved while Head or Joint is selected.
        self._coarse_liver_for_abdomen = None
        
        options_group.setLayout(options_layout)
        left_layout.addWidget(options_group)
        
        # Calculation Button — first baseline frame against first sequence frame
        self.calc_button = QPushButton('Calculate Temperature Change', self)
        self.calc_button.setToolTip(
            'Two-frame temperature: first baseline frame versus first sequence frame.'
        )
        self.calc_button.clicked.connect(self.on_calculate)
        left_layout.addWidget(self.calc_button)

        # Sequence path.  Cinema, CEM43, and the ROI curve start here,
        # not only from the Batch menu.
        sequence_group = QGroupBox("Sequence")
        sequence_layout = QVBoxLayout()
        self.sequence_next_label = QLabel(
            "Next: Load Sequence, then Generate Cinema."
        )
        self.sequence_next_label.setWordWrap(True)
        sequence_layout.addWidget(self.sequence_next_label)

        self.cinema_button = QPushButton('Generate Cinema', self)
        self.cinema_button.setToolTip(
            'Play back the loaded temperature sequence. CEM43 is shown '
            'afterwards only when Apply Motion Correction is checked.'
        )
        sequence_layout.addWidget(self.cinema_button)

        self.thermostat_button = QPushButton('Generate Thermostat', self)
        self.thermostat_button.setToolTip(
            'ROI temperature–time curve for the current ROI across the '
            'Cinema sequence. Draw the ROI on the main window first.'
        )
        sequence_layout.addWidget(self.thermostat_button)

        sequence_group.setLayout(sequence_layout)
        left_layout.addWidget(sequence_group)
        
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

        # Add stretch to push everything up
        left_layout.addStretch()

        # Do not pre-set a minimum size on the inner panel. QScrollArea with
        # widgetResizable shrinks the child down to its minimumSize; the floor
        # is applied after the layout is filled (see _freeze_left_panel_minimum).
        self.left_panel = left_panel
        left_scroll = QScrollArea()
        left_scroll.setObjectName("left_panel_scroll")
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setWidget(left_panel)
        # Placeholder until _freeze_left_panel_minimum sizes the column to
        # the widest row (the paired DICOM buttons). 340 is not wide enough
        # for those rows, and a narrower viewport clips them.
        left_scroll.setMinimumWidth(340)
        self.left_panel_scroll = left_scroll
        
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
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setSizes([340, 860])  # Set initial sizes
        main_vertical_layout.addWidget(splitter)
        
        # 添加作者信息
        author_label = QLabel("Authored by Xiaolei Zhu, zxl1412@gmail.com")
        author_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        author_label.setStyleSheet("color: #666; font-size: 9pt;")  # 设置样式
        main_vertical_layout.addWidget(author_label)
        
        # 将垂直布局添加到主布局
        main_layout.addLayout(main_vertical_layout)
        self._freeze_left_panel_minimum()

    def _freeze_left_panel_minimum(self):
        """Floor the inner column so a short viewport scrolls instead of flattening rows.

        The scroll area is widened to the layout's minimum width plus the
        vertical scrollbar, so a row of buttons is fully visible instead of
        sitting under a horizontal scrollbar. Growing wider is still allowed.
        Do not call this before the left layout is filled.
        """
        lay = self.left_panel.layout()
        if lay is None:
            return
        hint = lay.totalMinimumSize()
        self.left_panel.setMinimumHeight(hint.height())
        self.left_panel.setMinimumWidth(max(hint.width(), 320))

        scroll = self.left_panel_scroll
        bar_w = scroll.verticalScrollBar().sizeHint().width()
        if bar_w <= 0:
            bar_w = 16
        # A couple of pixels of slack stops an off-by-one from bringing the
        # horizontal bar back and shrinking the viewport again.
        needed = max(hint.width(), 320) + bar_w + 2 * scroll.frameWidth() + 4
        scroll.setMinimumWidth(needed)
        splitter = scroll.parentWidget()
        if isinstance(splitter, QSplitter):
            sizes = splitter.sizes()
            if len(sizes) >= 2 and sizes[0] < needed:
                extra = needed - sizes[0]
                sizes[0] = needed
                sizes[1] = max(0, sizes[1] - extra)
                splitter.setSizes(sizes)

    def showEvent(self, event):
        super().showEvent(event)
        self._freeze_left_panel_minimum()
        
    def setup_connections(self):
        """Set up internal signal connections."""
        self.mouse_move_signal.connect(self.update_cursor_info)
        # Connect Cinema and Thermostat buttons
        self.cinema_button.clicked.connect(self.generate_cinema_signal.emit)    # 新增: 生成温度图电影按钮
        self.thermostat_button.clicked.connect(self.generate_thermostat_signal.emit)    # 新增: 生成温度-时间曲线按钮
        
    # UI Event Handlers
    def on_load_baseline(self):
        """Import the reference series (old first frame)."""
        self._choose_sequence_import("baseline")

    def on_load_sequence(self):
        """Import the series to compare or play."""
        self._choose_sequence_import("sequence")

    def on_load_b0_map(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select B0 Map DICOM",
            "",
            "DICOM (*.dcm *);;All files (*)",
        )
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

    def _on_registration_site_changed(self, text):
        """Enable the coarse liver checkbox only for the abdomen site.

        Leaving abdomen remembers the check.  Coming back restores it,
        including a switch Head -> Joint -> Abdomen.
        """
        site = registration_site_from_label(text)
        checked, enabled, remembered = coarse_liver_checkbox_transition(
            site,
            self.coarse_liver_checkbox.isChecked(),
            self._coarse_liver_for_abdomen,
        )
        self._coarse_liver_for_abdomen = remembered
        self.coarse_liver_checkbox.setEnabled(enabled)
        self.coarse_liver_checkbox.setChecked(checked)
        
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

    def set_frame_counts(self, real_count, imag_count, b0_count=0):
        """Keep real/imag frame counts in the status bar after a sequence load.

        Counts stay visible after the transient status message times out.
        A kind with no frames is named in ``missing``, so Cinema can be
        started only after the user can see which half of the pair is absent.
        B0 is optional and is listed only when a map was found.
        """
        real_count = int(real_count)
        imag_count = int(imag_count)
        b0_count = int(b0_count)
        parts = [f"{real_count} real", f"{imag_count} imag"]
        if b0_count > 0:
            parts.append(f"{b0_count} B0")
        missing = []
        if real_count <= 0:
            missing.append("real")
        if imag_count <= 0:
            missing.append("imag")
        text = ", ".join(parts)
        if missing:
            text += " (missing: " + ", ".join(missing) + ")"
        self.frame_count_label.setText(f"Sequence: {text}")
        next_label = getattr(self, "sequence_next_label", None)
        if next_label is None:
            return
        if missing:
            named = " and ".join(missing)
            next_label.setText(
                f"Sequence loaded, but {named} frames are missing. "
                "Generate Cinema needs both real and imaginary frames."
            )
        elif real_count < 2 or imag_count < 2:
            next_label.setText(
                "Sequence loaded. Next: Load Sequence with at least "
                "2 real and 2 imaginary frames, then Generate Cinema."
            )
        else:
            next_label.setText(
                "Next: Generate Cinema. CEM43 is shown only if Apply Motion "
                "Correction is checked; otherwise that window explains why "
                "and does not show a dose. Then draw an ROI and Generate "
                "Thermostat for the temperature curve."
            )
        
    def update_status(self, message, timeout_ms=5000):
        """Update the status bar with a message.

        ``timeout_ms`` is how long the message stays visible.  ``0`` keeps it
        until the next status update, which is used for a registration warning
        that must not disappear behind "Temperature calculation complete".
        """
        self.statusBar.showMessage(message, timeout_ms)
        
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
        load_baseline_action = QAction('Load Baseline Sequence', self)
        load_baseline_action.triggered.connect(self.on_load_baseline)
        load_sequence_action = QAction('Load Sequence', self)
        load_sequence_action.triggered.connect(self.on_load_sequence)
        load_b0_action = QAction('Load B0 Map DICOM', self)
        load_b0_action.triggered.connect(self.on_load_b0_map)

        load_menu.addAction(load_baseline_action)
        load_menu.addAction(load_sequence_action)
        load_menu.addAction(load_b0_action)

        # 创建独立的Batch菜单
        batch_menu = menubar.addMenu('Batch')
        batch_baseline_action = QAction('Load Baseline Sequence', self)
        batch_baseline_action.setToolTip(
            'Same as Load Baseline Sequence on the main window.'
        )
        batch_baseline_action.triggered.connect(self.on_load_baseline)
        batch_menu.addAction(batch_baseline_action)
        load_folders_action = QAction('Load Sequence', self)
        load_folders_action.setToolTip(
            'Same as Load Sequence on the main window.'
        )
        load_folders_action.triggered.connect(self.on_load_sequence)
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
        # Drop any remembered abdomen check before the combo signal runs.
        self._coarse_liver_for_abdomen = None
        self.coarse_liver_checkbox.setEnabled(True)
        self.coarse_liver_checkbox.setChecked(False)
        self.registration_site_combo.blockSignals(True)
        self.registration_site_combo.setCurrentIndex(0)  # Abdomen
        self.registration_site_combo.blockSignals(False)
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
        
        # Clear cursor info.  The sequence frame counts stay: Reset All
        # clears the two-frame images, not the loaded sequence paths.
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
        """Sequence import.  Kept so older callers still open the same dialog."""
        self._choose_sequence_import("sequence")

    def _choose_sequence_import(self, role):
        """Pick files or folders for the baseline or the sequence."""
        if getattr(self, "sequence_busy", False):
            self.show_error(
                "A temperature sequence is already running. "
                "Wait for it to finish or cancel it before loading another sequence."
            )
            return
        title = (
            "Select baseline sequence"
            if role == "baseline"
            else "Select sequence"
        )
        dialog = FolderSelectionDialog(self, title)
        if dialog.exec_() == QDialog.Accepted:
            selected_folders = dialog.get_selected_folders()
            if selected_folders:
                self.import_role = role
                self.import_component_pair = dialog.component_pair()
                self.batch_load_dicom_signal.emit(selected_folders)
                self.update_status(
                    f"Scanning {len(selected_folders)} path(s) for DICOM and NIfTI..."
                )
            else:
                self.show_error("No folders or files selected")

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
        if not self._sequence_ready_for_cinema():
            return
            
        # 发送信号到控制器
        self.generate_cinema_signal.emit()

    def _sequence_ready_for_cinema(self):
        """Report a missing sequence, naming the kind that was not loaded."""
        real_files = getattr(self, "all_real_files", None) or []
        imag_files = getattr(self, "all_imag_files", None) or []
        if not real_files or not imag_files:
            missing = []
            if not real_files:
                missing.append("real")
            if not imag_files:
                missing.append("imag")
            self.show_error(
                "No sequence data available. Please use 'Load Sequence' first. "
                "Missing: " + ", ".join(missing) + "."
            )
            return False
        if len(real_files) < 2 or len(imag_files) < 2:
            self.show_error(
                "Need at least 2 real and 2 imaginary files for Cinema feature. "
                f"Have {len(real_files)} real and {len(imag_files)} imaginary."
            )
            return False
        return True
        
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
    """Pick folders and files for one baseline or sequence import."""
    def __init__(self, parent=None, title="Select sequence"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        
        layout = QVBoxLayout()
        
        # 添加说明标签
        instructions_layout = QVBoxLayout()
        label1 = QLabel("1. Click 'Add Folder' or 'Add Files' (.dcm, extensionless DICOM, .nii, .nii.gz)")
        label2 = QLabel("2. Hold Ctrl key to select multiple paths in the list")
        label3 = QLabel("3. Click 'OK' to import the selected paths")
        label4 = QLabel("Folders are scanned recursively. A missing .dcm suffix is still read.")
        
        # 使说明文本更加突出
        for label in [label1, label2, label3, label4]:
            label.setStyleSheet("font-weight: bold; color: #0066cc;")
        
        instructions_layout.addWidget(label1)
        instructions_layout.addWidget(label2)
        instructions_layout.addWidget(label3)
        instructions_layout.addWidget(label4)
        layout.addLayout(instructions_layout)

        self.component_combo = QComboBox(self)
        self.component_combo.addItem("Auto (DICOM tags or filename)", None)
        self.component_combo.addItem("Real + imaginary", "real_imag")
        self.component_combo.addItem("Magnitude + phase", "magnitude_phase")
        self.component_combo.setToolTip(
            "Used when NIfTI names do not say real/imag or magnitude/phase. "
            "Two unnamed volumes are sorted by filename: first is real or "
            "magnitude, second is imaginary or phase."
        )
        layout.addWidget(self.component_combo)
        
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

        add_files_button = QPushButton("Add Files")
        add_files_button.clicked.connect(self.add_files)
        buttons_layout.addWidget(add_files_button)
        
        # 添加删除文件夹按钮
        remove_button = QPushButton("Remove")
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
        self.status_label = QLabel("Ready to select folders or files")
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

    def add_files(self):
        """Add DICOM or NIfTI files.  DICOM does not need a .dcm suffix."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select DICOM or NIfTI files",
            "",
            "Images (*.dcm *.nii *.nii.gz *);;All files (*)",
        )
        items = [self.folder_list.item(i).text() for i in range(self.folder_list.count())]
        added = 0
        for path in files:
            if path not in items:
                self.folder_list.addItem(path)
                self.folder_list.item(self.folder_list.count() - 1).setSelected(True)
                items.append(path)
                added += 1
        if added:
            self.status_label.setText(f"Added {added} file(s)")
    
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
            self.status_label.setText("Please select at least one folder or file")
            return
            
        self.accept()

    def component_pair(self):
        """NIfTI pair choice, or None to follow tags and filenames."""
        return self.component_combo.currentData()
    
    def get_selected_folders(self):
        """获取所有选择的文件夹和文件"""
        return [self.folder_list.item(i).text() for i in range(self.folder_list.count()) 
                if self.folder_list.item(i).isSelected()]
