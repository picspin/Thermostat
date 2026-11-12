# MR热统计系统 (MR Thermostat System)  
> Author: Xiaolei Zhu, zxl1412@gmail.com  

MR热统计系统是一个用于磁共振成像（MRI）热图分析的Python应用程序，可以从DICOM图像计算组织温度变化。

## 主要功能

1. **热图计算与显示**
   - 通过相位差重构（PRF）方法计算温度变化
   - 可调整窗宽窗位的温度图像显示
   - 移除坐标轴，适合医学应用的清晰显示

2. **ROI分析**
   - 手动绘制感兴趣区域（ROI）
   - 计算ROI内的温度统计数据（平均值、标准差、最大/最小值）
   - 保存和加载ROI掩膜，便于后续分析

3. **DICOM文件处理**
   - 批量加载DICOM文件功能
   - 自动识别实部、虚部图像和采集时间
   - 支持B0场图校正
   - 可选的运动校正功能

4. **结果导出**
   - 导出温度图为图像格式
   - 导出温度数据为NumPy或CSV格式
   - 导出ROI统计信息

5. **CEM43 热剂量（Cinema 序列）**
   - 序列完成后按 Sapareto & Dewey 1984 做体素级累积：CEM43 = Σ Δtᵢ · R^(43−Tᵢ)。整段 Δt 用该间隔结束时的温度，不重建间隔内的升温过程
   - T < 43°C 时 R = 0.25，T ≥ 43°C 时 R = 0.5；Δt 来自各期相 DICOM 的 AcquisitionTime
   - AcquisitionTime 缺失或间隔无效时停止并报告，不假定等间隔
   - 只有勾选运动校正（每帧相对第一帧的绝对温度）才显示 CEM43 图、ROI 温度曲线和累积剂量。未勾选时每帧是相邻温差加基线，界面只说明原因，不把该数当成 CEM43
   - 39°C、CEM43 = 2 和 CEM43 = 9 只是可选参考线，来源为 van Rhoon 等，Eur Radiol 2013 的 MR 射频安全建议，不是治疗完成
   - 低信噪比或无效温度的体素不累积。若掩膜排除了超过 95% 的有温度体素，摘要会写出比例，并说明单个亮像素可能抬高 10% 幅值阈值

## 系统要求

- Python 3.7+
- 操作系统：Windows, macOS 或 Linux

## 安装步骤

1. **创建并激活Python虚拟环境**（推荐）

   ```bash
   # 创建虚拟环境
   python -m venv venv
   
   # Windows激活虚拟环境
   venv\Scripts\activate
   
   # macOS/Linux激活虚拟环境
   source venv/bin/activate
   ```

2. **安装依赖项**

   ```bash
   pip install -r requirements.txt
   ```

3. **运行应用程序**

   ```bash
   python run.py
   ```

## 使用说明

### 批量加载DICOM文件
1. 点击"Batch Load DICOM Files"按钮
2. 选择实部、虚部和B0图像文件（可多选）
3. 程序会自动识别并加载文件

### 计算温度图
1. 设置基准温度（通常为37°C）
2. 选择处理选项（高斯滤波、B0校正、运动校正）
3. 点击"Calculate Temperature Change"按钮

### ROI分析
1. 右键并拖动以绘制ROI
2. ROI统计数据将显示在左侧面板
3. 可使用"Save ROI Mask"保存ROI
4. 可使用"Load ROI Mask"加载之前保存的ROI

### 窗宽窗位调整
1. 使用"Window Center"控制显示的中心温度
2. 使用"Window Width"控制显示的温度范围
3. 点击"Auto Window"自动优化窗宽窗位

## 开发与测试

运行单元测试：

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

测试不需要显示服务器；配准模块的单元测试不依赖 Qt，模型状态测试只使用 QObject。

勾选运动校正时，默认改为刚体对齐后再做约 40 mm 网格的 B-spline，相似度是半径 4 体素的局部相关，mask 是 Otsu 组织区再膨胀 12 mm。这会改变勾选后的温度图。`RegistrationConfig()` 仍是原来的单阶段 B-spline 和 Mattes MI，需要旧结果时显式传入。MIND 只有在配置里显式写 `metric="mind"` 时才启用，勾选框不会改用它，INR 也不变。它使用半径 1 的邻域自相似（2D 为 8 邻域，3D 为 26 邻域），采样步长默认 2 个体素：半径 1 已经覆盖相邻体素，步长 2 在每个内部邻域里留一个样本，又避免 Mattes 那种 10% 随机采样漏掉小结构。采样格在高索引一侧会空出最后一行/列；步长大于 `2 * radius + 1` 时，有的邻域里会一个样本都没有。两者都可以在配置里改，不改时就是这两个默认值。全局强度平移不改变这个描述子。归一化除的是整幅图的均值，不是逐体素邻域均值，这样和 INR 描述子保持同一个公式。moving 图像先变到固定网格再算描述子，邻域方向始终在固定图像上，旋转不会把通道对错。梯度是这次线性重采样的斜率，不生成整幅描述子梯度体。刚体和仿射的平移预热与完整求解共用 `iterations`，上报的迭代数不超过这个上限。mask 只按初始位姿取一次，初始偏差很大时可能直接失败。没有肝脏分割模型；Otsu 不能把胸腹壁从肝脏里分开。校正结果额外记录 NCC、MAE、位移和折叠比例，但这些数值不改变是否接受这次校正。界面上的运动校正在后台线程运行，取消会丢弃尚未应用的结果，不改变已经完成的配准。Cinema 只有在勾选运动校正时才配准，并且每一帧都以第一帧为参考；序列也在后台计算，取消只停止下一帧，已经算出的帧保留。另有一个必须显式启用的单病例 INR 容积校正，以及独立的低秩加稀疏分析；说明见 [docs/inr_volume_correction.md](docs/inr_volume_correction.md)。

Cinema 序列完成后会另外计算 CEM43。计算在 `mvc/models/thermal_dose.py`，不调用配准，也不读取 `RegistrationConfig`。它直接使用已经算出的温度图，不改配准、不改温度值。未勾选运动校正时，每一帧仍是相邻两帧的温差再加基线，不是相对第一帧的温度；这时不显示剂量数字，只报告原因，因为 CEM43 对温度是指数敏感，用温差去累加会把剂量低估几个数量级。勾选运动校正后，每一帧都相对第一帧，才显示 CEM43 图。剂量窗口和 Cinema 播放器都是非模态的，播放时仍可在主界面画 ROI 并刷新曲线。Δt 只来自 DICOM Acquisition Time (0008,0032)；没有这个标签就停止，不会改用 SeriesTime，也不会换成等间隔。低信号体素沿用温度计算里已有的 10% 幅值掩膜，温度图本身不改。掩膜把超过 95% 的有温度体素排除时，摘要会说明比例，并指出单个亮像素可能抬高阈值。ROI 曲线只统计每一帧都有效的体素。39°C、CEM43 = 2 和 CEM43 = 9 默认不画，勾选后也只标注为 van Rhoon 等，Eur Radiol 2013 的 MR 射频安全参考，不能当成治疗完成。

推送和拉取请求会运行 `.github/workflows/tests.yml`：安装依赖、`compileall` 和 `pytest`。它不启动图形界面。

## 许可证

本项目采用[MIT许可证](LICENSE)。
引用本项目时请保留版权信息。
