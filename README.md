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
   - 序列完成后按 Sapareto & Dewey 1984 做体素级累积：CEM43 = Σ Δtᵢ · R^(43−Tᵢ)
   - T < 43°C 时 R = 0.25，T ≥ 43°C 时 R = 0.5；Δt 来自各期相 DICOM 的 AcquisitionTime
   - AcquisitionTime 缺失或间隔无效时停止并报告，不假定等间隔
   - 显示 CEM43 图、ROI 温度曲线和累积剂量
   - 39°C、CEM43 = 2 和 CEM43 = 9 只是可选参考线，来源为 van Rhoon 等，Eur Radiol 2013 的 MR 射频安全建议，不是治疗完成
   - 低信噪比或无效温度的体素不累积

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

勾选运动校正时，默认改为刚体对齐后再做约 40 mm 网格的 B-spline，相似度是半径 4 体素的局部相关，mask 是 Otsu 组织区再膨胀 12 mm。这会改变勾选后的温度图。`RegistrationConfig()` 仍是原来的单阶段 B-spline 和 Mattes MI，需要旧结果时显式传入。没有肝脏分割模型；Otsu 不能把胸腹壁从肝脏里分开。校正结果额外记录 NCC、MAE、位移和折叠比例，但这些数值不改变是否接受这次校正。界面上的运动校正在后台线程运行，取消会丢弃尚未应用的结果，不改变已经完成的配准。Cinema 只有在勾选运动校正时才配准，并且每一帧都以第一帧为参考；序列也在后台计算，取消只停止下一帧，已经算出的帧保留。另有一个必须显式启用的单病例 INR 容积校正，以及独立的低秩加稀疏分析；说明见 [docs/inr_volume_correction.md](docs/inr_volume_correction.md)。

Cinema 序列完成后会另外计算 CEM43。计算在 `mvc/models/thermal_dose.py`，不调用配准，也不读取 `RegistrationConfig`。它直接使用已经算出的温度图：未勾选运动校正时，每一帧仍是相邻两帧的温差再加基线，不是相对第一帧的温度。Δt 只来自 DICOM Acquisition Time (0008,0032)；没有这个标签就停止，不会改用 SeriesTime，也不会换成等间隔。低信号体素沿用温度计算里已有的 10% 幅值掩膜，温度图本身不改。39°C、CEM43 = 2 和 CEM43 = 9 默认不画，勾选后也只标注为 van Rhoon 等，Eur Radiol 2013 的 MR 射频安全参考，不能当成治疗完成。

推送和拉取请求会运行 `.github/workflows/tests.yml`：安装依赖、`compileall` 和 `pytest`。它不启动图形界面。

## 许可证

本项目采用[MIT许可证](LICENSE)。
引用本项目时请保留版权信息。
