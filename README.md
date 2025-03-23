# MR热统计系统 (MR Thermostat System)

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

## 许可证

本项目采用[MIT许可证](LICENSE)。
