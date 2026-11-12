# 可选 INR 容积校正（设计说明）

本文只描述新增的 opt-in 路径。默认运动校正仍然是 `mvc/models/registration.py` 里的 SimpleITK B-spline，参数和输出都没有改。

## 1. 不改什么

- UI 勾选 “Apply Motion Correction” 仍然调用 `SimpleITKRegistrar`。
- `RegistrationConfig` 的默认值不变：Mattes MI、50 bins、RANDOM 10%、无种子、学习率 1、每层最多 100 次、收敛窗口 10、金字塔 `(4, 2, 1)` / `(2, 1, 0)`、B-spline mesh 8。
- 温度计算不调用 INR，也不调用低秩加稀疏分解。
- 没有新的第三方依赖。实现只用已经在 `requirements.txt` 里的 numpy 和 scipy。

因此这条路径不会改变现有默认结果。要使用它，必须在构造控制器时显式换入 registrar，并同时换入 INR 自己的配置：

```python
from mvc.models.inr_registration import INRRegistrationConfig, INRVolumeRegistrar

controller = ThermometryController(
    model,
    view,
    registrar=INRVolumeRegistrar(),
    registration_config=INRRegistrationConfig(),
)
```

`INRVolumeRegistrar.register_complex` 拒绝 `RegistrationConfig`。两套参数不能混用。不换控制器时，也可以直接调用 `register_volume` 或 `register_complex`。

## 2. 单病例模型

没有训练集。一个病例的各期相一次性拟合。

坐标 MLP 固定为 4 个隐层、每层 64 个单元、`tanh` 激活，输出 `R` 个标量势，`R` 只能是 2 或 3。空间基函数不是网络的输出值，而是这些势对归一化坐标的解析 Jacobian `J(x)`，形状 `(R, D)`。反向传播也是解析的：链式法则带着 `1 - tanh(pre)^2`，不用有限差分近似网络。

第 `t` 期的位移是

```text
u(x, t) = s * tanh(J(x) c_t + a_t)
```

- `c_t` 长度 `R`，乘在解析 Jacobian 上，表示局部空间模式。
- `a_t` 长度 `D`，是这一期的整体平移。小权重 `tanh` 网络的 Jacobian 均值接近 0，单靠 `J c` 表达不了整体呼吸位移，所以平移单独保留。
- `s` 是 `displacement_scale`，单位是体素。`tanh` 把位移限制在 `±s` 内。
- 参考期的 `c` 和 `a` 固定为 0，所以参考期不被改写。

`R = 2` 或 `3` 是低秩呼吸模型：一个方向可以跟呼吸深度，其余方向可以跟往返不一致的迟滞。它不是群体主成分，只在这一个病例的时间系数里。

## 3. 目标函数

每一轮只最小化三项：

```text
projection_weight * 定位像投影一致性
+ mind_weight * MIND 跨期相距离
+ temporal_smooth_weight * 相邻期系数差的平方
```

定位像项比较的是同一归一化坐标上、按射线长度归一化后的采样强度。直接比较正交方向的原始投影和，对纯平移不敏感：平移后的体素仍落在同一条射线上，投影轮廓几乎不变。归一化采样保留了“沿射线组织移动了”的信息，恒等位移的损失仍然是 0。

MIND 是半径 1 的邻域自相似描述子（2D 为 8 邻域，3D 为 26 邻域），再做 softmax。全局强度平移不改变它。描述子的空间导数预先算好，优化时只在当前位移处采样。

没有使用互信息、NCC、光度 MSE 或任何训练好的正则网络作为数据项。时间平滑只惩罚系数和整体平移在期相之间的跳变。

优化是带动量的梯度步，每个参数组先除以自己的 RMS，所以同一学习率可以用于平移、低秩系数和 MLP 权重。损失不下降到数值容差时，按最大迭代次数停止。

## 4. 和温度计算的边界

`register_complex` 返回现有的 `RegistrationResult`，因此控制器原有的“失败则继续用原图”逻辑可以接住它。实部和虚部使用同一个位移；两者都存在时，位移在幅值上估计。

INR 结果不会写回 `real2/imag2`。只有调用方把它放进 `model.registration_result` 并且温度计算传入 `use_registration=True` 时才会被用到。默认控制器不会这么做。

## 5. 低秩加稀疏分解

`low_rank_sparse_decomposition(series, rank, sparsity)` 是校正之后的独立分析，不是配准的一部分。

输入形状 `(T, *空间)`。它交替做两件事：对残差做秩 `rank` 的 SVD 截断，得到背景 `L`；再把剩余项里最大的 `sparsity` 比例保留为稀疏变化 `S`，其余置零。这是 robust PCA / principal component pursuit 的实用形式。被阈值丢掉的小残差既不进 `L` 也不进 `S`，所以 `L + S` 不等于原序列。

返回值是 `LowRankSparseResult`。温度计算、`SimpleITKRegistrar` 和 `INRVolumeRegistrar` 都不调用它。

## 6. 数值检查

合成测试覆盖：

- 解析空间 Jacobian 与有限差分一致。
- Jacobian 对每一层权重和偏置的解析梯度与有限差分一致。
- 2D、3D、`R=2`、`R=3` 的输出形状。
- 位移、系数和重采样结果为有限值。
- 恒等输入保持零位移。
- 已知平移使目标下降且 NCC 上升。
- 同一随机种子可复现。
- 输入数组不被修改；实部和虚部共用一个位移。
- 非有限输入和优化器异常有明确失败结果，不返回半截图像。
- MIND 每个体素是概率分布，并且对全局强度平移不变。
- RPCA 把注入的孤立尖峰留在 `S` 里。

这些测试不读取 DICOM，也不使用 GPU。

## 7. 尚未改默认值的质量项

PR #3 之后的验证没有改 SimpleITK 的默认参数，也没有改它的通过/警告规则。原因是下面三项还没有真实数据可以定案：

1. `RegistrationConfig()` 仍是单阶段 B-spline 和 Mattes MI，梯度下降也不变。界面勾选框不再用它，而用 `liver_respiratory()`：先 rigid（不用 affine，避免无肝脏 mask 时剪切把肝脏拉斜），再接网格间距约 40 mm 的 B-spline。SimpleITK 没有弯曲能量度量，粗网格就是正则。相似度是半径 4 体素的 ANTS 邻域相关。Mask 是 Otsu 再膨胀 12 mm，只去掉空气并留一圈组织边缘，不是肝脏分割，亮的胸腹壁仍会留在 mask 里。另有一个默认关闭的粗肝脏 ROI（`mask_kind="coarse_liver"`，或界面勾选 “Use coarse liver ROI”）。它不改变这条默认 mask，INR 也不使用它。`transform_kind="rigid"` 或 `"affine"` 仍可单用。
2. 相似度仍默认是 Mattes MI、10% 随机采样、不设种子。换成 correlation 或全采样会改变默认温度图，不能只凭一个合成 phantom 替换。
3. 单病例路径：配准失败后仍按未校正数据继续算温度，并给出错误提示。改成失败即中止是产品行为，不是这次能自行切换的。Cinema 序列不适用这条：某一帧加载失败或配准失败时，该帧记为 `None`，不把上一帧的结果算进来，其余帧继续。

已经加上、且不参与是否接受配准的诊断：校正前后 NCC、MAE，位移 RMS/最大值，以及 B-spline 的折叠比例。DICE 仍是唯一的警告阈值。

Cinema 序列默认仍不做运动校正。勾选运动校正后，每一帧对第一帧配准，而不是对相邻帧配准。单次运动校正和 Cinema 序列都在界面线程之外运行；取消不会改已经完成的那一帧，也没有改默认配准参数。

## 8. 已知限制

- 空间基是梯度场加一个整体平移，适合平滑的呼吸状运动，不适合大角度旋转。
- 采样定位像一致性不是完整的投影算子。它避免了纯平移时投影和的盲区，但是采样数远小于体素数时，细结构仍然欠约束。
- 默认迭代和采样是为 CPU 上的合成用例选的，不是临床配准参数。
- 没有真实 DICOM 回归。默认 B-spline 的质量问题（恒等输入不稳定、已知平移恢复差）这次没有改，仍按原计划留在单独的算法评审里。
