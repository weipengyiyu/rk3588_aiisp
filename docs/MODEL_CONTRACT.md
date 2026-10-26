# 模型接口合同

本页描述默认 `rknn` FP32 适配器；新增 `rknn-native` 的原生 FP16、矩形张量和边缘 post 合同见 [快速路径](FAST_PATH.md)。

## 输入输出

当前 RKNN 适配器只接收一个输入、一个输出，二者形状为 `[1,4,T,T]`（NCHW）或 `[1,T,T,4]`（NHWC），元素类型 FP16/FP32，元素数和 T 必须相同。框架传入 NHWC float32，由 Runtime 完成必要的输入转换；适配器将输出转换成 NHWC float32。原生布局、INT8/UINT8、多输入输出、状态张量和 RGB/YUV 模型需要扩展 `InferenceBackend` 或现有 RKNN 适配器。

四通道固定为 `R, Gr, Gb, B`。Gr 是红像素行的绿，Gb 是蓝像素行的绿。pre 根据当前 Bayer 顺序映射通道，post 做逆映射。Bayer 顺序按实际裁剪后的图像确定；奇数偏移裁剪可能改变顺序。

pre 计算 `clamp((raw - black_level)/(white_level - black_level), 0, 1)`；post 计算 `round(clamp(y,0,1)*(white_level-black_level)+black_level)`。默认 black=0、white=4095。不要再次在模型转换配置中叠加相同的 mean/std 归一化。输出语义须为恢复后的绝对 RAW 值；若网络输出噪声残差，须在后端或 post 中明确实施残差合成。

## 分块

T 为 Bayer 平面尺寸，即原图每个方向的一半坐标域。halo 也以 Bayer 平面像素为单位，有效步长 `T - 2*halo`，必须大于零。对原图边缘复制填充，仅保留每块有效内部区域；不同块写入不重叠。

默认 T=512、halo=1 对应固定 3×3 示例。在 1936×1552 RAW 上每帧 4 块。更换多层网络时必须按实际感受野调整 halo，并进行整图/分块一致性验证。带全局统计、注意力、下采样相位约束等网络，不能只根据一个 halo 数值就认定分块等价。

运行时检查形状、输出大小、有效输出是否 NaN/Inf。模型失败会终止处理，不将 CPU 或旧帧当成 NPU 结果。

## 替换模型与扩展接口

符合上述合同的模型可用 `--model` 替换，同时提供正确的 `--tile`、`--halo`、`--black-level`、`--white-level`。模型必须针对 RK3588 转换，并与部署使用的 Runtime/驱动实际验证兼容。

不符合合同的模型：实现 `InferenceBackend::run(input, output)` 与 `calls()`，将其注入 `RawPipeline`。run 是同步接口，返回时输出必须可由 CPU 读取。若需要 ISO、曝光、噪声参数或前帧状态，扩展帧接口和模型输入；当前 `FrameMetadata` 中曝光增益字段只保留传输能力，不构成已完成的 3A 或时域算法。

`--strength` 只调整 CPU 固定高斯参考。NPU 示例强度固化于模型，修改时重新运行 `models/build_baseline.py --strength 数值`；运行 NPU 时不要用 CLI 的 `--strength` 冒充修改权重。

## 内存与时序

当前 V4L2 使用 MMAP，RAW 与张量之间有明确的数据转换。只有在 pre、NPU、post 均完成后才提交 ISP；等待 RAW 回读缓冲释放和匹配的 NV12 帧后再开始下一帧。不声称 DMA-BUF 零复制或并行流水。

统计中的 `mean_backend_ms` 包括 RKNN 输入提交、执行、输出获取及布局转换，不能解释为纯 NPU 内核耗时。`mean_pre_ms`、`mean_post_ms` 分别为分块准备和恢复；解包/打包单列，`mean_isp_ms` 包括提交、触发、等待回读和 YUV 完成。离线执行无 ISP，因此该项为 0，YUV 格式记为 null。
