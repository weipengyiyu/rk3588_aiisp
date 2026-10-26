# 2026-10-26 IMX623 板端验证

第二轮最新优化与完整验收见 [性能优化详解](LATENCY_ANALYSIS_20261026.md)：平均降噪 7.958 ms、P95 9.391 ms；下文保留较早阶段的历史记录。

SDK 工程为 `external/rk3588-aiisp`，目标为 RK3588 / IMX623。Linux 5.10.226，2026-10-26 15:50:27 构建；RKNN Runtime 2.3.0，驱动 0.9.8。未使用旧 `imx623-aiisp` 的实测数据替代本工程验收。

## 原始框架通路

- 主机 CTest 5/5、ARM64 交叉编译、BSP ioctl ABI 检查通过。
- IMX623 I2C7 0x1a，1936×1552 RGGB RAW12、两通道、18 MHz XVCLK。
- 自动发现 CIF `/dev/video11`、ISP rawrd2_s `/dev/video27`、ISP subdev `/dev/v4l-subdev3` 和 NV12 `/dev/video22`。编号属于本次启动，应用仍使用拓扑发现。
- bypass 完成 30 帧，输入和输出 LE16 RAW 逐字节相同。
- 默认 rknn 完成 30 帧、120 次推理，pre 12.112 ms、backend 75.417 ms、post 15.459 ms，整帧降噪平均 106.208 ms。
- 相同输入 CPU 高斯参考最大误差 2 RAW12 灰阶。
- 原框架运行前后所有四个 media 设备的拓扑逐字节一致。修复了 pad 快照遗漏 field/colorspace/xfer/ycbcr/quantization 属性的问题。

## 快速路径

在上述通路验证之后增加 `rknn-native`、原生 FP16 I/O、矩形张量、NEON 前后处理、输出复用以及融合到 post 的边缘门控。测试使用 `--performance`，并验证 governor 能恢复。

连续完成 300 帧、300 次 NPU 推理，输出为全尺寸 NV12：

- pre 平均 3.734 ms；backend 平均 8.536 ms；post 平均 7.745 ms。
- 整帧降噪平均 **20.446 ms**，P50 20.431 ms，P95 20.489 ms，最大 23.391 ms。
- RAW 解包到 NV12 返回 P50 36.799 ms，P95 **36.865 ms**，不含等待下一帧 Sensor 输出。
- 首帧后实际完成速度 27.146 fps；不能宣称端到端 20 ms 或 30 fps 全帧处理。
- RAW 解包平均 0.829 ms、回读打包 3.958 ms、ISP 提交/等待平均 10.792 ms。
- 处理期间 drain 31 帧、Sensor 序号间隙 31；两者不可相加。
- 输出序号与提交序号逐帧匹配，进程报告成功、媒体配置 cleanup 成功。

原始统计见 [final-metrics.json](validation/final-metrics.json)。本次是短时功能与性能验证，未完成长时间稳定性验收。

## 数值与画质边界

实采 3,004,672 像素与独立整图浮点边缘门控参考比较：最大误差 2、平均误差 0.1197688 RAW12 灰阶；2,878,663 个像素发生变化。

合成阶跃和高斯噪声测试中，默认阈值 96 的平坦区 MSE：sigma=4 时由原高斯 6.686 降至 4.288；sigma=8 时由 26.568 降至 18.485；sigma=16 时由 106.256 降至 88.138。sigma=32 时由 417.112 升至 490.088，平坦区噪声改善弱于原高斯。各测试阶跃边缘的 MSE 均下降。

数值细节见 [quality.json](validation/quality.json)。这些是明确的合成测试与同输入数值验证，不代表真实场景有无噪声真值的画质认证。当前没有训练模型、自动 AE/AWB 或 IMX623 专用 ISP IQ；实际画面仍有偏绿、偏暗问题。
