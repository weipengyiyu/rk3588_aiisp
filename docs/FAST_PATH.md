# IMX623 原生 FP16 快速路径

本路径沿用 `pre → NPU → post → ISP`。pre 将 RAW12 映射到 R/Gr/Gb/B 四平面并准备 FP16 张量；NPU 计算独立 Bayer 平面的高斯局部估计；post 融合边缘保护、恢复 Bayer RAW，再送入 ISP。

它是固定滤波工程基线，不是经过 IMX623 数据训练的神经网络。CPU 的边缘保护是明确的 post 处理，NPU 执行失败时不会回退为 CPU 推理。

## 构建与运行

将项目放在 SDK 的 `external/` 下面，按主 README 构建、打包。`models/fast-rect/` 已包含实测模型及 SHA256 清单。重新生成需要安装 RKNN Toolkit2 2.3.0、ONNX 1.16.1 及相应依赖：

```bash
python models/build_fast.py --algorithm gaussian --strength 1 --out models/fast-rect
bash build.sh
bash package.sh
```

板端包内运行：

```bash
sudo bash scripts/run_imx623_fast.sh --frames 300 --dump-last
cat run-fast/metrics.json run-fast/cleanup.json
```

该入口选择：`rknn-native`、976×784 张量、halo=1、边缘保护强度 0.75、阈值 96 RAW12 灰阶。1936×1552 输入每帧执行一次 NPU 推理。`--frames 0` 持续运行，Ctrl+C 停止并恢复配置。

`--performance` 临时限制进程到 CPU4–7，并将相应 CPU policy 设为 performance；正常退出、程序错误和受控信号退出时恢复 governor 与 affinity。此项不改变 NPU governor，不超出硬件已有频率范围。SIGKILL、断电或内核故障不能保证用户态清理。若不需要调速，用 `run_pipeline.py` 的下列显式命令并省略 `--performance`，延时将随调度与频率变化：

```bash
sudo python3 scripts/run_pipeline.py \
  --mode rknn-native --model models/fast-rect/bayer_gaussian.rknn \
  --tile 976 --tile-height 784 --halo 1 --post-filter edge \
  --edge-strength 0.75 --edge-threshold 96 --frames 300 --performance \
  --output-dir run-fast
```

## 模型与内存合同

- 使用 group=1 的对角权重，保持四 Bayer 平面独立，同时减少原分组卷积带来的布局填充开销。
- 输入输出使用预分配的可缓存 RKNN FP16 内存，pre/post 直接访问张量；支持检测通过的 NHWC 输入和 NC1HWC2 输出布局。
- 矩形张量减少无效填充行；NEON 完成前后处理；`process_into()` 复用输出缓冲。
- 这不等于相机到 ISP 全程 DMA-BUF 零复制，RAW 解包、打包和 ISP 回读仍有单独耗时。
- `--post-filter edge` 必须配合其纯高斯 strength=1 模型、halo=1，以及匹配的 JSON 几何/电平/模型哈希。启动器在改变媒体图前检查。
- `--edge-strength` 控制 post 融合强度；`--edge-threshold` 控制保护结构的阈值。它们不同于固定在模型里的高斯权重。

令原始 RAW 值为 x，高斯局部估计为 b，post 使用：

```text
delta = b - x
gate = max(0, 1 - abs(delta) / threshold)
output = round(x + strength * gate * delta)
```

默认参数在合成噪声 sigma=4/8/16 的平坦区域优于原 strength=0.5 高斯，并减少测试阶跃边缘的模糊。sigma=32 时，默认阈值的平坦区残余噪声比原高斯更高。这是边缘保护与强噪声平滑之间的取舍；尚未按真实 IMX623 的 ISO/增益/黑电平完成噪声标定，不应把这些结果推广到所有场景。

`models/fast/` 保留过往方形模型和将边缘门控放入 NPU 的实验产物；目前推荐 `models/fast-rect/` + post 融合方案。
