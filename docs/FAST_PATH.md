# IMX623 原生 FP16 快速路径

最新性能及 TIDL 参考分析见 [第二轮分析](LATENCY_ANALYSIS_20261026.md)。默认入口采用 16 通道 Bayer 打包、原生 FP16 I/O 和四 CPU 线程，1936×1552 全帧降噪平均 7.958 ms、P95 9.391 ms。链路仍为 RAW → pre → NPU → post → ISP → NV12。

## 生成与部署

```bash
python models/build_fast.py --algorithm gaussian --strength 1 --bayer-pack 2 --out models/fast-pack16
bash build.sh
bash package.sh
```

模型转换需要 RKNN Toolkit2 和兼容依赖，本次使用 2.3.0、ONNX 1.16.1。匹配的模型在 `models/fast-pack16/`。

板端包内：

```bash
sudo bash scripts/run_imx623_fast.sh --frames 300 --dump-last
cat run-fast/metrics.json run-fast/cleanup.json
```

入口选择 rknn-native、488×392 Tensor、`--bayer-pack 2`、halo=1、`--cpu-threads 4`、edge 强度 0.75、阈值 96。`--frames 0` 持续运行，Ctrl+C 退出并恢复。可用 `--cpu-threads 2` 或 `1` 降低 CPU 资源占用。

显式命令：

```bash
sudo python3 scripts/run_pipeline.py \
  --mode rknn-native --model models/fast-pack16/bayer_gaussian.rknn \
  --tile 488 --tile-height 392 --bayer-pack 2 --cpu-threads 4 --halo 1 \
  --post-filter edge --edge-strength 0.75 --edge-threshold 96 \
  --frames 300 --performance --output-dir run-fast
```

`--performance` 临时限制进程到 CPU4–7，修改对应 CPU governor 为 performance；正常、错误和受控信号退出时恢复。它不改变 NPU governor，不超出硬件已有频率范围。省略时保留系统调度和频率策略，延时会变化。SIGKILL、断电和内核故障不能保证用户态清理。

## 数据合同

- 2×2 Bayer 单元，即 4×4 RAW 像素，打包为 16 通道；先按单元位置 `(sy,sx)`，再按 R/Gr/Gb/B 排列，保持图像分辨率和 RAW12 精度。
- pre/post 使用查询到的 NHWC 或 NC1HWC2 FP16 原生布局，按实际 row、pixel 和 channel-block stride 寻址。
- bayer_pack=2 需要匹配的 16 通道原生模型。halo 以模型网格计，此时一格为 2×2 Bayer 单元。
- 卷积权重重映射仍实现每种 Bayer 颜色独立的 3×3 Gaussian；post 边缘保护公式不变。
- 缓冲复用、pre/post 共享持久 worker 池；阶段仍同步，不会把不同帧混成结果。
- 不是相机到 ISP 全程 DMA-BUF 零复制；RAW 解包、打包、ISP 等待仍单列。
- edge 模式检查模型 JSON 的几何、packing、电平和 SHA256。模型内 Gaussian strength 必须为 1。

边缘门控（x 为原始 RAW，b 为 Gaussian 估计）：

```text
delta = b - x
gate = max(0, 1 - abs(delta) / threshold)
output = round(x + strength * gate * delta)
```

默认阈值的合成 sigma=4/8/16 平坦区噪声和阶跃边缘优于旧 strength=0.5 Gaussian；sigma=32 平坦区残余噪声更高。它是固定滤波工程基线，不是经 IMX623 数据训练的网络，也未完成实际 ISO/增益噪声标定。

## 旧路径

`models/fast-rect/` 的 976×784 四通道模型保留。选 `--bayer-pack 1 --cpu-threads 1` 及匹配的 tile/manifest；不要将旧几何用于 16 通道模型。
