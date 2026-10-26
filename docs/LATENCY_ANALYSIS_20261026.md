# IMX623 第二轮性能优化与 TIDL 参考分析

日期：2026-10-26。基线提交 `6a47d6c`；同一 RK3588、IMX623、Linux 5.10.226、RKNN Runtime 2.3.0 / driver 0.9.8。保持 1936×1552 RAW12 全分辨率、归一化、Gaussian 权重和边缘保护公式，没有使用 INT8、缩小图像或 CPU 推理回退。

## 最终结果

- 本轮复测旧版：pre 3.737 ms、RKNN backend 8.536 ms、post 7.739 ms；整帧降噪平均 20.457 ms，P95 20.509 ms。
- 新版 300 帧：pre **1.904 ms**、backend **2.681 ms**、post **2.886 ms**；整帧降噪平均 **7.958 ms**，P50 7.389 ms，P95 **9.391 ms**，最大 12.516 ms。
- backend 平均降低约 69%，整帧降噪平均降低约 61%。
- RAW 已可读到对应 NV12 返回：P50 **23.843 ms**、P95 **25.797 ms**，不含等待 Sensor 下一帧。不能宣称端到端 20 ms。
- 完成 300 帧 / 300 次 NPU 执行；首帧后 30.018 fps，drain=0，Sensor sequence gaps=0。

`mean_denoise_ms` 是整个 RawPipeline 的墙钟耗时，除三阶段计时外还包含 RAW12 值域验证等工作，因此略大于三阶段之和。backend 包含同步 rknn_run 与运行时缓存管理，不是单独证明的纯卷积 MAC 时间。

测量均使用 `--performance`：临时使用 CPU4–7 和对应 CPU performance governor，退出后恢复；NPU governor 未修改。四线程使用更多 CPU 资源，不能泛化为未控制频率或带其他负载的环境。此轮不是硬实时或长稳认证。

## pre/post 为何较重

每帧有 3,004,672 个 RAW 像素。CPU 要完成 Bayer 排列、uint16→FP32→FP16、归一化、halo 填充；post 还要 FP16→FP32、有限值检查、边缘保护、恢复和舍入。这些软件操作不会自动由 NPU 加速。

基线虽已使用 NEON，但先将 Bayer 单元拆成通道向量，再转回像素内存，存在不必要的转置。新实现保持完整 Bayer 单元的分组访问，用连续加载、zip/unzip 与连续写回，保留颜色映射和 NaN/Inf 检查。

同一旧代码、30 次重复的微基准：

- pre 普通堆内存 3.721 ms，RKNN 可缓存映射内存 3.674 ms。
- post 普通堆内存 7.763 ms，RKNN 映射内存 7.767 ms。
- 关闭 post 边缘保护后为 4.504 ms；边缘门控额外约 3.26 ms。
- 约 6.1 MB 输入 memcpy 到 RKNN 缓冲为 0.796 ms。

因此主要成本不能归结为“DMA 未缓存”或单次 memcpy。已观察到的主要成本是逐像素运算、布局重排和单线程执行。只优化 SIMD 访问后，旧模型 pre 约 2.59 ms、post 约 5.99 ms，同一输入输出逐字节一致。

## 模型布局优化

旧模型逻辑上是四通道，输入 FP16 6,121,472 字节，原生输出却为 `NC1HWC2 [1,2,784,976,8]`、**24,485,888 字节**。直接要求紧凑 NHWC 输出时，backend 从约 8.39 ms 反而升到 19.95 ms，说明运行时转换不能忽略。

新方案把相邻 2×2 个 Bayer 单元，即 RAW 的 4×4 像素，打包为 16 通道。张量从 `976×784×4` 变为 `488×392×16`，有效元素数与 FP16 精度不变；卷积权重映射到对应邻居和相同 Bayer 颜色，仍计算原来的独立 3×3 Gaussian，post 公式也不变。

新模型原生输入、输出都是 `NC1HWC2 [1,2,392,488,8]`，缓冲各 **6,121,472 字节**。早期原型单独执行约 2.53 ms，最终模型完整链路 backend 平均 2.681 ms。这证明整个编译后 RKNN 路径更快，没有将布局内核、缓存维护和纯 MAC 各自单独归因。

## CPU 行块并行

新增共享持久 RowExecutor，pre/post 各自按不重叠行块并行，避免逐帧创建线程。阶段仍同步按 pre→NPU→post 执行；worker 异常在等待所有任务结束后传回调用者。

同一新版可执行文件、每组 180 帧：

- 1 线程：pre 4.314 ms、post 7.544 ms，降噪平均 14.935 ms，P95 14.961 ms。
- 2 线程：pre 2.297 ms、post 3.975 ms，降噪平均 9.569 ms，P95 9.580 ms。
- 4 线程：pre 1.931 ms、post 3.089 ms，降噪平均 8.362 ms，P95 9.741 ms。

四线程平均延时最低，但有更多 CPU 开销和调度抖动；两线程可作为资源与延时的折中。最终 300 帧使用四线程，结果见开头。单线程入口保留。

这里的单线程包含新的 16 通道打包与通用跨块 stride 管理，不等于前面“仅改 SIMD”的四通道微基准。多线程降低的是墙钟耗时，并不表示计算总工作量也同比降低。

## TDA4 / TIDL 参考价值

只读检查了用户指定的 `/home/logic/work/tda4-vm-ai-isp/vision_apps/kernels/img_proc`：

- `c66/vx_oc_pre_proc_target.c` 约 255–310 行：按照实际 Tensor stride 分离 Bayer，明确维度顺序、字节/元素步长；值得借鉴其数据合同。这里是 CHW/float32，不是 RKNN 原生 FP16。
- `c66/vx_oc_post_proc_target.c` 约 282–341 行：与 pre 对称地还原 Bayer，尊重 Tensor 和输出图像的行跨度。
- `c66/vx_dl_pre_proc_target.c` 约 2030–2250 行：通用 DL pre 使用 DMA→计算→DMA、L2 ping-pong 与双缓冲，可参考其重叠搬运思路。但这不能证明自定义 OC RAW kernel 已具备相同 DMA 性能。

不能照搬 OpenVX/TIDL 维度、C66/L2/UDMA 接口和 float32 CHW 布局。RKNN 实测已经说明强制布局转换可能变慢。本轮使用 RKNN 原生布局与 CPU 行块并行，没有声称移植 TI UDMA 或实现全程 DMA-BUF 零复制。

参考 OC 代码还保留逐帧调试输出、整幅清零和 `&0x0fff` 截断。性能评测需确认日志级别；本项目保留夹紧、舍入、形状和有限值检查，不照搬这些细节。

## 验证与复现

- 同一实采 RAW：新版与旧版逐字节一致，1/4 线程也逐字节一致，共 3,004,672 像素。
- 随机满幅 RAW、边界及极值：关闭门控，实际 NPU Gaussian 输出对照整图整数加权参考，最大误差 2、平均误差 0.441942 RAW12 灰阶。
- CTest 5/5、ARM64 编译与板端测试通过，覆盖四 Bayer 顺序、部分分块、跨 channel-block stride、worker 异常/重用、NaN/Inf、元数据。
- 错误模型拒绝；SIGINT 返回 130；四个 media 拓扑与 CPU governor 恢复；相关视频节点无占用；本轮内核日志未匹配检查的 CSI/IOMMU/RKNPU/BUG/Oops 错误模式。

仍是固定 Gaussian 工程基线和既有边缘保护，没有因此完成 Sensor 噪声训练、IQ/AE/AWB 或生产画质认证。

SDK 主机生成匹配模型与构建：

```bash
/home/logic/.cache/rk3588-aiisp-toolkit-venv/bin/python models/build_fast.py --algorithm gaussian --strength 1 --bayer-pack 2 --out models/fast-pack16
bash build.sh
bash package.sh
```

本次模型环境：`/home/logic/.cache/rk3588-aiisp-toolkit-venv/bin/python`，Toolkit2 2.3.0，ONNX 1.16.1。

61 板：

```bash
cd /home/linux/rk3588-aiisp-latency-20261026
sudo bash scripts/run_imx623_fast.sh --frames 300 --dump-last --output-dir run-fast
cat run-fast/metrics.json run-fast/cleanup.json
# 可增加 --cpu-threads 2 或 --cpu-threads 1 降低 CPU 占用
```

板端完整证据在部署目录 `validation/`；本次代码提交同时保存下述精简统计、实验日志、模型和复现说明。原始 RAW/NV12 留在板端，不作为源码提交。

## 16 通道等价映射的实现细节

定义 p=`bayer_pack`，b 为 R/Gr/Gb/B 的编号 0–3，`sy/sx` 为单个模型格内的 Bayer 单元位置。打包通道为：

```text
channel = ((sy * p + sx) * 4 + b)
cell_y = region.y + p * (tensor_y - halo) + sy
cell_x = region.x + p * (tensor_x - halo) + sx
```

pre 按实际 Bayer 顺序把该单元的对应 RAW 像素归一化并写入通道，边界采用同颜色相位的复制填充。p=2 时，每个模型格保存全部 16 个原始像素，而不是对 4×4 RAW 做均值或丢弃采样；输出仍为 1936×1552，保持全部 1552 行。

对于模型输出单元 `(sy,sx,b)` 的高斯邻居 `(dy,dx)`，令 `iy=sy+dy`、`ix=sx+dx`：

```text
input_channel = (((iy mod p) * p + (ix mod p)) * 4 + b)
kernel_y = floor(iy / p) + 1
kernel_x = floor(ix / p) + 1
weight = outer([1,2,1], [1,2,1])[dy+1, dx+1] / 16
```

模型生成器将权重累加到上述位置。相邻单元在同一模型格内时由通道连接取得；跨格时由卷积空间偏移取得；b 不变，因此不会混合不同 Bayer 颜色。JSON 保存 p、模型几何、电平、通道顺序、ONNX/RKNN 哈希。启动器在调整媒体图之前检查这些条件。

RKNN 的 NC1HWC2 不能简单假定 channel stride=1 后一直连续。在 `HalfTensor::offset()` 中使用：

```text
offset = y * row_stride + x * pixel_stride
       + (channel / C2) * block_stride
       + (channel % C2) * channel_stride
```

这些 stride 的单位为 FP16 元素，来自原生 Tensor 描述符。与 TI 描述符中的字节 stride 不能直接互换。

## 线程与资源生命周期

- `RawPipeline` 创建并共享持久 RowExecutor，pre/post 使用同一组 worker；线程数为 1–4，默认通用 API 仍为 1，快速脚本选择 4。
- 调用线程处理分区 0，其余 N−1 个 worker 使用 generation 与条件变量接收任务；只在每阶段边界同步，不逐像素加锁。
- 每个 pre 任务写入不同模型行；每个 post 任务写入不同 RAW 行，输入 RAW 和 NPU 输出只读。模型推理保持在调用线程，不并发操作同一 RKNN context。
- worker 异常先记录，等待本阶段全部任务结束后再抛回调用者，防止尚在访问缓冲区的线程遇到提前释放。任务结束后才进入 NPU/ISP；失败不会提交部分结果。
- 退出时唤醒并 join worker；V4L2、媒体路由和 CPU profile 沿既有异常清理路径释放和恢复。SIGKILL、断电仍不属于用户态可保证的清理条件。
- CMake 导出增加 Threads 依赖，并已用安装后的独立消费工程验证 `find_package(rk3588-aiisp)` 与 bypass 链接运行。

## 验证材料索引

- [基线统计](validation/latency2/baseline.json)、[300 帧最终统计](validation/latency2/final.json)。
- [一线程](validation/latency2/threads-1.json)、[两线程](validation/latency2/threads-2.json)、[四线程](validation/latency2/threads-4.json) 的同版程序对照。
- [堆内存/原生输出微基准](validation/latency2/probe-native-4c.log)、[NHWC 输出反例](validation/latency2/probe-nhwc-4c.log)、[仅改 SIMD](validation/latency2/probe-simd-v2.log)。
- [早期 16 通道原型](validation/latency2/probe-pack16-prototype.log)：约 2.53 ms 来自原型模型；最终模型以 300 帧统计中的 2.681 ms 为准。
- [300 帧进度日志](validation/latency2/final-run.log)、[A/B 与异常恢复验收](validation/latency2/acceptance.log)、[部署哈希与恢复验证](validation/latency2/package-recovery.log)。
- [逐字节一致性](validation/latency2/same-input.json)、[随机 RAW 独立参考](validation/latency2/synthetic.json)、[内核错误模式结果](validation/latency2/kernel-errors.json)。
- [CPU profile](validation/latency2/performance-profile.json)、[恢复前](validation/latency2/governors-before.txt)、[恢复后](validation/latency2/governors-after.txt)。
- [环境、模型/程序/数据哈希与随机种子](validation/latency2/provenance.json)、[历史微基准源码与说明](validation/latency2/experiments/README.md)。

随机 RAW 对照使用 seed=3588；四种 Bayer 相位分别复制边界并用整数高斯核计算 `(sum+8)//16`，直接与关闭门控的实际 NPU 输出比较，验收阈值为 2 RAW12 灰阶。本次最大值确为 2。实采 A/B 的原图和三份输出哈希也保存在 provenance，未把“噪声门控保留原值”当作高斯内核正确性的唯一证据。

模型重新转换可能生成新的二进制哈希，部署时必须同时使用新生成的 JSON，不应只替换 RKNN 文件。提交中的模型哈希见对应清单；板端已部署程序与 SDK 当前构建程序的 SHA256 均为 `303b9e1db4cff47be0d8cd25bc4191773ff55bc145cbbdaa626272cb67503641`。

## 后续优化边界

当前主要剩余开销是 RAW→回读存储打包约 3.996 ms、ISP 提交与等待约 10.806 ms。它们不属于 pre/NPU/post 本次软件核优化，不应通过更换计时口径来宣称端到端低于 20 ms。进一步降低该口径需要单独验证 RAW 存储路径、拷贝及 ISP 调度；当前结论只对应本次配置和短时实测。
