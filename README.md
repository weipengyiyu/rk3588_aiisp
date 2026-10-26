# RK3588 RAW → pre → NPU → post → ISP 框架

最新第二轮优化：IMX623 全帧降噪平均 **7.958 ms**、P95 **9.391 ms**，pre 1.904 ms、RKNN backend 2.681 ms、post 2.886 ms；使用 16 通道打包和四个 CPU 线程，保持完整 RAW12 分辨率。RAW→NV12 的 P95 为 25.797 ms。详见 [性能与 TIDL 参考分析](docs/LATENCY_ANALYSIS_20261026.md)。

第一轮历史结果（2026-10-26）：已在 RK3588 / IMX623 上验证完整通路，并增加原生 FP16 快速路径。300 帧实测整帧降噪平均 20.446 ms、P95 20.489 ms；此项含 pre、NPU、post，RAW 解包到 NV12 返回的 P95 为 36.865 ms。测试使用可自动恢复的 CPU performance 配置。详见 [快速路径](docs/FAST_PATH.md) 和 [第一轮板端记录](docs/BOARD_VALIDATION_20261026.md)。

新增独立帧处理共享库 `libaiisp_processor.so`，通过版本化 C ABI 向 RSDI 或其他宿主提供 RAW/YUV 处理；接口、模型合同和部署见 [处理插件](docs/PROCESSOR_PLUGIN.md)。

位置：`/home/logic/rk3588-sdk/rk3588_jm/external/rk3588-aiisp`。

在现有 RK3588 SDK 中独立构建的 C++17 工程。基于同级 `imx623-aiisp` 已有原型整理，保留原型，拆出可复用的软件处理核心和 V4L2/ISP 适配库。

```text
Sensor → CSI → VICAP → DDR RAW12
                           ↓
           unpack → pre → RKNN/NPU → post → pack
                                              ↓
                    rkisp_rawrd2_s → RKISP → NV12
```

pre/post 指 NPU 的软件前后处理。ISP 使用 BSP 现有 RAW 回读入口，不将硬件 ISP 任意拆成前后两段。

## 模块

- `include/preprocess.h`、`src/preprocess.cpp`：Bayer 拆分，统一 R/Gr/Gb/B 通道，黑白电平归一化，分块及 halo 边界填充。
- `include/npu.h`、`src/npu.cpp`：`InferenceBackend` 接口与 RKNN Runtime 后端，检查输入输出张量形状和类型，支持显式 CPU 参考后端。
- `include/postprocess.h`、`src/postprocess.cpp`：检查有效输出数值，裁去 halo，恢复黑白电平并还原 Bayer 排列。
- `include/pipeline.h`、`src/pipeline.cpp`：`RawPipeline` 串联 pre→NPU→post，携带帧号、时间戳和曝光增益扩展字段，输出分阶段耗时。
- `include/raw.h`、`src/raw.cpp`：Rockchip compact RAW12 / little-aligned LE16 解包和打包。
- `include/video.h`、`src/video.cpp`：V4L2 MMAP 队列、格式校验、超时与资源恢复。
- `include/isp.h`、`src/isp.cpp`：`IspReadback` 将 RAW12 回灌 ISP，匹配 NV12 帧号；可返回独立持有的 NV12 数据。
- `src/main.cpp`：离线 RAW 处理和实时 Sensor→ISP 完整示例。
- `scripts/run_pipeline.py`：媒体拓扑发现、占用检查、pad/链路配置与退出恢复。

## 编译

在 SDK 主机上：

```bash
cd /home/logic/rk3588-sdk/rk3588_jm/external/rk3588-aiisp

# 主机编译与逻辑验证，不需要 NPU 或相机
cmake -S . -B build-host -DCMAKE_BUILD_TYPE=Release
cmake --build build-host --parallel 4
ctest --test-dir build-host --output-on-failure

# ARM64 交叉编译；先核对 BSP 的 RAW 回读 ioctl ABI
bash build.sh

# 生成 RK3588 FP16 示例模型；当前 SDK 主机已有此环境
/home/logic/.cache/rk3588-aiisp-toolkit-venv/bin/python models/build_baseline.py --tile 512

# 生成板端包，包含 bin、RKNN Runtime、模型和运行脚本
bash package.sh
```

产物：`build-aarch64/rk3588_aiisp`、`libaiisp_core.a`、`libaiisp_v4l2.a`、`libraw_core.a`；部署包：`dist/rk3588-aiisp/`。

## 全面清理编译生成文件

在工程目录执行：

```bash
./scripts/clean.sh --dry-run  # 可选：先列出清理目标
./scripts/clean.sh            # 全面清理
```

清除 `build`、`build-host`、`build-aarch64`、`dist`、`bin`、`lib`、本工程其他直接子目录中的 CMake 构建、可识别的原地 CMake 构建产物、Python 缓存和 `validation/framework-build.log`。同时按照 `baseline.json` 的生成标记及 SHA-256 清除基线生成器生成的 ONNX/RKNN 模型和清单（含 `models` 下的子目录）。修改过或无法确认归属的模型保留并提示。

保留源码、构建/模型生成脚本、文档、自定义模型、采集的 RAW/YUV 和运行验收数据。清理范围限定在本工程，不进入符号链接指向的目录；工程之外自定义指定的构建路径需单独清理。脚本不依赖调用时的当前目录，可重复执行。清理后重建：

```bash
bash build.sh
/home/logic/.cache/rk3588-aiisp-toolkit-venv/bin/python models/build_baseline.py --tile 512
bash package.sh
```

默认 SDK 根目录由本工程的 `../..` 得出。移到其他位置构建时，通过 `-DSDK_ROOT=/实际/SDK/根目录` 指定。`build.sh` 可通过 `CXX` 指定兼容编译器。不自动修改 SDK 的系统镜像、内核、设备树或其他相机工程。

## 板端运行

将 `dist/rk3588-aiisp` 复制到板端后进入该目录。以下默认参数对应现有 IMX623 的 1936×1552、RGGB、线性 RAW12 配置。

```bash
# 仅读取当前媒体拓扑
sudo python3 scripts/run_pipeline.py --plan

# 先验证 RAW 原样回灌 ISP
sudo python3 scripts/run_pipeline.py --mode bypass --frames 10 \
  --dump-last --output-dir run-bypass

# 再验证 pre → NPU → post → ISP
sudo python3 scripts/run_pipeline.py --mode rknn --frames 30 \
  --dump-last --output-dir run-rknn
```

其他 Sensor 可指定 `--sensor`（entity 名称的匹配子串）、`--width`、`--height`、`--bayer rggb|bggr|grbg|gbrg`。这些参数校验当前 Sensor 模式，不负责设置 Sensor 新模式，也不保证未实测 Sensor 的 ISP/IQ 已适配。自动发现目前针对单路线性 `stream_cif_mipi_id0 → rawrd2_s` 拓扑；HDR、多路同步等需另做适配。

启动器在相机被占用或已有 RAW 回读链路时退出。保留 CIF 到 ISP 的 Sensor 关联链路，只恢复自身改动的 pad 和回读链路。与旧原型共用历史 per-media 锁，避免两个工程竞争同一路由。`--frames 0` 持续处理；Ctrl+C 返回 130，不能作为成功验收。SIGKILL、断电和驱动故障不保证用户态恢复。

`--dump-last` 保存最后一帧：`input.le16.raw`、`denoised.le16.raw` 是小端 16 位容器、低 12 位有效；`last.nv12` 为 NV12，stride 见 `metrics.json`。不逐帧写盘。源帧率高于处理能力时取最新帧，记录 drain 与序号间隙。

## 离线运行

```bash
LD_LIBRARY_PATH=./lib ./bin/rk3588_aiisp \
  --mode rknn --model models/bayer_gaussian.rknn \
  --width 1936 --height 1552 --bayer rggb --tile 512 --halo 1 \
  --input-raw input.le16.raw --input-stride 3872 --storage le16 \
  --dump-output output.le16.raw --json offline.json
```

SDK 主机使用 `build-host/rk3588_aiisp --mode cpu-reference` 可验证不依赖 NPU 的文件处理链路。CPU 参考仅对应固定高斯示例，不是任意训练模型的参考实现。

## 集成到现有程序

已有 V4L2 RAW 采集程序可先用 `unpack12()` 获取像素，再调用 `RawPipeline::process()`，将返回的 `ProcessedFrame` 交给 `IspReadback::submit()`。调用者提供当前 Sensor 的 `FrameMetadata`，帧号与时间戳沿处理链路传递；回灌结果须匹配同一帧号。

```cpp
#include "pipeline.h"
#include "isp.h"

aiisp::ProcessingConfig config;
config.tile = 512;
config.halo = 1;
config.bayer = aiisp::BayerPattern::RGGB;
auto backend = aiisp::make_backend("rknn", model_path, config.tile);
aiisp::RawPipeline pipeline(config, std::move(backend));
aiisp::IspReadback isp(rawrd_node, isp_subdev, yuv_node, width, height, config.bayer);

// unpacked_raw 来自当前采集帧；框架只支持低 12 位有效的 uint16_t 像素。
aiisp::FrameMetadata metadata;
metadata.sequence = sensor_sequence;
metadata.timestamp_ns = sensor_timestamp_ns;
auto processed = pipeline.process(unpacked_raw, width, height, metadata);
auto output = isp.submit(processed, 5000, true);
// output.nv12 持有独立数据；行跨度为 output.stride。
```

同一构建树中用 `add_subdirectory()` 加入本工程，链接 `aiisp_core`（软件核心）或 `aiisp_v4l2`（含 ISP）。头文件目录由 target 传递。工程也支持 `cmake --install build-aarch64 --prefix /打包目录` 导出静态库、头文件和 CMake targets；安装后使用 `find_package(rk3588-aiisp CONFIG REQUIRED)` 与 `rk3588::aiisp_core` / `rk3588::aiisp_v4l2`。运行环境仍需 RKNN Runtime。

`RawPipeline`、`IspReadback` 均为同步单帧接口，每个实例由一个线程使用。没有隐式 CPU 降级；bypass 只在明确配置时原样复制 RAW。

## 当前范围

支持 RAW12、四种 Bayer 顺序、偶数尺寸、compact/LE16 内存格式、单输入输出 FP16/FP32 RKNN 模型及可配置 halo。内存格式与 Sensor/ISP 实际协商结果须一致；Rockchip compact 并非 CSI 线上 MIPI RAW12 字节布局。

示例模型为固定权重 3×3 高斯卷积，未进行 Sensor 数据训练，不代表生产画质。尚未接入自动 AE/AWB/AF、Sensor 专用 IQ、时域模型、HDR、全程 DMA-BUF 共享或 RTSP 编码。曝光增益元数据当前只传递，示例后端不消费。黑电平处理与 ISP BLC 的职责必须在接入生产模型时统一，避免重复扣除。

模型接入规则见 [模型接口合同](docs/MODEL_CONTRACT.md)，历史验证见 [验证记录](docs/VALIDATION.md)，最新记录见 [第二轮优化详解](docs/LATENCY_ANALYSIS_20261026.md)。
