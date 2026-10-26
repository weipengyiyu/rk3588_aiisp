# RAW / YUV 通用帧处理插件

`libaiisp_processor.so` 是独立的同步帧处理模块。它不采集相机、不修改媒体图、不持有 ISP，也不依赖 RSDI 头文件；RSDI 或其他宿主通过 `include/processor_plugin.h` 中的版本化 C ABI 使用它。

## 接口和所有权

加载 `aiisp_processor_entry(AIISP_PROCESSOR_ABI_V1)`，检查 API 大小/版本及函数指针，再调用 `create → process（多次）→ destroy`。不支持的 ABI 返回 NULL。错误返回 0，错误缓冲始终有界写入，异常不跨越 C ABI；输出仅在 process 成功时有效。

配置固定输入格式、宽高、模型几何、packing、CPU 线程和后处理方式；更换配置须重建实例。每个句柄由调用方串行使用，不并发调用 process/destroy。模型加载、RKNN context、RAW pipeline、张量和 worker 在实例内持续复用。

`aiisp_frame_v1` 携带 FourCC、宽高、sequence、timestamp 和最多两个 CPU 可见 plane；每个 plane 都有指针、capacity 和字节 stride。NV12/NV21 使用两个 plane；packed YUV 和 RAW 使用一个。输入输出均由宿主分配，不支持原地处理或 plane 重叠。padding 清零，sequence/timestamp 原样传回。像素数据、C++ 对象、分配器和所有权不跨模块转移。

若宿主使用 DMA-BUF，宿主负责 mmap 以及 CPU 访问前后 `DMA_BUF_IOCTL_SYNC`。process 返回后，模块不会继续访问输入输出。它不是异步 NPU 提交接口，不接收未完成 fence，也不意味着相机到 ISP 全程零复制。

## RAW 合同

- 输入：RG12/BG12/BA12/GB12，显式声明 LE16 或本 BSP compact12 存储；输出：相同 Bayer 顺序的 LE16 RAW12。
- 归一化、原生 FP16、16 通道 Bayer packing、NEON 和共享 worker 复用已有 RawPipeline；支持四种 Bayer 顺序。
- 当前默认模型为 `models/fast-pack16/bayer_gaussian.rknn`：488×392×16、bayer_pack=2、halo=1，1936×1552 一次推理覆盖整帧。
- edge_post=1 需要纯 Gaussian strength=1 模型；门控参数保持 strength=.75、threshold=96。RAW 插件配置本身允许指定电平和门控参数，RSDI 当前路线采用上述默认电平/门控值。
- `AIISP_BACKEND_NATIVE` 使用 RKNN 原生 FP16 I/O，模型失败直接报错；不会自动回退 CPU。
- `REFERENCE` 是明确选择的离线参考，`BYPASS` 是明确选择的透传；两者的 npu_calls 为 0，不能冒充 NPU 验证。

插件不推断任意模型的语义。更换模型须确保形状、通道、归一化、绝对值输出、halo 和 edge 合同匹配；RSDI RAW 启动包装器会检查配套 manifest 和 RKNN SHA256。

## YUV 合同

输入/输出支持 NV12、NV21、YUYV、UYVY，保持原格式、尺寸和色度字节，仅对 Y 进行空间 Gaussian。按原始 Y 编码值除以 255 归一化，不做 RGB 转换、色彩矩阵转换或 full/limited range 重映射。输出是绝对归一化 Y，非噪声残差。

- `yuv_pack=1`：单通道 Y，模型 `[1,1,H,W]`。
- `yuv_pack=4`：4×4 个相邻 Y 样本打包为 16 通道，通道号 `by*4+bx`。空间尺寸缩小只是可逆重排，保留所有像素。Gaussian 的邻接关系重映射到相邻通道和模型格，四周按原始像素坐标复制边界。
- 默认示例 `models/yuv-pack16/luma_gaussian.rknn` 为 488×392×16，halo=1；1936×1552 一次推理。不得用相同几何的 RAW 模型替换，二者通道/权重语义不同。
- YUV 使用标准 RKNN float 输入输出适配器，内部模型仍为 FP16、实际执行 NPU。此前原生 I/O 实验未通过板端独立 oracle，未作为可选择的 YUV 运行路径发布；不是 CPU 回退。
- CPU reference 仅支持 yuv_pack=1。NV12M/NV21M、多 FD、10-bit YUV、RGB、多输入状态模型和时域降噪不属于此版合同。

CPU pre/post 继续使用持久行 worker。YUV 当前未实现 RAW 快速路径同等程度的 SIMD/native I/O 优化；完整 RSDI 链路约 12 fps，不能当作 20 ms/30 fps 能力。

生成模型：

```bash
PY=/home/logic/.cache/rk3588-aiisp-toolkit-venv/bin/python
$PY models/build_yuv.py --out models/yuv-pack16
$PY models/build_yuv.py --out models/yuv-pack16-test --tile 32 --tile-height 32 --pack 4
# 可另行生成 pack=1 单通道模型，形状和配置必须同步修改
```

## 构建和使用

```bash
cmake -S . -B build-host -DCMAKE_BUILD_TYPE=Release
cmake --build build-host -j4
ctest --test-dir build-host --output-on-failure
bash build.sh
cmake --install build-aarch64 --prefix /tmp/aiisp-sdk-arm-plugin
bash package.sh
```

安装 SDK 包含 C 头文件、动态处理库、原有 C++ SDK 和媒体图工具；board bundle 也包含处理库及匹配模型。宿主从可信绝对路径加载库，`dlclose` 必须晚于全部实例销毁。动态插件不是隔离安全沙箱。

板端直接验证实际 `.so`：

```bash
LD_LIBRARY_PATH=lib bin/test_processor_plugin lib/libaiisp_processor.so \
  models/fast-pack16/bayer_gaussian.rknn models/yuv-pack16-test/luma_gaussian.rknn
# 验证整帧几何的 YUV 模型：
LD_LIBRARY_PATH=lib bin/test_processor_plugin lib/libaiisp_processor.so \
  models/fast-pack16/bayer_gaussian.rknn models/yuv-pack16/luma_gaussian.rknn 488 392
```

独立测试用四种 Bayer、四种 YUV、非紧凑 stride、变化的色度、跨 tile 和部分边界输入，逐像素对照整数 Gaussian：RAW 最大误差 2 DN，YUV 最大误差 1 DN；BYPASS 精确一致，色度字节保持一致。还检查 padding、非法帧/ABI、重叠缓冲、拒绝后的实例复用。

RSDI 通过其 Camera 插件内部的处理适配器加载本模块，采集、输出 DMA 池和可选 ISP 回灌属于宿主。具体配置及三条通路实测见 RSDI 仓库 `docs/AIISP.md`。固定 Gaussian 是工程示例，未因此完成 IMX623 噪声训练、生产 IQ 或真实 YUV MIPI 相机验收。

## 本次验收记录（2026-10-26）

主机 CTest 6/6、ARM64 构建、C 语言头文件编译和 bundle manifest 检查通过。板端以 130×126 合成输入覆盖跨 tile/部分 Bayer 单元，并分别使用 32×32 和最终 488×392 YUV 模型：

- [小张量独立 oracle](validation/processor-plugin-20261026/oracle-small.log)。
- [最终模型独立 oracle](validation/processor-plugin-20261026/oracle-full.log)。
- [部署程序、模型哈希和运行条件](validation/processor-plugin-20261026/provenance.json)。

两个 `.so` 的 SDK 构建哈希与板端实测哈希一致。RSDI 三条 120 帧真实链路以及 SIGINT/SIGTERM、错误库拒绝、媒体图/governor 恢复另存于 RSDI 验证目录。测试后原有 imx623_back 服务已恢复并确认继续交付。
