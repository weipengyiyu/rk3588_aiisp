# 历史微基准源码

`stage_probe.cpp` 是本次实验使用的四通道微基准，保留原始实验逻辑，非运行时组件。它针对 IMX623 1936×1552 LE16 输入和基线 976×784 四通道模型；不要用于 16 通道新模型。

重现旧版 CPU 测量应使用 `6a47d6c` 的头文件和库；若链接新版本，会测到优化后的 pre/post。该源码每项预执行一次，再重复 30 次取平均。NPU run 单独计时，pre/post 分别用普通堆缓冲和 RKNN 缓冲，post 另测关闭门控的情况。

SDK 主机编译示意，BASE/BUILD 指向基线源码及其 ARM64 构建目录，SDK_ROOT 指向 SDK：

```bash
aarch64-linux-gnu-g++ -O3 -std=c++17 -march=armv8.2-a+fp16 \
  -I"$BASE/include" -I"$SDK_ROOT/external/rknpu2/runtime/Linux/librknn_api/include" \
  stage_probe.cpp "$BUILD/libaiisp_core.a" "$BUILD/libraw_core.a" -ldl -o stage-probe
```

板端在与主程序相同的、可恢复的 `performance_profile` 中运行：

```bash
LD_LIBRARY_PATH=./lib ./stage-probe models/fast-rect/bayer_gaussian.rknn nc input.le16.raw nc-output.raw
LD_LIBRARY_PATH=./lib ./stage-probe models/fast-rect/bayer_gaussian.rknn nhwc input.le16.raw nhwc-output.raw
```

本轮实测命令由启动器的 `performance_profile(True, ...)` 管理 CPU4–7 和 governor，未永久修改系统策略。输入是完整 1936×1552、LE16/低12位有效的数据，不是带3072字节行跨度的 CIF compact RAW。输入及输出原始文件留在板端，仓库只记录日志和哈希。

源码中的 query perf 数据与 rknn_run 墙钟耗时不应直接解释为纯 MAC 时间，也不能拿单次离线调用的启动成本替代主文档的稳定实时统计。
