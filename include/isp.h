#pragma once
#include "pipeline.h"
#include <memory>
#include <string>
namespace aiisp {
struct IspFrame {
    std::vector<uint8_t> nv12; // Owned copy, populated only when requested.
    FrameMetadata metadata;
    unsigned stride=0;
    double pack_ms=0,isp_ms=0;
};
// RK3588 BSP linear RAW12 -> ISP -> NV12 adapter. One frame in flight.
class IspReadback {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    IspReadback(const std::string& rawrd,const std::string& subdev,const std::string& yuv,
                int width,int height,BayerPattern bayer);
    ~IspReadback();
    IspFrame submit(const ProcessedFrame& frame,unsigned timeout_ms,bool copy_nv12=false);
    unsigned stride() const;
    void restore();
};
uint32_t raw12_fourcc(BayerPattern bayer);
}
