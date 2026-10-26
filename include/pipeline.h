#pragma once
#include "npu.h"
#include "postprocess.h"
namespace aiisp {
struct FrameMetadata {
    uint32_t sequence=0;
    uint64_t timestamp_ns=0;
    // Transport fields for later 3A integration; this baseline does not consume them.
    float exposure_us=0, analog_gain=0, digital_gain=0;
};
struct StageTimings { double pre_ms=0, npu_ms=0, post_ms=0; uint64_t tiles=0; };
struct ProcessedFrame {
    std::vector<uint16_t> raw;
    int width=0,height=0;
    BayerPattern bayer=BayerPattern::RGGB;
    FrameMetadata metadata;
    StageTimings timings;
};
class RawPipeline {
    ProcessingConfig config_;
    std::shared_ptr<RowExecutor> executor_;
    Preprocessor pre_;
    Postprocessor post_;
    std::unique_ptr<InferenceBackend> backend_;
    std::vector<float> input_tile_,output_tile_;
public:
    RawPipeline(ProcessingConfig config,std::unique_ptr<InferenceBackend> backend);
    ProcessedFrame process(const std::vector<uint16_t>& input,int width,int height,
                           FrameMetadata metadata={});
    void process_into(const std::vector<uint16_t>& input,int width,int height,
                      FrameMetadata metadata,ProcessedFrame& result);
    uint64_t calls() const;
};
}
