#include "pipeline.h"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>
namespace aiisp {
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
}
RawPipeline::RawPipeline(ProcessingConfig config,std::unique_ptr<InferenceBackend> backend):
    config_(config),pre_(config),post_(config),backend_(std::move(backend)) {
    if(!config_.bypass && !backend_) throw std::runtime_error("inference backend is required");
    if(config_.bypass && backend_) throw std::runtime_error("bypass must not create an inference backend");
}
ProcessedFrame RawPipeline::process(const std::vector<uint16_t>& input,int w,int h,FrameMetadata metadata) {
    ProcessedFrame result;process_into(input,w,h,metadata,result);return result;
}
void RawPipeline::process_into(const std::vector<uint16_t>& input,int w,int h,FrameMetadata metadata,ProcessedFrame& result) {
    uint16_t maximum=0;for(auto v:input)maximum=std::max(maximum,v);
    if(w<2 || h<2 || w>8192 || h>8192 || w%2 || h%2 || input.size()!=size_t(w)*h || maximum>4095)
        throw std::runtime_error("pipeline requires an even-sized unpacked RAW12 image");
    result.metadata=metadata; result.width=w; result.height=h; result.bayer=config_.bayer;result.timings={};
    if(config_.edge_post && (!backend_ || !backend_->native_io()))throw std::runtime_error("edge post requires native FP16 backend and a pure Gaussian model");
    if(config_.bypass) {result.raw=input; return;}
    result.raw.resize(input.size());
    const int step=config_.tile-2*config_.halo,rows=config_.rows()-2*config_.halo;
    for(int y=0;y<h/2;y+=rows) for(int x=0;x<w/2;x+=step) {
        const TileRegion region{x,y,std::min(step,w/2-x),std::min(rows,h/2-y)};
        auto tick=Clock::now();
        if(backend_->native_io()) pre_.prepare_native(input,w,h,region,backend_->input_tensor());
        else pre_.prepare(input,w,h,region,input_tile_);
        result.timings.pre_ms+=ms(tick);
        tick=Clock::now();
        if(backend_->native_io()) backend_->run_native();
        else backend_->run(input_tile_,output_tile_);
        result.timings.npu_ms+=ms(tick);
        tick=Clock::now();
        if(backend_->native_io()) post_.merge_native(backend_->output_tensor(),region,w,h,result.raw,&input);
        else post_.merge(output_tile_,region,w,h,result.raw);
        result.timings.post_ms+=ms(tick);
        ++result.timings.tiles;
    }
}
uint64_t RawPipeline::calls() const { return backend_?backend_->calls():0; }
}
