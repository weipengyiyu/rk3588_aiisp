#include "processor_plugin.h"
#include "pipeline.h"
#include "raw.h"
#include <linux/videodev2.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
bool raw_format(uint32_t f){return f==V4L2_PIX_FMT_SRGGB12 || f==V4L2_PIX_FMT_SBGGR12 || f==V4L2_PIX_FMT_SGRBG12 || f==V4L2_PIX_FMT_SGBRG12;}
bool semi(uint32_t f){return f==V4L2_PIX_FMT_NV12 || f==V4L2_PIX_FMT_NV21;}
aiisp::BayerPattern bayer(uint32_t f){
    if(f==V4L2_PIX_FMT_SRGGB12)return aiisp::BayerPattern::RGGB;
    if(f==V4L2_PIX_FMT_SBGGR12)return aiisp::BayerPattern::BGGR;
    if(f==V4L2_PIX_FMT_SGRBG12)return aiisp::BayerPattern::GRBG;
    if(f==V4L2_PIX_FMT_SGBRG12)return aiisp::BayerPattern::GBRG;
    throw std::runtime_error("unsupported Bayer format");
}
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void error(char* p,size_t n,const char* message){if(p && n)std::snprintf(p,n,"%s",message);}
struct Processor {
    aiisp_processor_config_v1 c;
    bool raw;
    std::unique_ptr<aiisp::RawPipeline> pipeline;
    std::unique_ptr<aiisp::InferenceBackend> yuv_backend;
    std::unique_ptr<aiisp::RowExecutor> executor;
    std::vector<uint16_t> unpacked;
    aiisp::ProcessedFrame processed;
    std::vector<uint16_t> y_input,y_output;
    std::vector<float> y_float_input,y_float_output;
    explicit Processor(const aiisp_processor_config_v1& in):c(in),raw(raw_format(in.fourcc)) {
        require(c.width>=2 && c.height>=2 && c.width<=8192 && c.height<=8192 && !(c.width%2) && !(c.height%2),"invalid frame dimensions");
        require(raw || semi(c.fourcc) || c.fourcc==V4L2_PIX_FMT_YUYV || c.fourcc==V4L2_PIX_FMT_UYVY,"unsupported frame format");
        require(c.backend>=1 && c.backend<=3 && c.raw_storage<=1,"invalid backend/storage");
        require(c.cpu_threads>=1 && c.cpu_threads<=4 && c.tile_width>=3 && c.tile_width<=1024 && c.tile_height>=3 && c.tile_height<=1024 &&
                c.halo==1 && (c.bayer_pack==1 || c.bayer_pack==2),"invalid tensor/worker configuration");
        require(raw || c.yuv_pack==1 || c.yuv_pack==4,"YUV packing must be 1 or 4");
        require(raw || (!c.edge_post && c.bayer_pack==1 && c.raw_storage==0),"YUV uses a single luma channel, no Bayer packing/RAW edge gate");
        require(c.backend!=AIISP_BACKEND_NATIVE || (c.model && *c.model),"native mode requires a model");
        if(raw){
            aiisp::ProcessingConfig config;
            config.tile=c.tile_width;config.tile_height=c.tile_height;config.halo=c.halo;
            config.bayer_pack=c.bayer_pack;config.cpu_threads=c.cpu_threads;config.bayer=bayer(c.fourcc);
            config.black_level=c.black_level;config.white_level=c.white_level;
            config.edge_post=c.edge_post!=0;config.edge_strength=c.edge_strength;config.edge_threshold=c.edge_threshold;
            config.bypass=c.backend==AIISP_BACKEND_BYPASS;if(config.bypass)config.bayer_pack=1;config.validate();
            auto backend=config.bypass?nullptr:aiisp::make_backend(c.backend==AIISP_BACKEND_NATIVE?"rknn-native":"cpu-reference",
                c.model?c.model:"",c.tile_width,1.0f,c.tile_height,c.bayer_pack);
            pipeline=std::make_unique<aiisp::RawPipeline>(config,std::move(backend));
        }else{
            executor=std::make_unique<aiisp::RowExecutor>(c.cpu_threads);
            require(raw || c.yuv_pack==1 || c.backend!=AIISP_BACKEND_REFERENCE,"reference backend uses unpacked YUV");
            if(c.backend==AIISP_BACKEND_NATIVE){
                // YUV native I/O experiments did not pass the Runtime 2.3.0 board oracle.
                // Use the standard RKNN float adapter for YUV; never fall back to CPU.
                yuv_backend=aiisp::make_backend("rknn",c.model,c.tile_width,1.0f,c.tile_height,1,c.yuv_pack==4?16:1);
                y_float_input.resize(size_t(c.tile_width)*c.tile_height*c.yuv_pack*c.yuv_pack);
            }
            y_input.resize(size_t(c.tile_width)*c.tile_height);y_output.resize(y_input.size());
        }
        c.model=nullptr; // no borrowed configuration pointer survives create
    }
    void validate(const aiisp_frame_v1& f,bool output) const {
        require(f.struct_size==sizeof(f) && f.fourcc==c.fourcc && f.width==c.width && f.height==c.height,"frame contract mismatch");
        require(f.plane_count==(semi(c.fourcc)?2u:1u),"incorrect plane count");
        const uint64_t row=raw?(output || c.raw_storage==AIISP_RAW_LE16?c.width*2:c.width*3/2):c.width*(semi(c.fourcc)?1:2);
        for(uint32_t i=0;i<f.plane_count;++i){
            const auto& p=f.planes[i];const auto rows=i?c.height/2:c.height;
            require(p.data && p.stride>=row && p.stride<=67108864 && p.capacity<=67108864 && p.capacity>=p.stride*rows,"short frame plane/stride");
        }
        if(f.plane_count==2){
            const auto a=reinterpret_cast<uintptr_t>(f.planes[0].data),b=reinterpret_cast<uintptr_t>(f.planes[1].data);
            require(a<=UINTPTR_MAX-f.planes[0].capacity && b<=UINTPTR_MAX-f.planes[1].capacity &&
                (a+f.planes[0].capacity<=b || b+f.planes[1].capacity<=a),"frame planes overlap");
        }
    }
    int luma_offset()const{return c.fourcc==V4L2_PIX_FMT_UYVY?1:0;}
    uint8_t luma(const aiisp_frame_v1& f,int y,int x)const{return f.planes[0].data[y*f.planes[0].stride+x*(semi(c.fourcc)?1:2)+luma_offset()];}
    void yuv_packed(const aiisp_frame_v1& in,aiisp_frame_v1& out,aiisp_processor_stats_v1& stats){
        const int tw=c.tile_width,th=c.tile_height,stepx=(tw-2)*4,stepy=(th-2)*4;
        for(int oy=0;oy<int(c.height);oy+=stepy)for(int ox=0;ox<int(c.width);ox+=stepx){
            auto tick=Clock::now();
            executor->run(th,[&](int begin,int end){for(int y=begin;y<end;++y)for(int x=0;x<tw;++x)for(int by=0;by<4;++by)for(int bx=0;bx<4;++bx){
                const int yy=std::clamp(oy+(y-1)*4+by,0,int(c.height)-1),xx=std::clamp(ox+(x-1)*4+bx,0,int(c.width)-1);
                y_float_input[(size_t(y)*tw+x)*16+by*4+bx]=luma(in,yy,xx)/255.0f;
            }});stats.pre_ms+=ms(tick);tick=Clock::now();yuv_backend->run(y_float_input,y_float_output);stats.backend_ms+=ms(tick);tick=Clock::now();
            const int rows=std::min(stepy,int(c.height)-oy),cols=std::min(stepx,int(c.width)-ox);
            executor->run(rows,[&](int begin,int end){for(int y=begin;y<end;++y)for(int x=0;x<cols;++x){
                const float v=y_float_output[(size_t(y/4+1)*tw+x/4+1)*16+(y%4)*4+x%4];
                require(std::isfinite(v),"YUV model returned NaN/Inf");
                out.planes[0].data[(oy+y)*out.planes[0].stride+(ox+x)*(semi(c.fourcc)?1:2)+luma_offset()]=uint8_t(std::lround(std::clamp(v,0.0f,1.0f)*255));
            }});stats.post_ms+=ms(tick);
        }
    }
    void yuv(const aiisp_frame_v1& in,aiisp_frame_v1& out,aiisp_processor_stats_v1& stats){
        const size_t bytes=c.width*(semi(c.fourcc)?1:2);
        for(uint32_t i=0;i<in.plane_count;++i)for(uint32_t y=0;y<(i?c.height/2:c.height);++y){
            auto* dst=out.planes[i].data+y*out.planes[i].stride;
            std::memset(dst,0,out.planes[i].stride);std::memcpy(dst,in.planes[i].data+y*in.planes[i].stride,bytes);
        }
        if(c.backend==AIISP_BACKEND_BYPASS)return;
        if(c.yuv_pack==4){yuv_packed(in,out,stats);return;}
        const int tw=c.tile_width,th=c.tile_height,stepx=tw-2,stepy=th-2;
        for(int oy=0;oy<int(c.height);oy+=stepy)for(int ox=0;ox<int(c.width);ox+=stepx){
            aiisp::HalfTensor input,output;
            {input.data=y_input.data();input.elements=y_input.size();input.width=tw;input.height=th;input.channels=1;input.row_stride=tw;input.pixel_stride=input.channel_stride=1;output=input;output.data=y_output.data();}
            auto tick=Clock::now();
            executor->run(th,[&](int begin,int end){for(int y=begin;y<end;++y)for(int x=0;x<tw;++x){
                const float value=luma(in,std::clamp(oy+y-1,0,int(c.height)-1),std::clamp(ox+x-1,0,int(c.width)-1))/255.0f;
                if(yuv_backend)y_float_input[size_t(y)*tw+x]=value;
                else input.data[input.offset(y,x,0)]=aiisp::float_to_half(value);
            }});
            stats.pre_ms+=ms(tick);tick=Clock::now();
            if(yuv_backend)yuv_backend->run(y_float_input,y_float_output);
            else for(int y=1;y<th-1;++y)for(int x=1;x<tw-1;++x){
                float value=0;for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)
                    value+=aiisp::half_to_float(input.data[input.offset(y+dy,x+dx,0)])*(dy?1:2)*(dx?1:2)/16;
                output.data[output.offset(y,x,0)]=aiisp::float_to_half(value);
            }
            stats.backend_ms+=ms(tick);tick=Clock::now();
            const int rows=std::min(stepy,int(c.height)-oy),cols=std::min(stepx,int(c.width)-ox);
            executor->run(rows,[&](int begin,int end){for(int y=begin;y<end;++y)for(int x=0;x<cols;++x){
                const float v=yuv_backend?y_float_output[size_t(y+1)*tw+x+1]:aiisp::half_to_float(output.data[output.offset(y+1,x+1,0)]);
                require(std::isfinite(v),"YUV model returned NaN/Inf");
                out.planes[0].data[(oy+y)*out.planes[0].stride+(ox+x)*(semi(c.fourcc)?1:2)+luma_offset()]=uint8_t(std::lround(std::clamp(v,0.0f,1.0f)*255));
            }});stats.post_ms+=ms(tick);
        }
    }
    void run(const aiisp_frame_v1& in,aiisp_frame_v1& out,aiisp_processor_stats_v1& stats){
        validate(in,false);validate(out,true);
        for(uint32_t i=0;i<in.plane_count;++i)for(uint32_t j=0;j<out.plane_count;++j){
            const auto a=reinterpret_cast<uintptr_t>(in.planes[i].data),b=reinterpret_cast<uintptr_t>(out.planes[j].data);
            require(a<=UINTPTR_MAX-in.planes[i].capacity && b<=UINTPTR_MAX-out.planes[j].capacity &&
                (a+in.planes[i].capacity<=b || b+out.planes[j].capacity<=a),"input/output planes overlap");
        }
        const auto tick=Clock::now();
        if(raw){
            aiisp::unpack12(in.planes[0].data,in.planes[0].capacity,c.width,c.height,in.planes[0].stride,
                c.raw_storage==AIISP_RAW_COMPACT12?aiisp::Storage::Compact12:aiisp::Storage::LE16,unpacked);
            const auto before=pipeline->calls();
            aiisp::FrameMetadata metadata;metadata.sequence=in.sequence;metadata.timestamp_ns=in.timestamp_ns;
            pipeline->process_into(unpacked,c.width,c.height,metadata,processed);
            stats.pre_ms=processed.timings.pre_ms;stats.backend_ms=processed.timings.npu_ms;stats.post_ms=processed.timings.post_ms;
            stats.npu_calls=pipeline->calls()-before;
            aiisp::pack12(processed.raw,c.width,c.height,out.planes[0].data,out.planes[0].capacity,out.planes[0].stride,aiisp::Storage::LE16);
        }else{const auto before=yuv_backend?yuv_backend->calls():0;yuv(in,out,stats);stats.npu_calls=yuv_backend?yuv_backend->calls()-before:0;}
        stats.total_ms=ms(tick);out.sequence=in.sequence;out.timestamp_ns=in.timestamp_ns;
    }
};
int create(const aiisp_processor_config_v1* c,void** out,char* e,size_t n) noexcept {
    if(out)*out=nullptr;
    try{require(c && out && c->struct_size==sizeof(*c),"invalid processor configuration ABI");*out=new Processor(*c);error(e,n,"");return 1;}
    catch(const std::exception& x){error(e,n,x.what());return 0;}catch(...){error(e,n,"processor create failed");return 0;}
}
int process(void* h,const aiisp_frame_v1* in,aiisp_frame_v1* out,aiisp_processor_stats_v1* stats,char* e,size_t n) noexcept {
    try{require(h && in && out && stats && stats->struct_size==sizeof(*stats),"invalid process ABI");*stats={};stats->struct_size=sizeof(*stats);
        static_cast<Processor*>(h)->run(*in,*out,*stats);error(e,n,"");return 1;}
    catch(const std::exception& x){error(e,n,x.what());return 0;}catch(...){error(e,n,"processor failed");return 0;}
}
void destroy(void* h) noexcept {delete static_cast<Processor*>(h);}
const aiisp_processor_api_v1 api{sizeof(api),AIISP_PROCESSOR_ABI_V1,"rk3588-aiisp",create,process,destroy};
}
extern "C" __attribute__((visibility("default"))) const aiisp_processor_api_v1* aiisp_processor_entry(uint32_t version){return version==AIISP_PROCESSOR_ABI_V1?&api:nullptr;}
