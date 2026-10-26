#include "postprocess.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace aiisp {
Postprocessor::Postprocessor(ProcessingConfig config,std::shared_ptr<RowExecutor> executor):config_(config),positions_(bayer_positions(config.bayer)) {
    config_.validate();
    executor_=executor?executor:(config.cpu_threads>1?std::make_shared<RowExecutor>(config.cpu_threads):nullptr);
}
void Postprocessor::merge(const std::vector<float>& tensor,TileRegion r,int w,int h,
                          std::vector<uint16_t>& raw) const {
    if(config_.bayer_pack!=1)throw std::runtime_error("packed Bayer requires native FP16 postprocessing");
    const int t=config_.tile;
    if(w<2 || h<2 || w>8192 || h>8192 || w%2 || h%2 || raw.size()!=size_t(w)*h ||
       tensor.size()!=size_t(t)*config_.rows()*4 || r.x<0 || r.y<0 || r.width<1 || r.height<1 ||
       r.x>w/2-r.width || r.y>h/2-r.height || r.width>t-2*config_.halo || r.height>config_.rows()-2*config_.halo)
        throw std::runtime_error("invalid postprocessing tensor or tile region");
    for(int y=0;y<r.height;++y) for(int x=0;x<r.width;++x) for(int c=0;c<4;++c) {
        const float value=tensor[(size_t(y+config_.halo)*t+x+config_.halo)*4+c];
        if(!std::isfinite(value)) throw std::runtime_error("model output contains NaN/Inf");
        const float restored=std::clamp(value,0.0f,1.0f)*(config_.white_level-config_.black_level)+config_.black_level;
        const int p=positions_[c];
        raw[size_t(2*(r.y+y)+p/2)*w+2*(r.x+x)+p%2]=uint16_t(std::lround(restored));
    }
}
void Postprocessor::merge_native(HalfTensor tensor,TileRegion r,int w,int h,
                                std::vector<uint16_t>& raw,const std::vector<uint16_t>* original) const {
    const int t=config_.tile,halo=config_.halo,pack=config_.bayer_pack;
    tensor.validate(t,config_.rows(),config_.channels());
    if(config_.edge_post && (!original || original->size()!=raw.size()))
        throw std::runtime_error("edge post needs the matching original RAW frame");
    if(w<2 || h<2 || w%2 || h%2 || raw.size()!=size_t(w)*h ||
       r.x<0 || r.y<0 || r.width<1 || r.height<1 || r.x>w/2-r.width || r.y>h/2-r.height ||
       r.width>(t-2*halo)*pack || r.height>(config_.rows()-2*halo)*pack)
        throw std::runtime_error("invalid native postprocessing region");
    const float range=config_.white_level-config_.black_level;
    const float inverse_threshold=1.0f/config_.edge_threshold;
#if defined(__aarch64__)
    // Process complete 2x2 Bayer cells. Keep channels packed instead of
    // transposing four cells into four planes and transposing them back.
    const auto zero=vdupq_n_f32(0),one=vdupq_n_f32(1),black=vdupq_n_f32(config_.black_level);
    const auto scale=vdupq_n_f32(range),threshold=vdupq_n_f32(inverse_threshold);
    const auto strength=vdupq_n_f32(config_.edge_strength),half=vdupq_n_f32(0.5f);
    const bool edge=config_.edge_post,identity=config_.bayer==BayerPattern::RGGB;
    uint8_t indices[16];
    for(int i=0;i<8;++i) {
        int canonical=0;while(positions_[canonical]!=i%4)++canonical;
        const int p=canonical+(i/4)*4;
        indices[2*i]=uint8_t(2*p);indices[2*i+1]=uint8_t(2*p+1);
    }
    const auto order=vld1q_u8(indices);
    const auto convert=[&](uint16x8_t packed,uint16x8_t source) {
        if(!identity)packed=vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(packed),order));
        auto a=vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(packed)));
        auto b=vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(packed)));
        a=vaddq_f32(vmulq_f32(vminq_f32(vmaxq_f32(a,zero),one),scale),black);
        b=vaddq_f32(vmulq_f32(vminq_f32(vmaxq_f32(b,zero),one),scale),black);
        if(edge) {
            const auto sa=vcvtq_f32_u32(vmovl_u16(vget_low_u16(source)));
            const auto sb=vcvtq_f32_u32(vmovl_u16(vget_high_u16(source)));
            const auto da=vsubq_f32(a,sa),db=vsubq_f32(b,sb);
            const auto ga=vmaxq_f32(vsubq_f32(one,vmulq_f32(vabsq_f32(da),threshold)),zero);
            const auto gb=vmaxq_f32(vsubq_f32(one,vmulq_f32(vabsq_f32(db),threshold)),zero);
            a=vaddq_f32(sa,vmulq_f32(vmulq_f32(da,ga),strength));
            b=vaddq_f32(sb,vmulq_f32(vmulq_f32(db,gb),strength));
        }
        return vcombine_u16(vmovn_u32(vcvtq_u32_f32(vaddq_f32(a,half))),
                            vmovn_u32(vcvtq_u32_f32(vaddq_f32(b,half))));
    };
#endif
    const auto rows=[&](int begin,int end) {
    for(int y=begin;y<end;++y) for(int x=0;x<r.width;) {
#if defined(__aarch64__)
        if(tensor.channel_stride==1 && (tensor.channel_block==0 || tensor.channel_block>=8) && x+3<r.width) {
            uint16x8_t a,b;
            if(pack==1) {
                const auto* base=tensor.data+tensor.offset(y+halo,x+halo,0);
                a=vcombine_u16(vld1_u16(base),vld1_u16(base+tensor.pixel_stride));
                b=vcombine_u16(vld1_u16(base+2*tensor.pixel_stride),vld1_u16(base+3*tensor.pixel_stride));
            } else {
                a=vld1q_u16(tensor.data+tensor.offset(y/2+halo,x/2+halo,(y%2)*8));
                b=vld1q_u16(tensor.data+tensor.offset(y/2+halo,x/2+halo+1,(y%2)*8));
            }
            const auto exponent=vdupq_n_u16(0x7c00);
            const auto invalid=vorrq_u16(vceqq_u16(vandq_u16(a,exponent),exponent),
                                        vceqq_u16(vandq_u16(b,exponent),exponent));
            if(vmaxvq_u16(invalid))throw std::runtime_error("model output contains NaN/Inf");
            auto sa=vdupq_n_u32(0),sb=vdupq_n_u32(0);
            if(edge) {
                const auto* src=original->data()+size_t(2*(r.y+y))*w+2*(r.x+x);
                const auto top=vreinterpretq_u32_u16(vld1q_u16(src));
                const auto bottom=vreinterpretq_u32_u16(vld1q_u16(src+w));
                sa=vzip1q_u32(top,bottom);sb=vzip2q_u32(top,bottom);
            }
            const auto ra=vreinterpretq_u32_u16(convert(a,vreinterpretq_u16_u32(sa)));
            const auto rb=vreinterpretq_u32_u16(convert(b,vreinterpretq_u16_u32(sb)));
            auto* out=raw.data()+size_t(2*(r.y+y))*w+2*(r.x+x);
            vst1q_u16(out,vreinterpretq_u16_u32(vuzp1q_u32(ra,rb)));
            vst1q_u16(out+w,vreinterpretq_u16_u32(vuzp2q_u32(ra,rb)));
            x+=4;continue;
        }
#endif
        for(int c=0;c<4;++c) {
            const float v=half_to_float(tensor.data[tensor.offset(y/pack+halo,x/pack+halo,
                ((y%pack)*pack+x%pack)*4+c)]);
            if(!std::isfinite(v))throw std::runtime_error("model output contains NaN/Inf");
            float restored=std::clamp(v,0.0f,1.0f)*range+config_.black_level;
            const int p=positions_[c];
            const size_t offset=size_t(2*(r.y+y)+p/2)*w+2*(r.x+x)+p%2;
            if(config_.edge_post) {
                const float src=(*original)[offset],delta=restored-src;
                restored=src+config_.edge_strength*std::max(0.0f,1.0f-std::abs(delta)*inverse_threshold)*delta;
            }
            raw[offset]=uint16_t(std::lround(restored));
        }
        ++x;
    }
    };
    if(executor_)executor_->run(r.height,rows);else rows(0,r.height);
}
}
