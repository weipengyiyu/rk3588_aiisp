#include "postprocess.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace aiisp {
Postprocessor::Postprocessor(ProcessingConfig config):config_(config),positions_(bayer_positions(config.bayer)) {
    config_.validate();
}
void Postprocessor::merge(const std::vector<float>& tensor,TileRegion r,int w,int h,
                          std::vector<uint16_t>& raw) const {
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
    const int t=config_.tile,halo=config_.halo;
    tensor.validate(t,config_.rows());
    if(config_.edge_post && (!original || original->size()!=raw.size()))
        throw std::runtime_error("edge post needs the matching original RAW frame");
    if(w<2 || h<2 || w%2 || h%2 || raw.size()!=size_t(w)*h ||
       r.x<0 || r.y<0 || r.width<1 || r.height<1 || r.x>w/2-r.width || r.y>h/2-r.height ||
       r.width>t-2*halo || r.height>config_.rows()-2*halo)
        throw std::runtime_error("invalid native postprocessing region");
    const float range=config_.white_level-config_.black_level;
    const float inverse_threshold=1.0f/config_.edge_threshold;
    for(int y=0;y<r.height;++y) for(int x=0;x<r.width;) {
#if defined(__aarch64__)
        if(tensor.channel_stride==1 && x+3<r.width) {
            const auto* base=tensor.data+tensor.offset(y+halo,x+halo,0);
            const auto a=vld1_u16(base),b=vld1_u16(base+tensor.pixel_stride);
            const auto c=vld1_u16(base+2*tensor.pixel_stride),d=vld1_u16(base+3*tensor.pixel_stride);
            const auto ab=vzip_u16(a,b),cd=vzip_u16(c,d);
            const auto lo=vzip_u32(vreinterpret_u32_u16(ab.val[0]),vreinterpret_u32_u16(cd.val[0]));
            const auto hi=vzip_u32(vreinterpret_u32_u16(ab.val[1]),vreinterpret_u32_u16(cd.val[1]));
            const uint16x4_t planes[4]={vreinterpret_u16_u32(lo.val[0]),vreinterpret_u16_u32(lo.val[1]),
                                       vreinterpret_u16_u32(hi.val[0]),vreinterpret_u16_u32(hi.val[1])};
            uint16x4_t raster[4];
            uint16x4_t source[4];
            if(config_.edge_post) {
                const auto* src=original->data()+size_t(2*(r.y+y))*w+2*(r.x+x);
                const auto top=vld2_u16(src),bottom=vld2_u16(src+w);
                source[0]=top.val[0];source[1]=top.val[1];source[2]=bottom.val[0];source[3]=bottom.val[1];
            }
            for(int cidx=0;cidx<4;++cidx) {
                auto value=vcvt_f32_f16(vreinterpret_f16_u16(planes[cidx]));
                if(vminvq_u32(vcleq_f32(vabsq_f32(value),vdupq_n_f32(65504)))==0)
                    throw std::runtime_error("model output contains NaN/Inf");
                value=vminq_f32(vmaxq_f32(value,vdupq_n_f32(0)),vdupq_n_f32(1));
                value=vaddq_f32(vmulq_n_f32(value,range),vdupq_n_f32(config_.black_level));
                if(config_.edge_post) {
                    const auto src=vcvtq_f32_u32(vmovl_u16(source[positions_[cidx]]));
                    const auto delta=vsubq_f32(value,src);
                    const auto gate=vmaxq_f32(vsubq_f32(vdupq_n_f32(1),vmulq_n_f32(vabsq_f32(delta),inverse_threshold)),vdupq_n_f32(0));
                    value=vaddq_f32(src,vmulq_n_f32(vmulq_f32(delta,gate),config_.edge_strength));
                }
                value=vaddq_f32(value,vdupq_n_f32(0.5f));
                raster[positions_[cidx]]=vmovn_u32(vcvtq_u32_f32(value));
            }
            uint16x4x2_t top{{raster[0],raster[1]}},bottom{{raster[2],raster[3]}};
            auto* out=raw.data()+size_t(2*(r.y+y))*w+2*(r.x+x);
            vst2_u16(out,top);vst2_u16(out+w,bottom);
            x+=4;continue;
        }
#endif
        for(int c=0;c<4;++c) {
            const float v=half_to_float(tensor.data[tensor.offset(y+halo,x+halo,c)]);
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
}
}
