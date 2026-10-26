#include "preprocess.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace aiisp {
BayerPattern parse_bayer(const std::string& name) {
    if(name=="rggb") return BayerPattern::RGGB;
    if(name=="bggr") return BayerPattern::BGGR;
    if(name=="grbg") return BayerPattern::GRBG;
    if(name=="gbrg") return BayerPattern::GBRG;
    throw std::runtime_error("Bayer pattern must be rggb, bggr, grbg or gbrg");
}
const char* bayer_name(BayerPattern pattern) {
    switch(pattern) {
    case BayerPattern::RGGB: return "rggb";
    case BayerPattern::BGGR: return "bggr";
    case BayerPattern::GRBG: return "grbg";
    case BayerPattern::GBRG: return "gbrg";
    }
    throw std::runtime_error("invalid Bayer enum");
}
std::array<int,4> bayer_positions(BayerPattern pattern) {
    switch(pattern) {
    case BayerPattern::RGGB: return {0,1,2,3};
    case BayerPattern::BGGR: return {3,2,1,0};
    case BayerPattern::GRBG: return {1,0,3,2};
    case BayerPattern::GBRG: return {2,3,0,1};
    }
    throw std::runtime_error("invalid Bayer enum");
}
void ProcessingConfig::validate() const {
    if(tile<3 || tile>1024 || rows()<3 || rows()>1024 || halo<0 || halo>(std::min(tile,rows())-1)/2 ||
       !std::isfinite(black_level) || !std::isfinite(white_level) ||
       black_level<0 || white_level>4095 || white_level<=black_level ||
       !std::isfinite(edge_strength) || edge_strength<0 || edge_strength>1 ||
       !std::isfinite(edge_threshold) || edge_threshold<=0 || edge_threshold>4095)
        throw std::runtime_error("invalid tile/halo or RAW12 black/white levels");
    (void)bayer_positions(bayer);
}
Preprocessor::Preprocessor(ProcessingConfig config):config_(config),positions_(bayer_positions(config.bayer)) {
    config_.validate();
}
void Preprocessor::prepare(const std::vector<uint16_t>& raw,int w,int h,
                          TileRegion r,std::vector<float>& tensor) const {
    if(w<2 || h<2 || w>8192 || h>8192 || w%2 || h%2 || raw.size()!=size_t(w)*h ||
       r.x<0 || r.y<0 || r.width<1 || r.height<1 || r.x>w/2-r.width || r.y>h/2-r.height ||
       r.width>config_.tile-2*config_.halo || r.height>config_.rows()-2*config_.halo)
        throw std::runtime_error("invalid preprocessing RAW or tile region");
    const int t=config_.tile;
    const float range=config_.white_level-config_.black_level;
    tensor.resize(size_t(t)*config_.rows()*4);
    for(int y=0;y<config_.rows();++y) for(int x=0;x<t;++x) {
        const int sy=std::clamp(r.y+y-config_.halo,0,h/2-1);
        const int sx=std::clamp(r.x+x-config_.halo,0,w/2-1);
        for(int c=0;c<4;++c) {
            const int p=positions_[c];
            const float value=raw[size_t(2*sy+p/2)*w+2*sx+p%2];
            tensor[(size_t(y)*t+x)*4+c]=std::clamp((value-config_.black_level)/range,0.0f,1.0f);
        }
    }
}
void Preprocessor::prepare_native(const std::vector<uint16_t>& raw,int w,int h,
                                 TileRegion r,HalfTensor tensor) const {
    tensor.validate(config_.tile,config_.rows());
    const int t=config_.tile,halo=config_.halo;
    if(w<2 || h<2 || w%2 || h%2 || raw.size()!=size_t(w)*h ||
       r.x<0 || r.y<0 || r.width<1 || r.height<1 || r.x>w/2-r.width || r.y>h/2-r.height ||
       r.width>t-2*halo || r.height>config_.rows()-2*halo)
        throw std::runtime_error("invalid native preprocessing region");
    const float inverse=1.0f/(config_.white_level-config_.black_level);
    for(int y=0;y<config_.rows();++y) {
        const int sy=std::clamp(r.y+y-halo,0,h/2-1);
#if defined(__aarch64__)
        const auto* top=raw.data()+size_t(2*sy)*w;
        const auto* bottom=top+w;
#endif
        for(int x=0;x<t;) {
            const int sx=r.x+x-halo;
#if defined(__aarch64__)
            if(tensor.pixel_stride==4 && tensor.channel_stride==1 && sx>=0 && sx+3<w/2 && x+3<t) {
                const auto a=vld2_u16(top+2*sx),b=vld2_u16(bottom+2*sx);
                const uint16x4_t raster[4]={a.val[0],a.val[1],b.val[0],b.val[1]};
                uint16x4x4_t values;
                for(int c=0;c<4;++c) {
                    auto f=vcvtq_f32_u32(vmovl_u16(raster[positions_[c]]));
                    f=vmulq_n_f32(vsubq_f32(f,vdupq_n_f32(config_.black_level)),inverse);
                    f=vminq_f32(vmaxq_f32(f,vdupq_n_f32(0)),vdupq_n_f32(1));
                    values.val[c]=vreinterpret_u16_f16(vcvt_f16_f32(f));
                }
                vst4_u16(tensor.data+tensor.offset(y,x,0),values);
                x+=4;continue;
            }
#endif
            const int clamped=std::clamp(sx,0,w/2-1);
            for(int c=0;c<4;++c) {
                const int p=positions_[c];
                const float v=raw[size_t(2*sy+p/2)*w+2*clamped+p%2];
                tensor.data[tensor.offset(y,x,c)]=float_to_half(std::clamp((v-config_.black_level)*inverse,0.0f,1.0f));
            }
            ++x;
        }
    }
}
}
