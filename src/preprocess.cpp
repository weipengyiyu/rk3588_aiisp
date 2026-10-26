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
    if(tile<3 || tile>1024 || rows()<3 || rows()>1024 || (bayer_pack!=1 && bayer_pack!=2) || cpu_threads<1 || cpu_threads>4 || halo<0 || halo>(std::min(tile,rows())-1)/2 ||
       !std::isfinite(black_level) || !std::isfinite(white_level) ||
       black_level<0 || white_level>4095 || white_level<=black_level ||
       !std::isfinite(edge_strength) || edge_strength<0 || edge_strength>1 ||
       !std::isfinite(edge_threshold) || edge_threshold<=0 || edge_threshold>4095)
        throw std::runtime_error("invalid tile/halo or RAW12 black/white levels");
    (void)bayer_positions(bayer);
}
Preprocessor::Preprocessor(ProcessingConfig config,std::shared_ptr<RowExecutor> executor):config_(config),positions_(bayer_positions(config.bayer)) {
    config_.validate();
    executor_=executor?executor:(config.cpu_threads>1?std::make_shared<RowExecutor>(config.cpu_threads):nullptr);
}
void Preprocessor::prepare(const std::vector<uint16_t>& raw,int w,int h,
                          TileRegion r,std::vector<float>& tensor) const {
    if(config_.bayer_pack!=1)throw std::runtime_error("packed Bayer requires native FP16 preprocessing");
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
    tensor.validate(config_.tile,config_.rows(),config_.channels());
    const int t=config_.tile,halo=config_.halo,pack=config_.bayer_pack;
    if(w<2 || h<2 || w%2 || h%2 || raw.size()!=size_t(w)*h ||
       r.x<0 || r.y<0 || r.width<1 || r.height<1 || r.x>w/2-r.width || r.y>h/2-r.height ||
       r.width>(t-2*halo)*pack || r.height>(config_.rows()-2*halo)*pack)
        throw std::runtime_error("invalid native preprocessing region");
    const float inverse=1.0f/(config_.white_level-config_.black_level);
#if defined(__aarch64__)
    const auto black=vdupq_n_f32(config_.black_level),scale=vdupq_n_f32(inverse);
    const auto zero=vdupq_n_f32(0),one=vdupq_n_f32(1);
    uint8_t indices[16];
    for(int i=0;i<8;++i) {
        const int p=positions_[i%4]+(i/4)*4;
        indices[2*i]=uint8_t(2*p);indices[2*i+1]=uint8_t(2*p+1);
    }
    const auto order=vld1q_u8(indices);
    const bool identity=config_.bayer==BayerPattern::RGGB;
    const auto normalize=[&](uint16x8_t packed) {
        if(!identity)packed=vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(packed),order));
        auto a=vcvtq_f32_u32(vmovl_u16(vget_low_u16(packed)));
        auto b=vcvtq_f32_u32(vmovl_u16(vget_high_u16(packed)));
        a=vminq_f32(vmaxq_f32(vmulq_f32(vsubq_f32(a,black),scale),zero),one);
        b=vminq_f32(vmaxq_f32(vmulq_f32(vsubq_f32(b,black),scale),zero),one);
        return vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(a),vcvt_f16_f32(b)));
    };
#endif
    if(pack==2) {
        const auto rows=[&](int begin,int end) {
        for(int y=begin;y<end;++y)for(int x=0;x<t;++x) {
            const int sy=r.y+(y-halo)*2,sx=r.x+(x-halo)*2;
#if defined(__aarch64__)
            if(tensor.channel_stride==1 && (tensor.channel_block==0 || tensor.channel_block==8) &&
               sx>=0 && sy>=0 && sx+1<w/2 && sy+1<h/2) {
                const auto* first=raw.data()+size_t(2*sy)*w+2*sx;
                const auto a=vreinterpret_u32_u16(vld1_u16(first));
                const auto b=vreinterpret_u32_u16(vld1_u16(first+w));
                const auto c=vreinterpret_u32_u16(vld1_u16(first+2*w));
                const auto d=vreinterpret_u32_u16(vld1_u16(first+3*w));
                const auto ab=vzip_u32(a,b),cd=vzip_u32(c,d);
                vst1q_u16(tensor.data+tensor.offset(y,x,0),normalize(vcombine_u16(vreinterpret_u16_u32(ab.val[0]),vreinterpret_u16_u32(ab.val[1]))));
                vst1q_u16(tensor.data+tensor.offset(y,x,8),normalize(vcombine_u16(vreinterpret_u16_u32(cd.val[0]),vreinterpret_u16_u32(cd.val[1]))));
                continue;
            }
#endif
            for(int by=0;by<2;++by)for(int bx=0;bx<2;++bx)for(int c=0;c<4;++c) {
                const int p=positions_[c],yy=std::clamp(sy+by,0,h/2-1),xx=std::clamp(sx+bx,0,w/2-1);
                const float value=raw[size_t(2*yy+p/2)*w+2*xx+p%2];
                tensor.data[tensor.offset(y,x,(by*2+bx)*4+c)]=float_to_half(std::clamp((value-config_.black_level)*inverse,0.0f,1.0f));
            }
        }
        };
        if(executor_)executor_->run(config_.rows(),rows);else rows(0,config_.rows());
        return;
    }
    const auto rows=[&](int begin,int end) {
    for(int y=begin;y<end;++y) {
        const int sy=std::clamp(r.y+y-halo,0,h/2-1);
#if defined(__aarch64__)
        const auto* top=raw.data()+size_t(2*sy)*w;
        const auto* bottom=top+w;
#endif
        for(int x=0;x<t;) {
            const int sx=r.x+x-halo;
#if defined(__aarch64__)
            if(tensor.pixel_stride==4 && tensor.channel_stride==1 && sx>=0 && sx+3<w/2 && x+3<t) {
                const auto a=vreinterpretq_u32_u16(vld1q_u16(top+2*sx));
                const auto b=vreinterpretq_u32_u16(vld1q_u16(bottom+2*sx));
                auto* dst=tensor.data+tensor.offset(y,x,0);
                vst1q_u16(dst,normalize(vreinterpretq_u16_u32(vzip1q_u32(a,b))));
                vst1q_u16(dst+8,normalize(vreinterpretq_u16_u32(vzip2q_u32(a,b))));
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
    };
    if(executor_)executor_->run(config_.rows(),rows);else rows(0,config_.rows());
}
}
