#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace aiisp {
// CPU-visible FP16 tensor; strides are in uint16_t elements, not bytes.
struct HalfTensor {
    uint16_t* data=nullptr;
    size_t elements=0,row_stride=0,pixel_stride=0,channel_stride=0;
    int width=0,height=0;
    size_t offset(int y,int x,int c) const {
        return size_t(y)*row_stride+size_t(x)*pixel_stride+size_t(c)*channel_stride;
    }
    void validate(int tile,int rows=0) const {
        if(!rows)rows=tile;
        if(!data || width!=tile || height!=rows || !row_stride || !pixel_stride || !channel_stride ||
           offset(rows-1,tile-1,3)>=elements) throw std::runtime_error("invalid native FP16 tensor view");
    }
};
inline uint16_t float_to_half(float x) {
#if defined(__aarch64__)
    const __fp16 h=static_cast<__fp16>(x); uint16_t bits; std::memcpy(&bits,&h,2); return bits;
#else
    uint32_t u;std::memcpy(&u,&x,4);
    const uint32_t sign=(u>>16)&0x8000,abs=u&0x7fffffff;
    if(abs>=0x7f800000)return uint16_t(sign|0x7c00|((abs&0x7fffff)?0x200:0));
    int e=int((u>>23)&255)-127+15; uint32_t m=u&0x7fffff;
    if(e>=31)return uint16_t(sign|0x7c00);
    if(e<=0) {
        if(e < -10)return uint16_t(sign);
        m|=0x800000;const unsigned shift=unsigned(14-e);
        const uint32_t v=(m+((1u<<(shift-1))-1)+((m>>shift)&1))>>shift;
        return uint16_t(sign|v);
    }
    m+=0xfff+((m>>13)&1);
    if(m&0x800000){m=0;++e;}
    return uint16_t(sign|(unsigned(e)<<10)|(m>>13));
#endif
}
inline float half_to_float(uint16_t h) {
#if defined(__aarch64__)
    __fp16 value;std::memcpy(&value,&h,2);return float(value);
#else
    uint32_t sign=uint32_t(h&0x8000)<<16,m=h&1023,u;
    int e=(h>>10)&31;
    if(!e) {
        if(!m)u=sign;
        else {e=1;while(!(m&1024)){m<<=1;--e;}u=sign|(uint32_t(e+112)<<23)|((m&1023)<<13);}
    } else u=sign|(uint32_t(e==31?255:e+112)<<23)|(m<<13);
    float x;std::memcpy(&x,&u,4);return x;
#endif
}
}
