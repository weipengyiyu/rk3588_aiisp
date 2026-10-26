#include "raw.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace aiisp {
static void validate(int w, int h, size_t stride, size_t size, Storage s) {
    if (w < 2 || h < 2 || w % 2 || h % 2 || w > 8192 || h > 8192)
        throw std::runtime_error("RAW dimensions must be even and in [2,8192]");
    const size_t row = s == Storage::Compact12 ? size_t(w) * 3 / 2 : size_t(w) * 2;
    if (stride < row || stride > size / size_t(h))
        throw std::runtime_error("RAW stride/payload too small");
}
void unpack12(const uint8_t* src, size_t size, int w, int h, size_t stride,
              Storage storage, std::vector<uint16_t>& dst) {
    validate(w, h, stride, size, storage);
    if (!src) throw std::runtime_error("null RAW input");
    dst.resize(size_t(w) * h);
    for (int y = 0; y < h; ++y) {
        const uint8_t* p = src + size_t(y) * stride;
        for (int x = 0; x < w; x += 2) {
            if (storage == Storage::Compact12) {
                dst[size_t(y)*w+x] = p[0] | ((p[1]&15) << 8);
                dst[size_t(y)*w+x+1] = (p[1] >> 4) | (p[2] << 4);
                p += 3;
            } else {
                dst[size_t(y)*w+x] = (p[0] | (p[1] << 8)) & 4095;
                dst[size_t(y)*w+x+1] = (p[2] | (p[3] << 8)) & 4095;
                p += 4;
            }
        }
    }
}
void pack12(const std::vector<uint16_t>& src, int w, int h, uint8_t* dst,
            size_t size, size_t stride, Storage storage) {
    validate(w, h, stride, size, storage);
    if (!dst || src.size() != size_t(w)*h) throw std::runtime_error("invalid RAW output");
    std::memset(dst, 0, stride * size_t(h));
    for (int y = 0; y < h; ++y) {
        uint8_t* p = dst + size_t(y)*stride;
        for (int x = 0; x < w; x += 2) {
            const unsigned a = std::min<unsigned>(src[size_t(y)*w+x], 4095);
            const unsigned b = std::min<unsigned>(src[size_t(y)*w+x+1], 4095);
            if (storage == Storage::Compact12) {
                p[0] = uint8_t(a); p[1] = uint8_t((a >> 8) | ((b & 15) << 4));
                p[2] = uint8_t(b >> 4); p += 3;
            } else {
                p[0] = uint8_t(a); p[1] = uint8_t(a >> 8);
                p[2] = uint8_t(b); p[3] = uint8_t(b >> 8); p += 4;
            }
        }
    }
}
void tiled_filter(const std::vector<uint16_t>& src, int w, int h, int t,
                  const TileFn& infer, std::vector<uint16_t>& dst) {
    if (w < 2 || h < 2 || w%2 || h%2 || src.size() != size_t(w)*h || t < 3 || t > 1024)
        throw std::runtime_error("invalid tiled Bayer contract");
    dst.resize(src.size());
    const int pw=w/2, ph=h/2, step=t-2;
    std::vector<float> in(size_t(t)*t*4), out;
    for (int by=0; by<ph; by+=step) for (int bx=0; bx<pw; bx+=step) {
        for (int y=0; y<t; ++y) for (int x=0; x<t; ++x) {
            const int sy=std::clamp(by+y-1,0,ph-1), sx=std::clamp(bx+x-1,0,pw-1);
            for (int c=0; c<4; ++c)
                in[(size_t(y)*t+x)*4+c] = src[size_t(2*sy+c/2)*w+2*sx+c%2]/4095.0f;
        }
        infer(in,out);
        if (out.size()!=in.size()) throw std::runtime_error("model output tensor size mismatch");
        for (int y=0; y<std::min(step,ph-by); ++y)
            for (int x=0; x<std::min(step,pw-bx); ++x) for (int c=0;c<4;++c) {
                const float f=out[(size_t(y+1)*t+x+1)*4+c];
                if (!std::isfinite(f)) throw std::runtime_error("model output contains NaN/Inf");
                dst[size_t(2*(by+y)+c/2)*w+2*(bx+x)+c%2] =
                    uint16_t(std::lround(std::clamp(f,0.0f,1.0f)*4095.0f));
            }
    }
}
void gaussian_tile(const std::vector<float>& in, int t, float s, std::vector<float>& out) {
    if (t<3 || in.size()!=size_t(t)*t*4 || !std::isfinite(s) || s<0 || s>1)
        throw std::runtime_error("invalid Gaussian contract");
    out.resize(in.size());
    const int k[3]={1,2,1};
    for(int y=0;y<t;++y) for(int x=0;x<t;++x) for(int c=0;c<4;++c) {
        float sum=0;
        for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx)
            sum+=k[dy+1]*k[dx+1]*in[(size_t(std::clamp(y+dy,0,t-1))*t+std::clamp(x+dx,0,t-1))*4+c]/16.0f;
        out[(size_t(y)*t+x)*4+c]=(1-s)*in[(size_t(y)*t+x)*4+c]+s*sum;
    }
}
}
