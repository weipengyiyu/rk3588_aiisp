#include "isp.h"
#include "raw.h"
#include "video.h"
#include "rkisp_abi.h"
#include <chrono>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>
namespace aiisp {
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t) {return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Subdevice {
    int fd;
    explicit Subdevice(const std::string& path):fd(open(path.c_str(),O_RDWR|O_CLOEXEC)) {
        if(fd<0) throw std::runtime_error("open ISP subdevice failed: "+path);
    }
    ~Subdevice() {close(fd);}
};
}
uint32_t raw12_fourcc(BayerPattern bayer) {
    switch(bayer) {
    case BayerPattern::RGGB:return V4L2_PIX_FMT_SRGGB12;
    case BayerPattern::BGGR:return V4L2_PIX_FMT_SBGGR12;
    case BayerPattern::GRBG:return V4L2_PIX_FMT_SGRBG12;
    case BayerPattern::GBRG:return V4L2_PIX_FMT_SGBRG12;
    }
    throw std::runtime_error("invalid Bayer enum");
}
struct IspReadback::Impl {
    VideoQueue rd,yuv;
    Subdevice isp;
    int width,height;
    BayerPattern bayer;
    unsigned raw_stride=0,yuv_stride=0;
    size_t yuv_size=0;
    bool started=false;
    Impl(const std::string& rawrd,const std::string& subdev,const std::string& output,
         int w,int h,BayerPattern pattern):rd(rawrd,true),yuv(output,false),isp(subdev),width(w),height(h),bayer(pattern) {
        if(w<2 || h<2 || w>8192 || h>8192 || w%2 || h%2)
            throw std::runtime_error("invalid ISP dimensions");
        rd.set_readback_memory(CSI_MEM_WORD_LITTLE_ALIGN);
        rd.configure(w,h,raw12_fourcc(bayer),2);
        yuv.configure(w,h,V4L2_PIX_FMT_NV12,3);
        raw_stride=rd.format().plane_fmt[0].bytesperline;
        yuv_stride=yuv.format().plane_fmt[0].bytesperline;
        int memory=-1;
        checked_ioctl(rd.fd(),RKISP_CMD_GET_CSI_MEMORY_MODE,&memory,"GET_CSI_MEMORY_MODE");
        if(memory!=CSI_MEM_WORD_LITTLE_ALIGN || raw_stride<unsigned(w*2))
            throw std::runtime_error("ISP readback must use little-aligned RAW12 in 16-bit words");
        yuv_size=size_t(yuv_stride)*h*3/2;
        if(yuv_stride<unsigned(w) || yuv_size>yuv.format().plane_fmt[0].sizeimage)
            throw std::runtime_error("NV12 size/stride mismatch");
        for(unsigned i=0;i<yuv.count();++i) yuv.queue(i);
    }
};
IspReadback::IspReadback(const std::string& rawrd,const std::string& subdev,const std::string& yuv,
                       int w,int h,BayerPattern bayer):impl_(new Impl(rawrd,subdev,yuv,w,h,bayer)) {}
IspReadback::~IspReadback()=default;
unsigned IspReadback::stride() const {return impl_->yuv_stride;}
void IspReadback::restore() {impl_->yuv.restore(); impl_->rd.restore(); impl_->started=false;}
IspFrame IspReadback::submit(const ProcessedFrame& frame,unsigned timeout_ms,bool copy_nv12) {
    auto& p=*impl_;
    if(frame.width!=p.width || frame.height!=p.height || frame.bayer!=p.bayer)
        throw std::runtime_error("processed RAW geometry/Bayer does not match ISP configuration");
    IspFrame result; result.metadata=frame.metadata; result.stride=p.yuv_stride;
    auto tick=Clock::now();
    pack12(frame.raw,p.width,p.height,p.rd.data(0),p.rd.size(0),p.raw_stride,Storage::LE16);
    result.pack_ms=ms(tick);
    tick=Clock::now();
    p.rd.queue(0,size_t(p.raw_stride)*p.height,frame.metadata.sequence,frame.metadata.timestamp_ns);
    if(!p.started) {p.rd.start(); p.yuv.start(); p.started=true;}
    isp2x_csi_trigger trigger{};
    trigger.frame_id=frame.metadata.sequence;
    trigger.sof_timestamp=frame.metadata.timestamp_ns;
    trigger.frame_timestamp=frame.metadata.timestamp_ns;
    trigger.mode=T_START_X1;
    checked_ioctl(p.isp.fd,RKISP_CMD_TRIGGER_READ_BACK,&trigger,"TRIGGER_READ_BACK");
    const auto done=p.rd.dequeue(timeout_ms);
    if(done.index!=0) throw std::runtime_error("unexpected ISP readback buffer completion");
    const auto output=p.yuv.dequeue(timeout_ms);
    if(output.bytes<p.yuv_size) throw std::runtime_error("short NV12 result");
    if(output.sequence!=frame.metadata.sequence)
        throw std::runtime_error("stale/unmatched ISP frame: submitted="+std::to_string(frame.metadata.sequence)+
                                 " returned="+std::to_string(output.sequence));
    result.isp_ms=ms(tick);
    if(copy_nv12) result.nv12.assign(p.yuv.data(output.index),p.yuv.data(output.index)+p.yuv_size);
    p.yuv.queue(output.index);
    return result;
}
}
