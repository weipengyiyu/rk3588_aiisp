#include "video.h"
#include "rkisp_abi.h"
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace aiisp {
int checked_ioctl(int fd, unsigned long request, void* arg, const char* op) {
    int rc;
    do { rc=ioctl(fd,request,arg); } while(rc<0 && errno==EINTR);
    if(rc<0) throw std::runtime_error(std::string(op)+": "+std::strerror(errno));
    return rc;
}
VideoQueue::VideoQueue(const std::string& path, bool output):
    type_(output?V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE:V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE),path_(path) {
    fd_=open(path.c_str(),O_RDWR|O_NONBLOCK|O_CLOEXEC);
    if(fd_<0) throw std::runtime_error("open "+path+": "+std::strerror(errno));
    try {
        v4l2_capability cap{};
        checked_ioctl(fd_,VIDIOC_QUERYCAP,&cap,"VIDIOC_QUERYCAP");
        const uint32_t caps=(cap.capabilities&V4L2_CAP_DEVICE_CAPS)?cap.device_caps:cap.capabilities;
        const uint32_t need=output?V4L2_CAP_VIDEO_OUTPUT_MPLANE:V4L2_CAP_VIDEO_CAPTURE_MPLANE;
        if(!(caps&need) || !(caps&V4L2_CAP_STREAMING))
            throw std::runtime_error(path+": wrong V4L2 queue type");
        original_.type=type_;
        checked_ioctl(fd_,VIDIOC_G_FMT,&original_,"snapshot VIDIOC_G_FMT");
    } catch(...) { close(fd_); fd_=-1; throw; }
}
VideoQueue::~VideoQueue() {
    try { restore(); } catch(const std::exception& e) {std::cerr<<"queue restore: "<<e.what()<<"\n";}
    for(auto& b:buffers_) if(b.address) munmap(b.address,b.size);
    if(fd_>=0) close(fd_);
}
void VideoQueue::set_readback_memory(int memory) {
    if(type_!=V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE || !buffers_.empty()) throw std::runtime_error("memory mode must precede allocation");
    checked_ioctl(fd_,RKISP_CMD_GET_CSI_MEMORY_MODE,&original_memory_,"snapshot CSI memory mode");
    checked_ioctl(fd_,RKISP_CMD_SET_CSI_MEMORY_MODE,&memory,"SET_CSI_MEMORY_MODE");
}
void VideoQueue::restore() {
    if(streaming_) { checked_ioctl(fd_,VIDIOC_STREAMOFF,&type_,"restore STREAMOFF"); streaming_=false; }
    const bool allocated=!buffers_.empty();
    for(auto& b:buffers_) if(b.address) {munmap(b.address,b.size); b.address=nullptr;}
    buffers_.clear();
    if(allocated) {
        v4l2_requestbuffers r{}; r.type=type_; r.memory=V4L2_MEMORY_MMAP;
        checked_ioctl(fd_,VIDIOC_REQBUFS,&r,"restore REQBUFS(0)");
    }
    if(original_memory_>=0) {
        checked_ioctl(fd_,RKISP_CMD_SET_CSI_MEMORY_MODE,&original_memory_,"restore CSI memory mode");
        original_memory_=-1;
    }
    if(restore_format_) {
        checked_ioctl(fd_,VIDIOC_S_FMT,&original_,"restore VIDIOC_S_FMT");
        restore_format_=false;
    }
}
void VideoQueue::configure(int w,int h,uint32_t fourcc,unsigned count) {
    if(!buffers_.empty()) throw std::runtime_error("queue already allocated");
    v4l2_format f{}; f.type=type_;
    f.fmt.pix_mp.width=w; f.fmt.pix_mp.height=h;
    f.fmt.pix_mp.pixelformat=fourcc; f.fmt.pix_mp.field=V4L2_FIELD_NONE;
    f.fmt.pix_mp.num_planes=1;
    restore_format_=true;
    checked_ioctl(fd_,VIDIOC_S_FMT,&f,"VIDIOC_S_FMT");
    fmt_=f.fmt.pix_mp;
    if(fmt_.width!=unsigned(w) || fmt_.height!=unsigned(h) ||
       fmt_.pixelformat!=fourcc || fmt_.num_planes!=1 || !fmt_.plane_fmt[0].sizeimage)
        throw std::runtime_error(path_+": negotiated unexpected format/dimensions/planes");
    v4l2_requestbuffers req{}; req.count=count; req.type=type_; req.memory=V4L2_MEMORY_MMAP;
    checked_ioctl(fd_,VIDIOC_REQBUFS,&req,"VIDIOC_REQBUFS");
    if(req.count<2 || req.count>32) throw std::runtime_error("invalid allocated buffer count");
    buffers_.resize(req.count);
    for(unsigned i=0;i<req.count;++i) {
        v4l2_plane p{}; v4l2_buffer b{};
        b.type=type_; b.memory=V4L2_MEMORY_MMAP; b.index=i; b.length=1; b.m.planes=&p;
        checked_ioctl(fd_,VIDIOC_QUERYBUF,&b,"VIDIOC_QUERYBUF");
        if(p.length<fmt_.plane_fmt[0].sizeimage) throw std::runtime_error("short V4L2 allocation");
        void* address=mmap(nullptr,p.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd_,p.m.mem_offset);
        if(address==MAP_FAILED) throw std::runtime_error("mmap "+path_+": "+std::strerror(errno));
        buffers_[i]={address,p.length};
    }
    std::cerr<<path_<<" "<<w<<"x"<<h<<" stride="<<fmt_.plane_fmt[0].bytesperline
             <<" size="<<fmt_.plane_fmt[0].sizeimage<<" buffers="<<req.count<<"\n";
}
uint8_t* VideoQueue::data(unsigned i) { return static_cast<uint8_t*>(buffers_.at(i).address); }
size_t VideoQueue::size(unsigned i) const { return buffers_.at(i).size; }
void VideoQueue::queue(unsigned i,size_t bytes,uint32_t seq,uint64_t ts) {
    if(bytes>size(i)) throw std::runtime_error("QBUF payload exceeds allocation");
    v4l2_plane p{}; p.length=unsigned(size(i)); p.bytesused=unsigned(bytes);
    v4l2_buffer b{}; b.index=i; b.type=type_; b.memory=V4L2_MEMORY_MMAP;
    b.length=1; b.m.planes=&p; b.sequence=seq;
    b.timestamp.tv_sec=ts/1000000000ULL; b.timestamp.tv_usec=(ts%1000000000ULL)/1000;
    checked_ioctl(fd_,VIDIOC_QBUF,&b,"VIDIOC_QBUF");
}
void VideoQueue::start() { checked_ioctl(fd_,VIDIOC_STREAMON,&type_,"VIDIOC_STREAMON"); streaming_=true; }
void VideoQueue::stop() noexcept {
    if(streaming_ && fd_>=0) { ioctl(fd_,VIDIOC_STREAMOFF,&type_); streaming_=false; }
}
bool VideoQueue::try_dequeue(Frame& f) {
    v4l2_plane p{}; v4l2_buffer b{}; b.type=type_; b.memory=V4L2_MEMORY_MMAP;
    b.length=1; b.m.planes=&p;
    int rc;
    do { rc=ioctl(fd_,VIDIOC_DQBUF,&b); } while(rc<0 && errno==EINTR);
    if(rc<0 && errno==EAGAIN) return false;
    if(rc<0) throw std::runtime_error("DQBUF "+path_+": "+std::strerror(errno));
    if(b.index>=count() || p.bytesused>size(b.index) || p.data_offset!=0 || (b.flags&V4L2_BUF_FLAG_ERROR))
        throw std::runtime_error(path_+": invalid or corrupt frame");
    f={b.index,p.bytesused,b.sequence,uint64_t(b.timestamp.tv_sec)*1000000000ULL+uint64_t(b.timestamp.tv_usec)*1000};
    return true;
}
Frame VideoQueue::dequeue(unsigned timeout_ms) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
    Frame f{};
    while(!try_dequeue(f)) {
        const auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(deadline-std::chrono::steady_clock::now()).count();
        if(ms<=0) throw std::runtime_error("frame timeout: "+path_);
        pollfd p{fd_,short(type_==V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE?POLLOUT:POLLIN),0};
        const int rc=poll(&p,1,int(ms));
        if(rc<0 && errno==EINTR) continue;
        if(rc<0) throw std::runtime_error("poll: "+std::string(std::strerror(errno)));
        if(p.revents&(POLLERR|POLLHUP|POLLNVAL)) throw std::runtime_error("V4L2 poll error: "+path_);
    }
    return f;
}
}
