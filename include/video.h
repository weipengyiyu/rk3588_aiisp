#pragma once
#include <linux/videodev2.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aiisp {
int checked_ioctl(int fd, unsigned long request, void* arg, const char* operation);
struct Frame { unsigned index; size_t bytes; uint32_t sequence; uint64_t timestamp_ns; };
class VideoQueue {
    int fd_=-1;
    v4l2_buf_type type_;
    bool streaming_=false;
    struct Buffer { void* address=nullptr; size_t size=0; };
    std::vector<Buffer> buffers_;
    v4l2_pix_format_mplane fmt_{};
    v4l2_format original_{};
    bool restore_format_=false;
    int original_memory_=-1;
    std::string path_;
public:
    VideoQueue(const std::string& path, bool output);
    ~VideoQueue();
    VideoQueue(const VideoQueue&)=delete;
    VideoQueue& operator=(const VideoQueue&)=delete;
    int fd() const { return fd_; }
    void configure(int width, int height, uint32_t fourcc, unsigned count);
    void set_readback_memory(int memory);
    void restore();
    const v4l2_pix_format_mplane& format() const { return fmt_; }
    unsigned count() const { return unsigned(buffers_.size()); }
    uint8_t* data(unsigned index);
    size_t size(unsigned index) const;
    void queue(unsigned index, size_t bytes=0, uint32_t sequence=0, uint64_t timestamp_ns=0);
    void start();
    void stop() noexcept;
    bool try_dequeue(Frame& frame);
    Frame dequeue(unsigned timeout_ms);
};
}
