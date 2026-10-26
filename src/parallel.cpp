#include "parallel.h"
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cstdint>
namespace aiisp {
struct RowExecutor::Impl {
    int threads,rows=0,pending=0;
    uint64_t generation=0;
    bool stopping=false;
    std::mutex mutex,submission;
    std::condition_variable start,done;
    std::function<void(int,int)> job;
    std::exception_ptr error;
    std::vector<std::thread> workers;
    explicit Impl(int n):threads(n) {
        if(n<1 || n>4)throw std::runtime_error("CPU worker count must be in [1,4]");
        try {for(int id=1;id<n;++id)workers.emplace_back([this,id]{worker(id);});}
        catch(...) {stop();throw;}
    }
    void stop() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        start.notify_all();
        for(auto& worker:workers)if(worker.joinable())worker.join();
    }
    ~Impl(){stop();}
    void execute(int id,int count,const std::function<void(int,int)>& fn) {
        try {fn(count*id/threads,count*(id+1)/threads);}
        catch(...) {std::lock_guard<std::mutex> lock(mutex);if(!error)error=std::current_exception();}
    }
    void worker(int id) {
        uint64_t seen=0;
        for(;;) {
            std::unique_lock<std::mutex> lock(mutex);
            start.wait(lock,[&]{return stopping || generation!=seen;});
            if(stopping)return;
            seen=generation;const auto* fn=&job;const int count=rows;
            lock.unlock();execute(id,count,*fn);lock.lock();
            if(--pending==0)done.notify_one();
        }
    }
    void run(int count,const std::function<void(int,int)>& fn) {
        if(count<0)throw std::runtime_error("negative row count");
        if(!count)return;
        std::lock_guard<std::mutex> serial(submission);
        if(threads==1){fn(0,count);return;}
        {std::lock_guard<std::mutex> lock(mutex);rows=count;job=fn;error=nullptr;pending=threads-1;++generation;}
        start.notify_all();execute(0,count,fn);
        std::unique_lock<std::mutex> lock(mutex);done.wait(lock,[&]{return pending==0;});
        auto failed=error;job={};lock.unlock();if(failed)std::rethrow_exception(failed);
    }
};
RowExecutor::RowExecutor(int threads):impl_(new Impl(threads)){}
RowExecutor::~RowExecutor()=default;
void RowExecutor::run(int rows,const std::function<void(int,int)>& job){impl_->run(rows,job);}
}
