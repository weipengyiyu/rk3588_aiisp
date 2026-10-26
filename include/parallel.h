#pragma once
#include <functional>
#include <memory>
namespace aiisp {
// Synchronous row jobs backed by persistent workers; errors propagate to caller.
class RowExecutor {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit RowExecutor(int threads);
    ~RowExecutor();
    RowExecutor(const RowExecutor&)=delete;
    RowExecutor& operator=(const RowExecutor&)=delete;
    void run(int rows,const std::function<void(int,int)>& job);
};
}
