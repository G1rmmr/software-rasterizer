#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

// ParallelFor is a completion boundary: its callback and captured references
// may be local values. The pool must outlive its clients.
class ParallelExecutor {
public:
    static std::size_t DefaultWorkerCount() noexcept {
        const unsigned count = std::thread::hardware_concurrency();
        return count > 1 ? count - 1 : 0;
    }

    // workerCount excludes the calling thread, which also executes work.
    explicit ParallelExecutor(std::size_t workerCount = DefaultWorkerCount()) {
        try {
            workers.reserve(workerCount);
            for(std::size_t i = 0; i < workerCount; ++i) workers.emplace_back([this] { WorkerLoop(); });
        }
        catch(...) {
            StopWorkers();
            throw;
        }
    }

    ~ParallelExecutor() {
        std::lock_guard dispatchLock(dispatchMutex);
        StopWorkers();
    }

    ParallelExecutor(const ParallelExecutor&) = delete;
    ParallelExecutor& operator=(const ParallelExecutor&) = delete;
    ParallelExecutor(ParallelExecutor&&) = delete;
    ParallelExecutor& operator=(ParallelExecutor&&) = delete;

    [[nodiscard]] std::size_t GetWorkerCount() const noexcept { return workers.size(); }

    template <typename Function>
    void ParallelFor(std::size_t begin, std::size_t end, Function&& function, std::size_t chunkSize = 64) {
        if(chunkSize == 0) throw std::invalid_argument("ParallelFor chunk size must be positive");
        if(begin >= end) return;

        // Nested work stays on its current thread, even through another pool.
        // This avoids pool exhaustion and cross-pool deadlocks.
        if(executingCallback) {
            CallbackScope scope;
            for(std::size_t i = begin; i < end; ++i) function(i);
            return;
        }

        // Concurrent callers cannot replace an active dispatch's state.
        std::lock_guard dispatchLock(dispatchMutex);
        // A single chunk can only be claimed by one thread. Execute it here
        // instead of waking every worker and waiting for their empty dispatches.
        // begin < end was checked above, so this subtraction cannot underflow.
        if(workers.empty() || end - begin <= chunkSize) {
            CallbackScope scope;
            for(std::size_t i = begin; i < end; ++i) function(i);
            return;
        }
        Dispatch dispatch;
        dispatch.Count = end - begin;
        dispatch.ChunkSize = chunkSize;
        dispatch.Callback = [&](std::size_t first, std::size_t last) {
            for(std::size_t i = first; i < last; ++i) function(begin + i);
        };
        {
            std::lock_guard stateLock(stateMutex);
            current = &dispatch;
            remainingWorkers = workers.size();
            ++generation;
        }
        startCondition.notify_all();
        Execute(dispatch);
        {
            std::unique_lock stateLock(stateMutex);
            finishCondition.wait(stateLock, [this] { return remainingWorkers == 0; });
            current = nullptr;
        }
        if(dispatch.Exception) std::rethrow_exception(dispatch.Exception);
    }

private:
    struct Dispatch {
        std::function<void(std::size_t, std::size_t)> Callback;
        std::size_t Count = 0;
        std::size_t ChunkSize = 1;
        std::atomic<std::size_t> Next{0};
        std::atomic<bool> Cancelled{false};
        std::mutex ExceptionMutex;
        std::exception_ptr Exception;
    };

    inline static thread_local bool executingCallback = false;
    struct CallbackScope {
        bool previous = executingCallback;
        CallbackScope() { executingCallback = true; }
        ~CallbackScope() { executingCallback = previous; }
    };

    std::vector<std::thread> workers;
    std::mutex dispatchMutex;
    std::mutex stateMutex;
    std::condition_variable startCondition;
    std::condition_variable finishCondition;
    Dispatch* current = nullptr;
    std::size_t remainingWorkers = 0;
    std::uint64_t generation = 0;
    bool stopping = false;

    static void Execute(Dispatch& dispatch) noexcept {
        CallbackScope scope;
        try {
            while(!dispatch.Cancelled.load(std::memory_order_relaxed)) {
                std::size_t first = dispatch.Next.load(std::memory_order_relaxed);
                std::size_t last;
                do {
                    if(first >= dispatch.Count) return;
                    last = first + std::min(dispatch.ChunkSize, dispatch.Count - first);
                } while(!dispatch.Next.compare_exchange_weak(first, last, std::memory_order_relaxed));
                dispatch.Callback(first, last);
            }
        }
        catch(...) {
            dispatch.Cancelled.store(true, std::memory_order_relaxed);
            std::lock_guard lock(dispatch.ExceptionMutex);
            if(!dispatch.Exception) dispatch.Exception = std::current_exception();
        }
    }

    void WorkerLoop() {
        std::uint64_t previousGeneration = 0;
        for(;;) {
            Dispatch* dispatch;
            {
                std::unique_lock lock(stateMutex);
                startCondition.wait(lock, [&] { return stopping || generation != previousGeneration; });
                if(stopping) return;
                previousGeneration = generation;
                dispatch = current;
            }
            Execute(*dispatch);
            {
                std::lock_guard lock(stateMutex);
                if(--remainingWorkers == 0) finishCondition.notify_one();
            }
        }
    }

    void StopWorkers() noexcept {
        {
            std::lock_guard lock(stateMutex);
            stopping = true;
        }
        startCondition.notify_all();
        for(auto& worker : workers)
            if(worker.joinable()) worker.join();
    }
};
