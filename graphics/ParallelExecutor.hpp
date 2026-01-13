#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Wait-Free Atomic Counter
class ParallelExecutor {
public:
    static ParallelExecutor& GetInstance() {
        static ParallelExecutor executor(std::thread::hardware_concurrency() - 1);
        return executor;
    }

    ParallelExecutor() = delete;

    ParallelExecutor(const ParallelExecutor& other) = delete;
    ParallelExecutor& operator=(const ParallelExecutor& other) = delete;

    ParallelExecutor(ParallelExecutor&& other) = delete;
    ParallelExecutor& operator=(ParallelExecutor&& other) = delete;

    template <typename T>
    void ParallelFor(const std::size_t begin, const std::size_t end, T&& func, const std::size_t chunkSize = 64) {
        if(begin >= end) return;

        {
            std::unique_lock<std::mutex> lock(mutex);
            currentTask = [&](std::size_t start, std::size_t end) {
                for(std::size_t i = start; i < end; ++i) func(i);
            };

            globalIndex.store(begin);
            globalEnd = end;
            globalChunkSize = chunkSize;
            finishedThreadCount.store(0);

            ++currTaskID;
        }

        conditionStart.notify_all();
        workerTask();

        std::unique_lock<std::mutex> lock(mutex);
        conditionFinish.wait(lock, [this] { return finishedThreadCount.load() == workers.size(); });
    }

private:
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable conditionStart;
    std::condition_variable conditionFinish;

    std::function<void(std::size_t, std::size_t)> currentTask;
    std::atomic<std::size_t> globalIndex;
    std::size_t globalEnd;
    std::size_t globalChunkSize;
    std::atomic<std::size_t> finishedThreadCount;

    std::uint16_t currTaskID = 0;
    bool stop;

    ParallelExecutor(std::size_t threadCount) : stop(false), finishedThreadCount() {
        for(std::size_t i = 0; i < threadCount; ++i) workers.emplace_back([this] { this->workerLoop(); });
    }

    ~ParallelExecutor() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            stop = true;
        }

        conditionStart.notify_all();

        for(auto& worker : workers) {
            if(worker.joinable()) worker.join();
        }
    }

    void workerLoop() {
        std::uint16_t lastTaskID = 0;

        while(true) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                conditionStart.wait(lock, [this, &lastTaskID] { return stop || (lastTaskID != currTaskID); });

                if(stop) return;
                lastTaskID = currTaskID;
            }

            workerTask();

            if(finishedThreadCount.fetch_add(1) + 1 == workers.size()) {
                std::unique_lock<std::mutex> lock(mutex);
                conditionFinish.notify_one();
            }
        }
    }

    void workerTask() {
        while(true) {
            std::size_t start = globalIndex.fetch_add(globalChunkSize, std::memory_order_relaxed);
            if(start >= globalEnd) break;

            std::size_t end = std::min(start + globalChunkSize, globalEnd);
            currentTask(start, end);
        }
    }
};