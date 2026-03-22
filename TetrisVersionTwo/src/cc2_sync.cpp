#include "tetris_v2/cc2_sync.hpp"

namespace tetris_v2::cc2 {

Synchronizer::Synchronizer() : worker_([this] { worker_loop(); }) {}

Synchronizer::~Synchronizer() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        active_ = false;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void Synchronizer::set_weights(const FreestyleWeights& weights) {
    std::lock_guard<std::mutex> lock(mutex_);
    weights_ = weights;
}

void Synchronizer::set_exploitation(double exploitation) {
    std::lock_guard<std::mutex> lock(mutex_);
    exploitation_ = exploitation;
}

void Synchronizer::start(GameState root, const std::deque<Piece>& queue, bool speculate) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        dag_.reset(root, queue, speculate);
        stats_ = DagStatistics{};
        latest_suggestion_ = DagSuggestion{};
        search_started_ = std::chrono::steady_clock::now();
        active_ = true;
    }
    cv_.notify_all();
}

void Synchronizer::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_ = false;
    }
    cv_.notify_all();
}

void Synchronizer::wait_until(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (active_ && !shutdown_) {
        if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            break;
        }
    }
}

SyncSnapshot Synchronizer::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    SyncSnapshot out{};
    out.stats = stats_;
    out.suggestion = latest_suggestion_;

    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - search_started_).count();
    if (elapsed > 0.0) {
        out.nps = static_cast<double>(out.stats.nodes) / elapsed;
    }
    return out;
}

DagStatistics Synchronizer::run_one_chunk() {
    FreestyleWeights weights{};
    double exploitation = 0.0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_ || shutdown_) {
            return {};
        }
        weights = weights_;
        exploitation = exploitation_;
    }

    const auto delta = dag_.do_work(weights, exploitation);
    const auto suggestion = dag_.suggest();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_ || shutdown_) {
            return {};
        }
        stats_.accumulate(delta);
        if (suggestion.valid) {
            latest_suggestion_ = suggestion;
        }
        cv_.notify_all();
    }
    return delta;
}

void Synchronizer::worker_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        cv_.wait(lock, [this] { return shutdown_ || active_; });
        if (shutdown_) {
            return;
        }

        const auto weights = weights_;
        const auto exploitation = exploitation_;
        lock.unlock();

        const auto delta = dag_.do_work(weights, exploitation);
        const auto suggestion = dag_.suggest();

        lock.lock();
        if (!active_ || shutdown_) {
            continue;
        }
        stats_.accumulate(delta);
        if (suggestion.valid) {
            latest_suggestion_ = suggestion;
        }
        cv_.notify_all();
        std::this_thread::yield();
    }
}

}  // namespace tetris_v2::cc2
