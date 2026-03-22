#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "tetris_v2/cc2_dag.hpp"
#include "tetris_v2/cc2_eval.hpp"

namespace tetris_v2::cc2 {

struct SyncSnapshot {
    DagSuggestion suggestion{};
    DagStatistics stats{};
    double nps{0.0};
};

class Synchronizer {
public:
    Synchronizer();
    ~Synchronizer();

    void set_weights(const FreestyleWeights& weights);
    void set_exploitation(double exploitation);

    void start(GameState root, const std::deque<Piece>& queue, bool speculate);
    void stop();

    void wait_until(std::chrono::steady_clock::time_point deadline);
    SyncSnapshot snapshot() const;
    DagStatistics run_one_chunk();

private:
    void worker_loop();

    mutable std::mutex mutex_{};
    mutable std::condition_variable cv_{};
    std::thread worker_{};

    bool shutdown_{false};
    bool active_{false};

    Dag dag_{};
    FreestyleWeights weights_{};
    double exploitation_{0.6931471805599453};  // ln(2)
    DagStatistics stats_{};
    DagSuggestion latest_suggestion_{};
    std::chrono::steady_clock::time_point search_started_{};
};

}  // namespace tetris_v2::cc2
