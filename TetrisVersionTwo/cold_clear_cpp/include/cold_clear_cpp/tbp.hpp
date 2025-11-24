#pragma once

#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <istream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <shared_mutex>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "cold_clear_cpp/bot.hpp"
#include "cold_clear_cpp/state.hpp"
#include "nlohmann/json.hpp"

namespace cold_clear_cpp {

struct MoveInfo {
    std::uint64_t nodes{0};
    double nps{0.0};
    std::string extra;
};

struct Randomizer {
    enum class Kind { Unknown, SevenBag };

    Kind kind{Kind::Unknown};
    PieceSet bag_state{};
};

struct StartMessage {
    Board board{};
    std::vector<Piece> queue;
    std::optional<Piece> hold;
    std::uint32_t combo{0};
    bool back_to_back{false};
    Randomizer randomizer{};
};

struct PlayMessage {
    Placement mv{};
};

struct NewPieceMessage {
    Piece piece{Piece::I};
};

struct SuggestMessage {};
struct StopMessage {};
struct QuitMessage {};
struct RulesMessage {};
struct UnknownMessage {};

using FrontendMessage =
    std::variant<RulesMessage, StartMessage, PlayMessage, NewPieceMessage, SuggestMessage, StopMessage, QuitMessage,
                 UnknownMessage>;

struct InfoMessage {
    std::string name;
    std::string version;
    std::string author;
    std::vector<std::string> features;
};

struct ReadyMessage {};

struct SuggestionMessage {
    std::vector<Placement> moves;
    MoveInfo move_info;
};

using BotMessage = std::variant<InfoMessage, ReadyMessage, SuggestionMessage>;

class BotSynchronizer {
public:
    BotSynchronizer();

    void start(Bot bot);
    void stop();
    std::optional<std::pair<std::vector<Placement>, MoveInfo>> suggest();
    void advance(const Placement& mv);
    void new_piece(Piece piece);
    void work_loop();

private:
    struct State {
        Statistics stats{};
        std::chrono::steady_clock::time_point last_advance{};
        std::uint64_t node_limit{std::numeric_limits<std::uint64_t>::max()};
        std::chrono::steady_clock::time_point start{};
        std::uint64_t nodes_since_start{0};
    };

    std::mutex state_mutex_;
    std::condition_variable blocker_;
    std::shared_mutex bot_mutex_;
    std::optional<Bot> bot_;
    State state_;
};

FrontendMessage parse_frontend_message(const std::string& line);
nlohmann::json bot_message_to_json(const BotMessage& msg);
Bot create_bot_from_start(StartMessage start, const std::shared_ptr<BotConfig>& config);
void spawn_workers(const std::shared_ptr<BotSynchronizer>& bot, std::size_t count);
void run_tbp(std::istream& in, std::ostream& out, const std::shared_ptr<BotConfig>& config);

}  // namespace cold_clear_cpp
