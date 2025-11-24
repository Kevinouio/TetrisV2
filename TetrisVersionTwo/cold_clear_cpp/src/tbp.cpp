// TBP front-end/state machine notes (copied from cold-clear-2 Rust impl):
// - main.rs wires stdin->serde_json->FrontendMessage stream and stdout<-BotMessage sink.
//   It immediately sends BotMessage::Info { name: "Cold Clear 2", version: "<pkg> <hash>",
//   author: "MinusKelvin", features: &[] }.
// - A BotSyncronizer (sync.rs) owns the active Bot plus search stats. spawn_workers launches
//   one worker thread running work_loop().
// - Incoming message handling loop (lib.rs::run):
//   * FrontendMessage::Start(start):
//       - If start.hold is None AND start.queue is empty, stash start in waiting_on_first_piece.
//         (The bot cannot start until the first piece arrives via NewPiece.)
//       - Otherwise, call bot.start(create_bot(start, config.clone())).
//   * FrontendMessage::Stop: bot.stop(); waiting_on_first_piece = None.
//   * FrontendMessage::Suggest: if bot.suggest() returns Some, send BotMessage::Suggestion
//       { moves, move_info } back to stdout.
//   * FrontendMessage::Play { mv }: bot.advance(mv); puffin global profiler new_frame().
//   * FrontendMessage::NewPiece { piece }:
//       - If waiting_on_first_piece exists: mutate its randomizer SevenBag bag_state by
//         filling an empty bag_state to EnumSet::all() and removing the delivered piece,
//         push piece into start.queue, then start the bot with the completed Start.
//       - Otherwise: bot.new_piece(piece).
//   * FrontendMessage::Rules: respond with BotMessage::Ready.
//   * FrontendMessage::Quit: break loop. Unknown: ignore.
// - create_bot(Start, config):
//   * reserve = start.hold.unwrap_or_else(|| start.queue.remove(0)).
//   * speculate = matches!(start.randomizer, Randomizer::SevenBag { .. }).
//   * bag initialization:
//       Randomizer::Unknown => EnumSet::all();
//       Randomizer::SevenBag { bag_state } => walk start.queue in reverse; if bag_state ==
//         EnumSet::all() then clear to empty before inserting each queued piece; final bag_state
//         is used.
//   * GameState fields: reserve, back_to_back, combo (clamped via try_into.unwrap_or(255)),
//     bag (from above), board (converted via Board::from Vec<[Option<char>;10]>).
//   * Bot::new(BotOptions { speculate, config }, state, &start.queue)
// - BotSyncronizer (sync.rs):
//   * Holds Mutex<State { stats, last_advance, node_limit=u64::MAX, start, nodes_since_start }],
//     Condvar blocker, and RwLock<Option<Bot>> bot.
//   * start(initial_bot): reset stats/nodes_since_start, start=Instant::now(), set bot=Some,
//     notify_all.
//   * stop(): bot=None.
//   * suggest(): read bot RwLock, if Some -> lock state and call bot.suggest(); build MoveInfo:
//       nodes = state.stats.nodes
//       nps = nodes / state.last_advance.elapsed().as_secs_f64()
//       extra = format!("{:.1}% of selections expanded, overall speed: {:.1} Mnps",
//                       state.stats.expansions as f64 / state.stats.selections as f64 * 100.0,
//                       state.nodes_since_start as f64 / state.start.elapsed().as_secs_f64()
//                           / 1_000_000.0)
//   * advance(mv): lock state, reset stats Default, last_advance=Instant::now(); write-lock bot
//     and forward advance; notify_all.
//   * new_piece(piece): write-lock bot and forward new_piece; notify_all.
//   * work_loop(): lock state; loop:
//       - if state.stats.nodes > state.node_limit => wait(blocker) and continue
//       - read-lock bot; if None -> drop guard, wait(blocker), continue
//       - drop state lock; call bot.do_work(); drop bot guard
//       - relock state; accumulate stats with new_stats; nodes_since_start += new_stats.nodes
// - spawn_workers() launches exactly one std::thread executing work_loop().
#include "cold_clear_cpp/tbp.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "cold_clear_cpp/types.hpp"

#ifndef COLD_CLEAR_CPP_VERSION
#define COLD_CLEAR_CPP_VERSION "0.1.0"
#endif

#ifndef COLD_CLEAR_CPP_GIT_HASH
#define COLD_CLEAR_CPP_GIT_HASH "unknown"
#endif

namespace cold_clear_cpp {
namespace {

template <typename... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};
template <typename... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

std::string piece_to_string(Piece p) {
    switch (p) {
        case Piece::I:
            return "I";
        case Piece::O:
            return "O";
        case Piece::T:
            return "T";
        case Piece::L:
            return "L";
        case Piece::J:
            return "J";
        case Piece::S:
            return "S";
        case Piece::Z:
            return "Z";
    }
    throw std::logic_error("unreachable piece");
}

Piece piece_from_string(const std::string& s) {
    if (s == "I") {
        return Piece::I;
    } else if (s == "O") {
        return Piece::O;
    } else if (s == "T") {
        return Piece::T;
    } else if (s == "L") {
        return Piece::L;
    } else if (s == "J") {
        return Piece::J;
    } else if (s == "S") {
        return Piece::S;
    } else if (s == "Z") {
        return Piece::Z;
    }
    throw std::runtime_error("unknown piece: " + s);
}

std::string rotation_to_string(Rotation r) {
    switch (r) {
        case Rotation::North:
            return "north";
        case Rotation::West:
            return "west";
        case Rotation::South:
            return "south";
        case Rotation::East:
            return "east";
    }
    throw std::logic_error("unreachable rotation");
}

Rotation rotation_from_string(const std::string& s) {
    if (s == "north") {
        return Rotation::North;
    } else if (s == "west") {
        return Rotation::West;
    } else if (s == "south") {
        return Rotation::South;
    } else if (s == "east") {
        return Rotation::East;
    }
    throw std::runtime_error("unknown rotation: " + s);
}

std::string spin_to_string(Spin s) {
    switch (s) {
        case Spin::None:
            return "none";
        case Spin::Mini:
            return "mini";
        case Spin::Full:
            return "full";
    }
    throw std::logic_error("unreachable spin");
}

Spin spin_from_string(const std::string& s) {
    if (s == "none") {
        return Spin::None;
    } else if (s == "mini") {
        return Spin::Mini;
    } else if (s == "full") {
        return Spin::Full;
    }
    throw std::runtime_error("unknown spin: " + s);
}

PieceLocation location_from_json(const nlohmann::json& j) {
    PieceLocation loc;
    loc.piece = piece_from_string(j.at("type").get<std::string>());
    loc.rotation = rotation_from_string(j.at("orientation").get<std::string>());
    loc.x = j.at("x").get<std::int8_t>();
    loc.y = j.at("y").get<std::int8_t>();
    return loc;
}

nlohmann::json location_to_json(const PieceLocation& loc) {
    return nlohmann::json{
        {"type", piece_to_string(loc.piece)},
        {"orientation", rotation_to_string(loc.rotation)},
        {"x", loc.x},
        {"y", loc.y},
    };
}

Placement placement_from_json(const nlohmann::json& j) {
    Placement placement;
    placement.location = location_from_json(j.at("location"));
    placement.spin = spin_from_string(j.at("spin").get<std::string>());
    return placement;
}

nlohmann::json placement_to_json(const Placement& placement) {
    return nlohmann::json{
        {"location", location_to_json(placement.location)},
        {"spin", spin_to_string(placement.spin)},
    };
}

PieceSet parse_piece_set(const nlohmann::json& arr) {
    PieceSet set;
    for (const auto& entry : arr) {
        auto piece = piece_from_string(entry.get<std::string>());
        set.set(piece_index(piece));
    }
    return set;
}

std::vector<Piece> parse_piece_array(const nlohmann::json& arr) {
    std::vector<Piece> pieces;
    pieces.reserve(arr.size());
    for (const auto& entry : arr) {
        pieces.push_back(piece_from_string(entry.get<std::string>()));
    }
    return pieces;
}

Board parse_board(const nlohmann::json& board_json) {
    Board board;
    for (std::size_t x = 0; x < 10; ++x) {
        for (std::size_t y = 0; y < 40; ++y) {
            const auto& cell = board_json.at(y).at(x);
            if (cell.is_null()) {
                continue;
            }
            // Force the same type expectations as Option<char> by attempting to parse a string.
            cell.get<std::string>();
            board.cols[x] |= 1ull << y;
        }
    }
    return board;
}

Randomizer parse_randomizer(const nlohmann::json& j) {
    Randomizer randomizer;
    auto type = j.at("type").get<std::string>();
    if (type == "seven_bag") {
        randomizer.kind = Randomizer::Kind::SevenBag;
        randomizer.bag_state = parse_piece_set(j.at("bag_state"));
    } else {
        randomizer.kind = Randomizer::Kind::Unknown;
    }
    return randomizer;
}

StartMessage parse_start(const nlohmann::json& j) {
    StartMessage start;
    start.board = parse_board(j.at("board"));
    start.queue = parse_piece_array(j.at("queue"));
    if (j.contains("hold") && !j.at("hold").is_null()) {
        start.hold = piece_from_string(j.at("hold").get<std::string>());
    }
    start.combo = j.at("combo").get<std::uint32_t>();
    start.back_to_back = j.at("back_to_back").get<bool>();
    if (j.contains("randomizer")) {
        start.randomizer = parse_randomizer(j.at("randomizer"));
    }
    return start;
}

std::string version_string() {
    return std::string{COLD_CLEAR_CPP_VERSION} + " " + COLD_CLEAR_CPP_GIT_HASH;
}

std::string format_extra(
    const Statistics& stats, std::uint64_t nodes_since_start, std::chrono::steady_clock::time_point start_time) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    double ratio = static_cast<double>(stats.expansions) / static_cast<double>(stats.selections);
    double overall = static_cast<double>(nodes_since_start) /
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() /
                     1'000'000.0;
    oss << ratio * 100.0 << "% of selections expanded, overall speed: " << overall << " Mnps";
    return oss.str();
}

}  // namespace

BotSynchronizer::BotSynchronizer() {
    auto now = std::chrono::steady_clock::now();
    state_.last_advance = now;
    state_.start = now;
}

void BotSynchronizer::start(Bot bot) {
    {
        std::unique_lock<std::mutex> state_lock(state_mutex_);
        state_.stats = Statistics{};
        state_.nodes_since_start = 0;
        state_.start = std::chrono::steady_clock::now();
        std::unique_lock<std::shared_mutex> bot_lock(bot_mutex_);
        bot_ = std::move(bot);
    }
    blocker_.notify_all();
}

void BotSynchronizer::stop() {
    std::unique_lock<std::shared_mutex> bot_lock(bot_mutex_);
    bot_.reset();
}

std::optional<std::pair<std::vector<Placement>, MoveInfo>> BotSynchronizer::suggest() {
    std::shared_lock<std::shared_mutex> bot_lock(bot_mutex_);
    if (!bot_) {
        return std::nullopt;
    }

    std::unique_lock<std::mutex> state_lock(state_mutex_);
    auto suggestion = bot_->suggest();
    MoveInfo info;
    info.nodes = state_.stats.nodes;
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - state_.last_advance).count();
    info.nps = elapsed == 0.0 ? 0.0 : static_cast<double>(info.nodes) / elapsed;
    info.extra = format_extra(state_.stats, state_.nodes_since_start, state_.start);
    return std::make_pair(std::move(suggestion), info);
}

void BotSynchronizer::advance(const Placement& mv) {
    std::unique_lock<std::mutex> state_lock(state_mutex_);
    state_.stats = Statistics{};
    state_.last_advance = std::chrono::steady_clock::now();
    std::unique_lock<std::shared_mutex> bot_lock(bot_mutex_);
    if (bot_) {
        bot_->advance(mv);
    }
    blocker_.notify_all();
}

void BotSynchronizer::new_piece(Piece piece) {
    std::unique_lock<std::shared_mutex> bot_lock(bot_mutex_);
    if (bot_) {
        bot_->new_piece(piece);
    }
    blocker_.notify_all();
}

void BotSynchronizer::work_loop() {
    std::unique_lock<std::mutex> state_lock(state_mutex_);
    while (true) {
        if (state_.stats.nodes > state_.node_limit) {
            blocker_.wait(state_lock);
            continue;
        }

        std::shared_lock<std::shared_mutex> bot_lock(bot_mutex_);
        if (!bot_) {
            bot_lock.unlock();
            blocker_.wait(state_lock);
            continue;
        }

        Bot* bot = &bot_.value();
        state_lock.unlock();
        auto new_stats = bot->do_work();
        bot_lock.unlock();

        state_lock.lock();
        state_.stats.accumulate(new_stats);
        state_.nodes_since_start += new_stats.nodes;
    }
}

FrontendMessage parse_frontend_message(const std::string& line) {
    auto j = nlohmann::json::parse(line);
    auto type = j.at("type").get<std::string>();
    if (type == "rules") {
        return RulesMessage{};
    }
    if (type == "start") {
        return parse_start(j);
    }
    if (type == "play") {
        return PlayMessage{placement_from_json(j.at("move"))};
    }
    if (type == "new_piece") {
        return NewPieceMessage{piece_from_string(j.at("piece").get<std::string>())};
    }
    if (type == "suggest") {
        return SuggestMessage{};
    }
    if (type == "stop") {
        return StopMessage{};
    }
    if (type == "quit") {
        return QuitMessage{};
    }
    return UnknownMessage{};
}

nlohmann::json bot_message_to_json(const BotMessage& msg) {
    return std::visit(
        Overloaded{
            [](const InfoMessage& info) {
                return nlohmann::json{
                    {"type", "info"},
                    {"name", info.name},
                    {"version", info.version},
                    {"author", info.author},
                    {"features", info.features},
                };
            },
            [](const ReadyMessage&) { return nlohmann::json{{"type", "ready"}}; },
            [](const SuggestionMessage& suggestion) {
                nlohmann::json moves = nlohmann::json::array();
                for (const auto& mv : suggestion.moves) {
                    moves.push_back(placement_to_json(mv));
                }
                return nlohmann::json{
                    {"type", "suggestion"},
                    {"moves", moves},
                    {"move_info",
                     {{"nodes", suggestion.move_info.nodes},
                      {"nps", suggestion.move_info.nps},
                      {"extra", suggestion.move_info.extra}}},
                };
            },
        },
        msg);
}

Bot create_bot_from_start(StartMessage start, const std::shared_ptr<BotConfig>& config) {
    Piece reserve;
    if (start.hold.has_value()) {
        reserve = *start.hold;
    } else {
        if (start.queue.empty()) {
            throw std::runtime_error("start queue empty when hold is none");
        }
        reserve = start.queue.front();
        start.queue.erase(start.queue.begin());
    }

    bool speculate = start.randomizer.kind == Randomizer::Kind::SevenBag;
    PieceSet bag = all_pieces();
    if (start.randomizer.kind == Randomizer::Kind::SevenBag) {
        bag = start.randomizer.bag_state;
        for (auto it = start.queue.rbegin(); it != start.queue.rend(); ++it) {
            if (bag == all_pieces()) {
                bag.reset();
            }
            bag.set(piece_index(*it));
        }
    }

    GameState state;
    state.reserve = reserve;
    state.back_to_back = start.back_to_back;
    state.combo = start.combo > 255 ? 255 : static_cast<std::uint8_t>(start.combo);
    state.bag = bag;
    state.board = start.board;

    BotOptions options;
    options.speculate = speculate;
    options.config = config;
    return Bot(options, state, start.queue);
}

void spawn_workers(const std::shared_ptr<BotSynchronizer>& bot, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        auto cloned = bot;
        std::thread([cloned]() { cloned->work_loop(); }).detach();
    }
}

void run_tbp(std::istream& in, std::ostream& out, const std::shared_ptr<BotConfig>& config) {
    BotMessage info = InfoMessage{
        "Cold Clear 2",
        version_string(),
        "MinusKelvin",
        {},
    };
    out << bot_message_to_json(info).dump() << '\n';
    out.flush();

    auto bot = std::make_shared<BotSynchronizer>();
    spawn_workers(bot, 1);

    std::optional<StartMessage> waiting_on_first_piece;
    std::string line;
    while (std::getline(in, line)) {
        auto msg = parse_frontend_message(line);
        bool quit = false;
        std::visit(
            Overloaded{
                [&](const StartMessage& start) {
                    if (!start.hold.has_value() && start.queue.empty()) {
                        waiting_on_first_piece = start;
                    } else {
                        bot->start(create_bot_from_start(start, config));
                    }
                },
                [&](const StopMessage&) {
                    bot->stop();
                    waiting_on_first_piece.reset();
                },
                [&](const SuggestMessage&) {
                    auto suggestion = bot->suggest();
                    if (suggestion) {
                        BotMessage outgoing = SuggestionMessage{
                            std::move(suggestion->first),
                            suggestion->second,
                        };
                        out << bot_message_to_json(outgoing).dump() << '\n';
                        out.flush();
                    }
                },
                [&](const PlayMessage& play) {
                    bot->advance(play.mv);
#ifdef PROFILE_COLD_CLEAR
                    // Placeholder hook for a profiler frame boundary.
#endif
                },
                [&](const NewPieceMessage& np) {
                    if (waiting_on_first_piece.has_value()) {
                        auto start = *waiting_on_first_piece;
                        waiting_on_first_piece.reset();
                        if (start.randomizer.kind == Randomizer::Kind::SevenBag) {
                            if (start.randomizer.bag_state.none()) {
                                start.randomizer.bag_state = all_pieces();
                            }
                            start.randomizer.bag_state.reset(piece_index(np.piece));
                        }
                        start.queue.push_back(np.piece);
                        bot->start(create_bot_from_start(start, config));
                    } else {
                        bot->new_piece(np.piece);
                    }
                },
                [&](const RulesMessage&) {
                    BotMessage ready = ReadyMessage{};
                    out << bot_message_to_json(ready).dump() << '\n';
                    out.flush();
                },
                [&](const QuitMessage&) { quit = true; },
                [&](const UnknownMessage&) {},
            },
            msg);
        if (quit) {
            break;
        }
    }
}

}  // namespace cold_clear_cpp
