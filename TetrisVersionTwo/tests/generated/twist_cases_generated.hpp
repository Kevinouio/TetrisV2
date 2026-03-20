#pragma once

#include <array>
#include <string_view>
#include <vector>

namespace tetris_v2::twist_dataset {

struct TwistActive {
    char piece;
    int rotation;
    int x;
    int y;
};

struct TwistState {
    bool back_to_back;
    int combo;
    bool spin_eligible;
    bool last_rotate_used_kick;
};

struct TwistExpect {
    int lines_cleared;
    bool spin_clear;
    bool difficult_clear;
    bool b2b_out;
    bool b2b_bonus_applied;
    bool must_observe_kick;
};

struct TwistCase {
    std::string_view id;
    std::string_view family;
    std::string_view source_url;
    std::string_view source_note;
    std::array<std::string_view, 20> board_top_to_bottom;
    TwistActive active;
    TwistState state;
    std::vector<std::string_view> actions;
    TwistExpect expect;
};

inline const std::vector<TwistCase> kTwistCases{
    TwistCase{
        "T_01",
        "T",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (CW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "#.########"},
        TwistActive{"T"[0], 1, 0, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"CW", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "T_02",
        "T",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (CW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "#.########"},
        TwistActive{"T"[0], 1, 0, 14},
        TwistState{true, -1, false, false},
        std::vector<std::string_view>{"CW", "HD"},
        TwistExpect{1, true, true, true, true, true},
    },
    TwistCase{
        "T_03",
        "T",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (CCW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", ".#........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"T"[0], 2, 1, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"CCW", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "T_04",
        "T",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (CCW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", ".#........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"T"[0], 2, 1, 14},
        TwistState{true, -1, false, false},
        std::vector<std::string_view>{"CCW", "HD"},
        TwistExpect{1, true, true, true, true, true},
    },
    TwistCase{
        "T_05",
        "T",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (R180)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "#.########"},
        TwistActive{"T"[0], 1, 0, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"R180", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "T_06",
        "T",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (R180)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "#.########"},
        TwistActive{"T"[0], 1, 0, 14},
        TwistState{true, -1, false, false},
        std::vector<std::string_view>{"R180", "HD"},
        TwistExpect{1, true, true, true, true, true},
    },
    TwistCase{
        "I_01",
        "I",
        "https://harddrop.com/wiki/I-Spins_in_SRS",
        "Auto-discovered SRS kick-required spin clear fixture (CW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..........", "..#.......", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"I"[0], 0, 1, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"CW", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "I_02",
        "I",
        "https://harddrop.com/wiki/I-Spins_in_SRS",
        "Auto-discovered SRS kick-required spin clear fixture (CW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..........", "..#.......", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"I"[0], 0, 1, 14},
        TwistState{true, -1, false, false},
        std::vector<std::string_view>{"CW", "HD"},
        TwistExpect{1, true, true, true, true, true},
    },
    TwistCase{
        "I_03",
        "I",
        "https://harddrop.com/wiki/I-Spins_in_SRS",
        "Auto-discovered SRS kick-required spin clear fixture (CCW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"I"[0], 0, 1, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"CCW", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "I_04",
        "I",
        "https://harddrop.com/wiki/I-Spins_in_SRS",
        "Auto-discovered SRS kick-required spin clear fixture (CCW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"I"[0], 0, 1, 14},
        TwistState{true, -1, false, false},
        std::vector<std::string_view>{"CCW", "HD"},
        TwistExpect{1, true, true, true, true, true},
    },
    TwistCase{
        "I_05",
        "I",
        "https://harddrop.com/wiki/I-Spins_in_SRS",
        "Auto-discovered SRS kick-required spin clear fixture (R180)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..#.......", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"I"[0], 3, 0, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"R180", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "I_06",
        "I",
        "https://harddrop.com/wiki/I-Spins_in_SRS",
        "Auto-discovered SRS kick-required spin clear fixture (R180)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..#.......", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"I"[0], 3, 0, 14},
        TwistState{true, -1, false, false},
        std::vector<std::string_view>{"R180", "HD"},
        TwistExpect{1, true, true, true, true, true},
    },
    TwistCase{
        "J_01",
        "J",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (CW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..#.......", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"J"[0], 0, 1, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"CW", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "J_02",
        "J",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (CCW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..#.......", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"J"[0], 2, 1, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"CCW", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "J_03",
        "J",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (R180)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "#.........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "#.########"},
        TwistActive{"J"[0], 3, 1, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"R180", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "L_01",
        "L",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (CW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"L"[0], 1, 0, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"CW", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "L_02",
        "L",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (CCW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", ".#........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"L"[0], 3, 2, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"CCW", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "L_03",
        "L",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (R180)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", ".#........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"L"[0], 0, 1, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"R180", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "S_01",
        "S",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (CW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..#.......", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "#.########"},
        TwistActive{"S"[0], 0, 1, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"CW", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "S_02",
        "S",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (CCW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", ".#........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "#.########"},
        TwistActive{"S"[0], 2, 1, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"CCW", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "S_03",
        "S",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (R180)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "#.########"},
        TwistActive{"S"[0], 1, 0, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"R180", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "Z_01",
        "Z",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (CW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..#.......", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"Z"[0], 0, 1, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"CW", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "Z_02",
        "Z",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (CCW)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..#.......", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"Z"[0], 2, 1, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"CCW", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
    TwistCase{
        "Z_03",
        "Z",
        "https://harddrop.com/wiki/List_of_twists",
        "Auto-discovered SRS kick-required spin clear fixture (R180)",
        std::array<std::string_view, 20>{"..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", "..........", ".#########"},
        TwistActive{"Z"[0], 1, 0, 14},
        TwistState{false, -1, false, false},
        std::vector<std::string_view>{"R180", "HD"},
        TwistExpect{1, true, true, true, false, true},
    },
};

}  // namespace tetris_v2::twist_dataset
