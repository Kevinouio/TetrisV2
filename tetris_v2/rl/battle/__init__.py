"""Battle learning primitives; the environment remains a separate dependency."""

from tetris_v2.rl.battle.checkpoint import (
    BattleCheckpointPaths,
    BattleTrainingBundle,
    load_battle_training_checkpoint,
    save_battle_training_checkpoint,
)
from tetris_v2.rl.battle.dqn import (
    BATTLE_ACTION_ORDER,
    BATTLE_FEATURE_NAMES,
    BATTLE_OBSERVATION_DIM,
    BATTLE_OBSERVATION_SCHEMA,
    BattleDQNAgent,
    BattleDQNConfig,
    BattlePlacementQNet,
    LinearSchedule,
)
from tetris_v2.rl.battle.config import BattleRewardConfig, BattleRulesConfig
from tetris_v2.rl.battle.curriculum import (
    CurriculumStage,
    CurriculumState,
    PromotionRequirement,
    default_curriculum,
)
from tetris_v2.rl.battle.env import (
    BATTLE_FEATURE_SLICE,
    OPP_BOARD_SLICE,
    OWN_OBS_SLICE,
    BattleEnv,
)
from tetris_v2.rl.battle.opponents import (
    OpponentDescriptor,
    OpponentPool,
    OpponentSelection,
)
from tetris_v2.rl.battle.evaluation import (
    ScheduledBattle,
    ScriptedGarbagePressure,
    compare_repeated_matches,
    evaluate_paired_battles,
    paired_seat_schedule,
    run_battle_match,
)
from tetris_v2.rl.battle.metrics import (
    BattleMatchMetrics,
    evaluate_battle_gate,
    summarize_battle_matches,
    win_matrix,
)
from tetris_v2.rl.battle.policies import (
    BattleDQNPolicy,
    ColdClearBattlePolicy,
    RandomBattlePolicy,
    load_battle_dqn_policy,
)
from tetris_v2.rl.battle.replay import PackedBattleReplayBuffer
from tetris_v2.rl.battle.reward import BattleReward, compute_battle_rewards
from tetris_v2.rl.battle.runtime import BattleRuntime
from tetris_v2.rl.battle.stats import (
    BattleStats,
    BoardStats,
    PlayerBattleStats,
    PlayerStepStats,
    compute_board_stats,
)


__all__ = [
    "BATTLE_ACTION_ORDER",
    "BATTLE_FEATURE_SLICE",
    "BATTLE_FEATURE_NAMES",
    "BATTLE_OBSERVATION_DIM",
    "BATTLE_OBSERVATION_SCHEMA",
    "OPP_BOARD_SLICE",
    "OWN_OBS_SLICE",
    "BattleEnv",
    "BattleDQNPolicy",
    "BattleMatchMetrics",
    "BattleReward",
    "BattleRewardConfig",
    "BattleRulesConfig",
    "BattleRuntime",
    "BattleStats",
    "BattleCheckpointPaths",
    "BattleDQNAgent",
    "BattleDQNConfig",
    "BattlePlacementQNet",
    "BattleTrainingBundle",
    "BoardStats",
    "ColdClearBattlePolicy",
    "CurriculumStage",
    "CurriculumState",
    "LinearSchedule",
    "OpponentDescriptor",
    "OpponentPool",
    "OpponentSelection",
    "PackedBattleReplayBuffer",
    "PlayerBattleStats",
    "PlayerStepStats",
    "PromotionRequirement",
    "RandomBattlePolicy",
    "ScheduledBattle",
    "ScriptedGarbagePressure",
    "compare_repeated_matches",
    "compute_battle_rewards",
    "compute_board_stats",
    "default_curriculum",
    "evaluate_battle_gate",
    "evaluate_paired_battles",
    "load_battle_training_checkpoint",
    "load_battle_dqn_policy",
    "paired_seat_schedule",
    "run_battle_match",
    "save_battle_training_checkpoint",
    "summarize_battle_matches",
    "win_matrix",
]
