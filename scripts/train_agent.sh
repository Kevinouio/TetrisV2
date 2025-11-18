#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_agent.sh [options] [-- extra-flags]

General options:
  --algo {ppo|dqn}          Algorithm to train (default: ppo)
  --env {nes|modern|custom} Environment choice (default: nes)
  --env-id ID               Custom Gymnasium id when --env=custom
  --total-steps N           Total timesteps to train (default: 1500000)
  --log-dir PATH            Output directory for checkpoints/logs
  --resume-from PATH        Path to checkpoint to resume from
  --device DEV              Torch device override, e.g. cuda:0 or cpu
  --seed N                  RNG seed (default: 123)
  --num-envs N              Parallel env workers (default: 8)
  --reward-weight KEY=VAL   Override AdvancedRewardConfig field (repeatable)

PPO-specific defaults (overridable):
  --n-steps N               Rollout horizon per update (default: 4096)
  --minibatch-size N        SGD minibatch size (default: 2048)

DQN-specific defaults (overridable):
  --buffer-size N           Replay buffer capacity (default: 200000)
  --batch-size N            SGD batch size (default: 256)

Any arguments after "--" are appended verbatim to the python command so you can
set niche flags without editing this script.
EOF
}

ALGO="ppo"
ENV_NAME="nes"
ENV_ID=""
TOTAL_STEPS=1500000
LOG_DIR=""
RESUME_FROM=""
DEVICE=""
SEED=123
NUM_ENVS=8
PPO_N_STEPS=4096
PPO_MINIBATCH=2048
DQN_BUFFER=400000
DQN_BATCH=256
EXTRA_ARGS=()
REWARD_WEIGHTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --algo)
      ALGO="$2"
      shift 2
      ;;
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --env-id)
      ENV_ID="$2"
      shift 2
      ;;
    --total-steps)
      TOTAL_STEPS="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --n-steps)
      PPO_N_STEPS="$2"
      shift 2
      ;;
    --minibatch-size)
      PPO_MINIBATCH="$2"
      shift 2
      ;;
    --buffer-size)
      DQN_BUFFER="$2"
      shift 2
      ;;
    --batch-size)
      DQN_BATCH="$2"
      shift 2
      ;;
    --reward-weight)
      REWARD_WEIGHTS+=("$2")
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="runs/${ALGO}_${ENV_NAME}"
fi

case "$ALGO" in
  ppo)
    MODULE="tetris_v2.agents.ppo.train"
    CMD=(python -m "$MODULE"
      --env "$ENV_NAME"
      --total-timesteps "$TOTAL_STEPS"
      --n-steps "$PPO_N_STEPS"
      --num-envs "$NUM_ENVS"
      --minibatch-size "$PPO_MINIBATCH"
      --log-dir "$LOG_DIR"
      --seed "$SEED"
    )
    ;;
  dqn)
    MODULE="tetris_v2.agents.dqn.train"
    CMD=(python -m "$MODULE"
      --env "$ENV_NAME"
      --total-timesteps "$TOTAL_STEPS"
      --buffer-size "$DQN_BUFFER"
      --batch-size "$DQN_BATCH"
      --num-envs "$NUM_ENVS"
      --log-dir "$LOG_DIR"
      --seed "$SEED"
    )
    ;;
  *)
    echo "Unsupported --algo '$ALGO'. Use 'ppo' or 'dqn'." >&2
    exit 1
    ;;
esac

if [[ -n "$ENV_ID" ]]; then
  CMD+=(--env-id "$ENV_ID")
fi
if [[ -n "$RESUME_FROM" ]]; then
  CMD+=(--resume-from "$RESUME_FROM")
fi
if [[ -n "$DEVICE" ]]; then
  CMD+=(--device "$DEVICE")
fi

for weight in "${REWARD_WEIGHTS[@]}"; do
  CMD+=(--advanced-reward-weight "$weight")
done

CMD+=("${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
