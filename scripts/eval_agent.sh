#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/eval_agent.sh [options] [-- extra-flags]

Options:
  --algo {ppo|dqn}          Algorithm checkpoint to evaluate (default: ppo)
  --env {nes|modern|custom} Environment choice (default: nes)
  --env-id ID               Custom Gymnasium id when --env=custom
  --checkpoint PATH         Path to .pt checkpoint (default: runs/<algo>_<env>/final_model.pt)
  --episodes N              Number of eval episodes (default: 10)
  --seed N                  RNG seed for eval (default: 321)
  --device DEV              Torch device override (e.g., cuda:0 or cpu)
  --render                  Enable pygame window rendering
  --reward-weight KEY=VAL   Override AdvancedRewardConfig field (repeatable)

All additional arguments after "--" are appended verbatim to the python command.
EOF
}

ALGO="ppo"
ENV_NAME="nes"
ENV_ID=""
CHECKPOINT=""
EPISODES=10
SEED=321
DEVICE=""
RENDER=0
REWARD_WEIGHTS=()
EXTRA_ARGS=()

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
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --render)
      RENDER=1
      shift
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

if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="runs/${ALGO}_${ENV_NAME}/final_model.pt"
fi

case "$ALGO" in
  ppo)
    MODULE="tetris_v2.agents.ppo.eval"
    ;;
  dqn)
    MODULE="tetris_v2.agents.dqn.eval"
    ;;
  *)
    echo "Unsupported --algo '$ALGO'. Use 'ppo' or 'dqn'." >&2
    exit 1
    ;;
esac

CMD=(python -m "$MODULE"
  "$CHECKPOINT"
  --env "$ENV_NAME"
  --episodes "$EPISODES"
  --seed "$SEED"
)

if [[ -n "$ENV_ID" ]]; then
  CMD+=(--env-id "$ENV_ID")
fi
if [[ -n "$DEVICE" ]]; then
  CMD+=(--device "$DEVICE")
fi
if [[ $RENDER -eq 1 ]]; then
  CMD+=(--render)
fi

for weight in "${REWARD_WEIGHTS[@]}"; do
  CMD+=(--advanced-reward-weight "$weight")
done

CMD+=("${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
