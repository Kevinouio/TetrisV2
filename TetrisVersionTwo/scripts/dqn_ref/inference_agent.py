from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

from .config import DQNRefConfig
from .model import LinearQNet


class DQNRefInferenceAgent:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        cfg = DQNRefConfig()
        self.model = LinearQNet(
            input_size=int(cfg.model.input_size),
            hidden_sizes=tuple(int(v) for v in cfg.model.hidden_sizes),
            output_size=int(cfg.model.output_size),
        ).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_action(self, candidates: Sequence[Any]) -> Any:
        action, _ = self.predict_action_with_diagnostics(candidates)
        return action

    def predict_action_with_diagnostics(
        self,
        candidates: Sequence[Any],
    ) -> Tuple[Any, Dict[str, object]]:
        if not candidates:
            raise RuntimeError("No candidates available for DQN inference.")

        feature_batch = np.stack(
            [np.asarray(candidate.feature_vector, dtype=np.float32) for candidate in candidates],
            axis=0,
        )
        x = torch.as_tensor(feature_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.model(x).view(-1)
        best_idx = int(torch.argmax(q_values).item())
        best_q = float(q_values[best_idx].item())

        chosen = candidates[best_idx]
        diagnostics: Dict[str, object] = {
            "best_q": best_q,
            "candidate_count": int(len(candidates)),
            "chosen_index": int(best_idx),
        }
        action_tuple = getattr(chosen, "action_tuple", None)
        if action_tuple is not None:
            diagnostics["chosen_action_tuple"] = list(action_tuple)
        return chosen.native_action, diagnostics
