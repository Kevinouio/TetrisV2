from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import EncoderConfig
from .encoders import encode_state, flatten_aux_features
from .model import BCPolicyNet
from .utils import ActionCodec, ActionTuple, NativeAction


class BCAgent:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Optional[str] = None,
        env_adapter=None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.env_adapter = env_adapter

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model_config = checkpoint["model_config"]
        encoder_config = checkpoint["encoder_config"]
        id_to_action = checkpoint["id_to_action"]

        self.encoder_config = EncoderConfig(
            board_height=int(encoder_config["board_height"]),
            board_width=int(encoder_config["board_width"]),
            queue_length=int(encoder_config["queue_length"]),
            include_scalars=bool(encoder_config["include_scalars"]),
        )
        self.codec = ActionCodec(id_to_action=id_to_action)
        self.model = BCPolicyNet(
            action_vocab_size=int(model_config["action_vocab_size"]),
            aux_dim=int(model_config["aux_dim"]),
            board_height=int(model_config["board_height"]),
            board_width=int(model_config["board_width"]),
            conv_channels=tuple(int(v) for v in model_config["conv_channels"]),
            mlp_hidden=tuple(int(v) for v in model_config["mlp_hidden"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_logits(self, state: Dict[str, object]) -> np.ndarray:
        encoded = encode_state(state, self.encoder_config)
        board = torch.from_numpy(encoded["board"]).unsqueeze(0).to(self.device)
        aux = torch.from_numpy(flatten_aux_features(encoded)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(board, aux)[0].detach().cpu().numpy()
        return logits

    def predict_action(
        self,
        state: Dict[str, object],
        legal_actions: Optional[List[Tuple[NativeAction, ActionTuple]]] = None,
    ) -> NativeAction:
        action, _ = self.predict_action_with_diagnostics(state, legal_actions=legal_actions)
        return action

    def predict_action_with_diagnostics(
        self,
        state: Dict[str, object],
        legal_actions: Optional[List[Tuple[NativeAction, ActionTuple]]] = None,
    ) -> Tuple[NativeAction, Dict[str, object]]:
        if legal_actions is None:
            if self.env_adapter is None:
                raise ValueError(
                    "legal_actions must be passed when env_adapter is not attached to BCAgent."
                )
            legal_actions = self.env_adapter.enumerate_legal_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions available.")

        logits = self.predict_logits(state)
        raw_argmax_id = int(np.argmax(logits))
        raw_tuple = self.codec.decode_id(raw_argmax_id)

        legal_tuples = [t for _, t in legal_actions]
        legal_ids, unseen_legal = self.codec.legal_ids(legal_tuples)
        raw_invalid = int(raw_tuple not in legal_tuples)

        diagnostics = {
            "raw_argmax_id": raw_argmax_id,
            "raw_argmax_tuple": list(raw_tuple),
            "raw_argmax_invalid": bool(raw_invalid),
            "unseen_legal_count": int(unseen_legal),
            "used_fallback_unseen_legal": False,
            "selected_action_id": None,
            "selected_action_tuple": None,
        }

        if legal_ids:
            legal_scores = [(aid, float(logits[aid])) for aid in legal_ids]
            legal_scores.sort(key=lambda row: row[1], reverse=True)
            selected_action_id = int(legal_scores[0][0])
            selected_tuple = self.codec.decode_id(selected_action_id)
            for native, tup in legal_actions:
                if tup == selected_tuple:
                    diagnostics["selected_action_id"] = selected_action_id
                    diagnostics["selected_action_tuple"] = list(selected_tuple)
                    return native, diagnostics

        diagnostics["used_fallback_unseen_legal"] = True
        fallback_native, fallback_tuple = sorted(legal_actions, key=lambda row: row[1])[0]
        diagnostics["selected_action_id"] = None
        diagnostics["selected_action_tuple"] = list(fallback_tuple)
        return fallback_native, diagnostics

