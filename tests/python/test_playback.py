from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import play_rl_cli, play_rl_pygame


@pytest.mark.parametrize("module", [play_rl_cli, play_rl_pygame])
def test_playback_routes_flow_dqn_and_rejects_observation_mismatch(
    module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = []
    environments = []

    class Policy:
        obs_dim = 451
        action_dim = 3

    class Env:
        action_space = SimpleNamespace(n=3)
        observation_space = SimpleNamespace(shape=(254,))

        def __init__(self, **_: object):
            self.closed = False
            environments.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        module,
        "load_policy",
        lambda algo, *_args, **_kwargs: requested.append(algo) or Policy(),
    )
    monkeypatch.setattr(module, "CCTetrisEnv", Env)

    with pytest.raises(SystemExit, match="obs_dim=451"):
        module.main(["checkpoint.pt", "--algo", "flow_dqn"])

    assert requested == ["flow_dqn"]
    assert environments[0].closed

