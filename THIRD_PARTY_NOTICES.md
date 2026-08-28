# Third-party notices

The `cc2_*` planner and the Cold Clear adapter in `include/tetris_v2/` and
`src/` are C++ adaptations of
[Cold Clear 2](https://github.com/MinusKelvin/cold-clear-2), originally by
Mark Carlson. The port was based on commit
`ed8b19327b6bd1410ddd873d8611485bd45d8fae` and has been modified for this
repository's runtime and C API.

Cold Clear 2 is offered under MIT or Apache-2.0 at the user's option. This
repository uses it under the MIT option; see
[`licenses/COLD_CLEAR_2_MIT.txt`](licenses/COLD_CLEAR_2_MIT.txt).

The discrete Flow-DQN design in `tetris_v2/rl/flow_dqn/` is inspired by
[Flow Q-Learning](https://arxiv.org/abs/2502.02538) by Seohong Park, Qiyang Li,
and Sergey Levine. Its full placement-map policy is an independent PyTorch
adaptation for TetrisV2's masked discrete action space, rather than a literal
implementation of the paper's continuous-action algorithm. The authors'
[official FQL implementation](https://github.com/seohongpark/fql) is available
separately under the MIT license.
