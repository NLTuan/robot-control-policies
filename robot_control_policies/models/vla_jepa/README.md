# VLA-JEPA scaffold

This folder is a small reproduction path, not a copy of LeRobot's full policy.

## Milestone 1: transition-slot layout

Finish `TransitionSlotLayout` in `vla_jepa.py`. It should be pure prompt
layout code: no Qwen loading and no tensors are needed to complete this step.

For `n_action_steps=3` and `slots_per_transition=8`, it should describe:

```text
<|action_0|> x 8, <|action_1|> x 8, <|action_2|> x 8
```

Implement these read-only properties:

- `action_tokens`: one distinct token string per transition step;
- `transition_token_sequence`: each token repeated `slots_per_transition` times;
- `transition_prompt`: the sequence joined into the string placed in Qwen's prompt.

The contract check is intentionally narrow: it checks this layout without
downloading Qwen weights or constructing a world model:

```bash
uv run python -m robot_control_policies.models.vla_jepa.check_layout
```

## Later milestones

1. Add the tokens to Qwen's tokenizer and resize its embedding table.
2. Insert `transition_prompt` into a Qwen image/language prompt.
3. Gather Qwen hidden states at the special-token positions.
4. Attach a small world-model predictor, then V-JEPA2 targets.
5. Add the continuous flow-matching action head last.

For reference only, LeRobot's equivalent token expansion and prompt assembly
are in `src/lerobot/policies/vla_jepa/qwen_interface.py`.
