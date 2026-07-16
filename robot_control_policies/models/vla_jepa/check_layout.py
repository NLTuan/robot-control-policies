"""Run after implementing TransitionSlotLayout.

Usage:
    uv run python -m robot_control_policies.models.vla_jepa.check_layout
"""

from .vla_jepa import TransitionSlotLayout


def main() -> None:
    layout = TransitionSlotLayout(n_actions_steps=3, slots_per_transition=2)

    assert layout.action_tokens == [
        "<|action_0|>",
        "<|action_1|>",
        "<|action_2|>",
    ]
    assert layout.transition_token_sequence == [
        "<|action_0|>",
        "<|action_0|>",
        "<|action_1|>",
        "<|action_1|>",
        "<|action_2|>",
        "<|action_2|>",
    ]
    assert layout.transition_prompt == "".join(layout.transition_token_sequence)
    print("TransitionSlotLayout contract passes.")


if __name__ == "__main__":
    main()
