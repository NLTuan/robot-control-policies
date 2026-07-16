from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class TransitionSlotLayout:
    """The prompt-side layout for VLA-JEPA's transition slots."""

    def __init__(self, n_actions_steps=3, slots_per_transition=2):
        self.action_tokens = [f"<|action_{i}|>" for i in range(n_actions_steps)]
        self.transition_token_sequence = [
            token for token in self.action_tokens for _ in range(slots_per_transition)
        ]
        self.transition_prompt = "".join(self.transition_token_sequence)

    def add_to_tokenizer(self, tokenizer) -> list[int]:
        vocabulary = tokenizer.get_vocab()
        missing_tokens = [token for token in self.action_tokens if token not in vocabulary]
        if missing_tokens:
            tokenizer.add_tokens(missing_tokens, special_tokens=True)
        return [tokenizer.convert_tokens_to_ids(token) for token in self.action_tokens]


class QwenTransitionSlotEncoder(nn.Module):
    """Milestone 2 scaffold: turn an image/instruction into Qwen slot features.

    Fill these methods in order. Keep this class limited to Qwen and the
    transition-slot interface; V-JEPA2 and the robot action head belong in
    later modules.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        layout: TransitionSlotLayout | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.layout = layout or TransitionSlotLayout()
        self.processor: Any | None = None
        self.model: Qwen3VLForConditionalGeneration | None = None
        self.slot_ids: Tensor | None = None

    def load_backbone(self) -> None:
        """Load the processor/model, add slot tokens, and resize embeddings."""
        raise NotImplementedError

    def build_inputs(
        self, images: Sequence[Sequence[Any]], instructions: Sequence[str]
    ) -> Any:
        """Build Qwen's chat-template inputs with ``layout.transition_prompt``."""
        raise NotImplementedError

    def encode(self, images: Sequence[Sequence[Any]], instructions: Sequence[str]) -> Tensor:
        """Return contextual transition slots with shape [B, slots, hidden_size]."""
        raise NotImplementedError
