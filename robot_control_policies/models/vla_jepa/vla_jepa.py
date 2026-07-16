import torch
from torch import nn
import torch.nn.functional as F


from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class TransitionSlotLayout():

    def __init__(self, n_actions_steps=3, slots_per_transition=2):
        self.action_tokens = [f"<|action_{i}|>" for i in range(n_actions_steps)]

        self.transition_token_sequence = [token for token in self.action_tokens for i in range(slots_per_transition)]

        self.transition_prompt = "".join(self.transition_token_sequence)

    def add_to_tokenizer(self, tokenizer):
        vocabulary = tokenizer.get_vocab()

        missing_tokens = [
            token for token in self.action_tokens
            if token not in vocabulary
        ]

        if missing_tokens:
            tokenizer.add_tokens(missing_tokens, special_tokens=True)

        return [
            tokenizer.convert_tokens_to_ids(token)
            for token in self.action_tokens
        ]