import torch
from torch import nn
import torch.nn.functional as F


from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class TransitionSlotLayout():

    def __init__(self, n_actions_steps=3, slots_per_transition=8):
