import torch
from torch import nn


class RotaryPositionEmbedding(nn.Module):
    """Rotary position embeddings for attention q/k tensors.

    RoPE is applied inside attention to query/key tensors, not added directly to
    token embeddings like learned absolute position embeddings.

    The returned tensors are shaped `[1, 1, seq_len, head_dim]` so they
    broadcast over attention tensors shaped `[B, num_heads, seq_len, head_dim]`.
    """

    def __init__(self, head_dim, theta=10000):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device=None):
        device = device or self.inv_freq.device
        positions = torch.arange(seq_len, device=device)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        freqs = torch.cat([freqs, freqs], dim=-1)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary embeddings to attention queries or keys.

    Args:
        x: Tensor shaped `[B, num_heads, seq_len, head_dim]`.
        cos: Cosine frequencies broadcastable to `x`.
        sin: Sine frequencies broadcastable to `x`.
    """

    first_half, second_half = x.chunk(2, dim=-1)
    rotated_x = torch.cat([-second_half, first_half], dim=-1)
    return x * cos + rotated_x * sin


class TinyImageEncoder(nn.Module):
    """Small ViT-style image encoder for shape plumbing.

    Target contract:
      images is either:
        None
        [B, T, C, H, W]

      return:
        [B, T * num_patches, hidden_dim]

    Treat `T` as time, camera-view count, or time x camera after the dataloader
    standardizes it.
    """

    def __init__(
        self,
        channels,
        hidden_dim=128,
        image_size=64,
        patch_size=16,
        num_blocks=2,
        num_heads=4,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.channels = channels
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.null_image_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.patchifier = nn.Conv2d(
            channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

        self.vit_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, images, batch_size):
        if images is None:
            return self.null_image_token.expand(batch_size, -1, -1)

        if images.ndim != 5:
            raise ValueError(f"images must have shape [B, T, C, H, W], got {images.shape}")

        batch, num_images, channels, height, width = images.shape
        if batch != batch_size:
            raise ValueError(f"batch_size={batch_size} but images batch={batch}")
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} image channels, got {channels}")
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"expected image size {self.image_size}x{self.image_size}, "
                f"got {height}x{width}"
            )

        images = self.patchifier(images.flatten(0, 1))  # [B*T, hidden_dim, H', W']
        images = images.flatten(2).transpose(1, 2)  # [B*T, num_patches, hidden_dim]
        images = images + self.pos_emb

        for layer in self.vit_layers:
            images = layer(images)

        return images.reshape(batch_size, num_images * self.num_patches, self.hidden_dim)


class TaskTokenEmbedder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.null_task_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, task_token_ids, batch_size):
        if task_token_ids is None:
            return self.null_task_token.expand(batch_size, -1, -1)
        return self.token_emb(task_token_ids)

class Pi0Tiny(nn.Module):
    """Tiny pi0-style scaffold for learning the model contract.

    Goal:
      state + optional image/task context + noisy action chunk + time
      -> predicted action flow

    This should stay small enough for CPU smoke tests.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        action_horizon,
        hidden_dim=128,
        depth=2,
        num_heads=4,
        vocab_size=32000,
        image_channels=3,
        max_action_horizon=256,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim

        # TODO: state -> one state token [B, 1, hidden_dim]
        # self.state_proj = ...

        # TODO: images -> image tokens [B, num_image_tokens, hidden_dim]
        # self.image_encoder = ...

        # TODO: task token ids -> task tokens [B, task_len, hidden_dim]
        # self.task_token_emb = ...
        # self.null_task_token = ...

        # TODO: noisy action chunk -> action tokens [B, action_horizon, hidden_dim]
        # self.action_proj = ...

        # TODO: scalar flow time -> conditioning vector [B, hidden_dim]
        # self.time_emb = ...

        # TODO: RoPE for attention positions.
        # This requires a custom attention block instead of plain
        # nn.TransformerEncoderLayer, because PyTorch's stock layer does not
        # expose q/k before attention.
        #
        # head_dim = hidden_dim // num_heads
        # self.rope = RotaryPositionEmbedding(head_dim)

        # TODO: tiny token mixer.
        # If using RoPE, write a small Transformer block yourself:
        #   x -> norm -> qkv projection
        #   q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        #   attention -> residual -> MLP -> residual
        # self.transformer = ...

        # TODO: action tokens -> predicted flow [B, action_horizon, action_dim]
        # self.flow_head = ...

    def embed_task(self, task_tokens, batch_size):
        # TODO:
        # 1. If task_tokens is None, return a learned null task token.
        # 2. Otherwise embed token ids with self.task_token_emb.
        raise NotImplementedError

    def forward(self, state, noisy_actions, t, images=None, task_tokens=None):
        """Predict flow for noisy action tokens.

        Inputs:
          state:         [B, state_dim]
          noisy_actions: [B, action_horizon, action_dim]
          t:             [B]
          images:        optional image tensor [B, T, C, H, W]
          task_tokens:   optional int token ids [B, task_len]

        Output:
          pred_flow:     [B, action_horizon, action_dim]
        """

        # TODO:
        # 1. Read batch_size and action_horizon from noisy_actions.
        # 2. Project state to one token.
        # 3. Encode images to image tokens.
        # 4. Embed task tokens.
        # 5. Project noisy_actions to action tokens.
        # 6. Add time embedding to action tokens.
        # 7. Concatenate [state, image, task, action] tokens.
        # 8. Build RoPE cos/sin for total token sequence length.
        # 9. Run the transformer with RoPE applied inside attention.
        # 10. Slice out the final action tokens.
        # 11. Project action tokens to predicted flow.
        raise NotImplementedError

    def compute_loss(self, state, actions, images=None, task_tokens=None):
        """TODO: Flow-matching objective.

        Target recipe:
          noise = randn_like(actions)
          t = sample from Beta(1.5, 1.0), shape [B]
          x_t = t * noise + (1 - t) * actions
          target_flow = noise - actions
          pred_flow = self.forward(state, x_t, t, images, task_tokens)
          return mse(pred_flow, target_flow)
        """
        raise NotImplementedError

    def sample_actions(self, state, images=None, task_tokens=None, num_steps=10):
        """TODO: Euler sampler.

        Target recipe:
          actions = randn([B, action_horizon, action_dim])
          dt = -1 / num_steps
          for t from 1 -> 0:
              flow = self.forward(state, actions, t, images, task_tokens)
              actions = actions + dt * flow
          return actions
        """
        raise NotImplementedError
