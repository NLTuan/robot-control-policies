import torch
from torch import nn
from torch.nn import functional as F


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


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        return x / norm.sqrt() * self.weight


class SelfAttention(nn.Module):
    """Self attention with RoPE embeddings applied inside attention."""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.prenorm = RMSNorm(hidden_dim)

    def forward(self, x, cos, sin):
        batch_size, seq_len, _ = x.shape
        residual = x
        x = self.prenorm(x)

        qkv = self.qkv_proj(x)  # [B, seq_len, 3*hidden_dim]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        return residual + self.out_proj(attn_output)


class FeedForward(nn.Module):
    """Standard transformer MLP with residual connection."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.prenorm = RMSNorm(hidden_dim)

    def forward(self, x):
        return x + self.mlp(self.prenorm(x))


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
        image_size=64,
        patch_size=16,
        image_encoder_depth=2,
        max_action_horizon=256,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if action_horizon > max_action_horizon:
            raise ValueError("action_horizon cannot exceed max_action_horizon")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_action_horizon = max_action_horizon

        # state -> one state token [B, 1, hidden_dim]
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # images -> image tokens [B, num_image_tokens, hidden_dim]
        self.image_encoder = TinyImageEncoder(
            channels=image_channels,
            hidden_dim=hidden_dim,
            image_size=image_size,
            patch_size=patch_size,
            num_blocks=image_encoder_depth,
            num_heads=num_heads,
        )

        # task token ids -> task tokens [B, task_len, hidden_dim]
        self.task_embedder = TaskTokenEmbedder(vocab_size=vocab_size, hidden_dim=hidden_dim)

        # noisy action chunk -> action tokens [B, action_horizon, hidden_dim]
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # scalar flow time -> conditioning vector [B, hidden_dim]
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # RoPE for the custom token mixer attention.
        head_dim = hidden_dim // num_heads
        self.rope = RotaryPositionEmbedding(head_dim)

        self.depth = depth
        self.attn_layers = nn.ModuleList([SelfAttention(hidden_dim, num_heads) for _ in range(depth)])
        self.mlp_action_layers = nn.ModuleList(
            [
                FeedForward(hidden_dim)
                for _ in range(depth)
            ]
        )
        self.mlp_obs_layers = nn.ModuleList(
            [
                FeedForward(hidden_dim)
                for _ in range(depth)
            ]
        )   

        # action tokens -> predicted flow [B, action_horizon, action_dim]
        self.flow_head = nn.Linear(hidden_dim, action_dim)

    def embed_task(self, task_tokens, batch_size):
        return self.task_embedder(task_tokens, batch_size)

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

        batch_size, action_horizon, _ = noisy_actions.shape
        if action_horizon != self.action_horizon:
            raise ValueError(
                f"expected action_horizon={self.action_horizon}, got {action_horizon}"
            )

        if t.ndim == 0:
            t = t.expand(batch_size)

        state_token = self.state_proj(state.float())[:, None, :]
        image_tokens = self.image_encoder(images, batch_size=batch_size)
        task_tokens = self.embed_task(task_tokens, batch_size=batch_size)

        context_tokens = torch.cat([state_token, image_tokens, task_tokens], dim=1)

        action_tokens = self.action_proj(noisy_actions.float())
        time_emb = self.time_emb(t.float().reshape(batch_size, 1))
        action_tokens = action_tokens + time_emb[:, None, :]

        for layer_idx in range(self.depth):
            tokens = torch.cat([context_tokens, action_tokens], dim=1)
            cos, sin = self.rope(tokens.shape[1], device=tokens.device)
            tokens = self.attn_layers[layer_idx](tokens, cos, sin)

            context_tokens, action_tokens = torch.split(
                tokens,
                [context_tokens.shape[1], action_tokens.shape[1]],
                dim=1,
            )

            context_tokens = self.mlp_obs_layers[layer_idx](context_tokens)
            action_tokens = self.mlp_action_layers[layer_idx](action_tokens)

        return self.flow_head(action_tokens)

    def compute_loss(self, state, actions, images=None, task_tokens=None):
        """Flow-matching loss for action generation."""
        batch_size = actions.shape[0]
        t = torch.rand(batch_size, device=actions.device)
        noise = torch.randn_like(actions)

        noised_actions = (1.0 - t[:, None, None]) * noise + t[:, None, None] * actions
        target_flow = actions - noise

        pred_flow = self.forward(
            state,
            noised_actions,
            t,
            images=images,
            task_tokens=task_tokens,
        )
        loss = F.mse_loss(pred_flow, target_flow)
        return loss

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
