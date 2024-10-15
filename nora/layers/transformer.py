# Copyright (c) Facebook, Inc. and its affiliates.

import math
import warnings
from typing import Dict
from typing import Tuple
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .activation import get_activation
from .norm import get_norm

__all__ = [
    "ConditionalCrossAttention",
    "ConditionalSelfAttention",
    "DropPath",
    "FFN",
    "MLP",
    "MultiHeadAttention",
    "SinePositionEmbedding",
    "add_decomposed_rel_pos",
    "coordinate_to_embedding",
    "get_abs_pos",
    "get_rel_pos",
    "window_partition",
    "window_unpartition",
]


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob: float = 0, scale_by_keep: bool = True):
        super().__init__()

        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: Union[str, Dict] = "GELU",
        norm: Union[str, Dict] = "",
        bias: bool = True,
        drop: float = 0,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _pair(bias)
        drop_probs = _pair(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.activation = get_activation(activation)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = get_norm(norm, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop1(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FFN(MLP):
    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.drop1(out)
        if self.norm is not None:
            out = self.norm(out)
        out = self.fc2(out)
        out = self.drop2(out)
        out += x
        return out


class SinePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 1000,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
        normalize: bool = False,
    ):
        """
        Args:
            num_pos_feats (int): The feature dimension for each position along
                x-axis or y-axis. The final returned dimension for each position
                is 2 times of the input value.
            temperature (int, optional): The temperature used for scaling
                the position embedding. Default: 10000.
            scale (float, optional): A scale factor that scales the position
                embedding. The scale will be used only when `normalize` is True.
                Default: 2*pi.
            eps (float, optional): A value added to the denominator for numerical
                stability. Default: 1e-6.
            offset (float): An offset added to embed when doing normalization.
            normalize (bool, optional): Whether to normalize the position embedding.
                Default: False.
        """
        super().__init__()

        if normalize:
            assert isinstance(scale, (float, int)), f"when normalize is set, scale should be provided and in float or int type, found {type(scale)}"

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: torch.Tensor):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        B, H, W = mask.size()
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class MultiHeadAttention(nn.Module):
    """
    A wrapper for `torch.nn.MultiheadAttention`.

    implement MultiHeadAttention with identity connection,
    and position embedding is also passed as input.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        batch_first: bool = False,
    ):
        """
        Args:
            embed_dim (int): the embedding dimension for attention.
            num_heads (int): the number of attention head.
            attn_drop (float): A Dropout layer on attn_output_weights. Default: 0.0
            proj_drop (float): A Dropout layer after `MultiHeadAttention`. Default: 0.0
            batch_first (bool): If True, then the input and output tensors will be provided as (B, N, C).
                Default: False.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            query (Tensor): query embedding with shape `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`.
            key (Tensor): key embedding with shape `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`.
            value (Tensor): value embedding with the same shape as `key`. if None, it will be set to `key`.
            identity (Tensor): the tensor, with the same shape as query, will be used for identity addition.
                Default: None. If None, `query` will be used.
            query_pos (Tensor): the position embedding of query with the shape as `query`. Default: None.
            key_pos (Tensor): the position embedding of key with the shape as `key`. Default: None.
            attn_mask (Tensor): ByteTensor mask with shape `(num_query, num_key)`. Default: None.
            key_padding_mask (Tensor): ByteTensor mask with shape `(bs, num_key)` which indicates
                which elements within `key` to be ignored in attention. Default: None.
        """
        if key is None:
            key = query

        if value is None:
            value = key

        if identity is None:
            identity = query

        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f"position encoding of key is missing in {self.__class__.__name__}.")

        if query_pos is not None:
            query = query + query_pos

        if key_pos is not None:
            key = key + key_pos

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]

        return identity + self.proj_drop(out)


class ConditionalSelfAttention(nn.Module):
    """
    Conditional Self-Attention used in Conditional DETR.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        batch_first: bool = False,
    ):
        """
        Args:
            embed_dim (int): the embedding dimension for attention.
            num_heads (int): the number of attention head.
            attn_drop (float): A Dropout layer on attn_output_weights. Default: 0.0
            proj_drop (float): A Dropout layer after `MultiHeadAttention`. Default: 0.0
            batch_first (bool): If True, then the input and output tensors will be provided as (B, N, C).
                Default: False.
        """
        super().__init__()

        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.batch_first = batch_first

        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            query (Tensor): query embedding with shape `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`.
            key (Tensor): key embedding with shape `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`.
            value (Tensor): value embedding with the same shape as `key`. if None, it will be set to `key`.
            identity (Tensor): the tensor, with the same shape as query, will be used for identity addition.
                Default: None. If None, `query` will be used.
            query_pos (Tensor): the position embedding of query with the shape as `query`. Default: None.
            key_pos (Tensor): the position embedding of key with the shape as `key`. Default: None.
            attn_mask (Tensor): ByteTensor mask with shape `(num_query, num_key)`. Default: None.
            key_padding_mask (Tensor): ByteTensor mask with shape `(bs, num_key)` which indicates
                which elements within `key` to be ignored in attention. Default: None.
        """
        if key is None:
            key = query

        if value is None:
            value = key

        if identity is None:
            identity = query

        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f"position encoding of key is missing in {self.__class__.__name__}.")

        assert query_pos is not None and key_pos is not None, "query_pos and key_pos must be passed into ConditionalSelfAttention"

        # query/key/value content and position embedding projection
        query_content = self.query_content_proj(query)
        query_pos = self.query_pos_proj(query_pos)
        key_content = self.key_content_proj(key)
        key_pos = self.key_pos_proj(key_pos)
        value = self.value_proj(value)

        # attention calculation
        B, L, C = query_content.shape
        q = query_content + query_pos
        k = key_content + key_pos
        v = value

        q = q.reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, num_heads, L, head_dim)
        k = k.reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # add attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float("-inf"))
            else:
                attn += attn_mask

        if key_padding_mask is not None:
            attn = attn.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        out = self.out_proj(out)

        return identity + self.proj_drop(out)


class ConditionalCrossAttention(nn.Module):
    """
    Conditional Cross-Attention used in Conditional DETR.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        batch_first: bool = False,
        is_first: bool = False,
    ):
        """
        Args:
            embed_dim (int): the embedding dimension for attention.
            num_heads (int): the number of attention head.
            attn_drop (float): A Dropout layer on attn_output_weights. Default: 0.0
            proj_drop (float): A Dropout layer after `MultiHeadAttention`. Default: 0.0
            batch_first (bool): If True, then the input and output tensors will be provided as (B, N, C).
                Default: False.
        """
        super().__init__()

        self.num_heads = num_heads
        self.batch_first = batch_first

        self.query_content_proj = nn.Linear(embed_dim, embed_dim)
        if is_first:
            self.query_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.query_pos_sine_proj = nn.Linear(embed_dim, embed_dim)
        self.key_content_proj = nn.Linear(embed_dim, embed_dim)
        self.key_pos_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        query_sine_embed: Optional[torch.Tensor] = None,
        is_first: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            query (Tensor): query embedding with shape `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`.
            key (Tensor): key embedding with shape `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`.
            value (Tensor): value embedding with the same shape as `key`. if None, it will be set to `key`.
            identity (Tensor): the tensor, with the same shape as query, will be used for identity addition.
                Default: None. If None, `query` will be used.
            query_pos (Tensor): the position embedding of query with the shape as `query`. Default: None.
            key_pos (Tensor): the position embedding of key with the shape as `key`. Default: None.
            attn_mask (Tensor): ByteTensor mask with shape `(num_query, num_key)`. Default: None.
            key_padding_mask (Tensor): ByteTensor mask with shape `(bs, num_key)` which indicates
                which elements within `key` to be ignored in attention. Default: None.
        """
        if key is None:
            key = query

        if value is None:
            value = key

        if identity is None:
            identity = query

        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f"position encoding of key is missing in {self.__class__.__name__}.")

        assert query_pos is not None and key_pos is not None, "query_pos and key_pos must be passed into ConditionalCrossAttention"

        # content projection
        query_content = self.query_content_proj(query)
        key_content = self.key_content_proj(key)
        value = self.value_proj(value)

        # shape info
        B, N, C = query_content.shape
        _, L, _ = key_content.shape

        # position projection
        key_pos = self.key_pos_proj(key_pos)
        if is_first:
            query_pos = self.query_pos_proj(query_pos)
            q = query_content + query_pos
            k = key_content + key_pos
        else:
            q = query_content
            k = key_content
        v = value

        # preprocess
        q = q.view(B, N, self.num_heads, C // self.num_heads)
        query_sine_embed = self.query_pos_sine_proj(query_sine_embed).view(B, N, self.num_heads, C // self.num_heads)
        q = torch.cat([q, query_sine_embed], dim=3).view(B, N, C * 2)

        k = k.view(B, L, self.num_heads, C // self.num_heads)  # N, 16, 256
        key_pos = key_pos.view(B, L, self.num_heads, C // self.num_heads)
        k = torch.cat([k, key_pos], dim=3).view(B, L, C * 2)

        # attention calculation
        q = q.reshape(B, N, self.num_heads, C * 2 // self.num_heads).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        k = k.reshape(N, L, self.num_heads, C * 2 // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(N, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        scale = (C * 2 // self.num_heads) ** -0.5
        q = q * scale
        attn = q @ k.transpose(-2, -1)

        # add attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn.masked_fill_(attn_mask, float("-inf"))
            else:
                attn += attn_mask

        if key_padding_mask is not None:
            attn = attn.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        return identity + self.proj_drop(out)


def window_partition(x: torch.Tensor, window_size: int):
    """
    Partition into non-overlapping windows with padding if needed.

    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]):
    """
    Window unpartition into original sequences and removing padding.

    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)

    return attn


def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.

    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


def coordinate_to_embedding(
    coord_tensor: torch.Tensor,
    num_feats: int = 128,
    temperature: int = 10000,
    scale: Optional[float] = None
):
    """Convert coordinate tensor to positional encoding.

    Args:
        coord_tensor (Tensor): Coordinate tensor to be converted to
            positional encoding. With the last dimension as 2 or 4.
        num_feats (int, optional): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value. Defaults to 128.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
    Returns:
        Tensor: Returned encoded positional tensor.
    """
    if scale is None:
        scale = 2 * math.pi

    dim_t = torch.arange(num_feats, dtype=torch.float32, device=coord_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
    x_embed = coord_tensor[..., 0] * scale
    y_embed = coord_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(2)

    if coord_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1)
    elif coord_tensor.size(-1) == 4:
        w_embed = coord_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(2)

        h_embed = coord_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1): {coord_tensor.size(-1)}")
    return pos
