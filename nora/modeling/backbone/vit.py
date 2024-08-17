from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from nora.layers import CNNBlock
from nora.layers import Conv2d
from nora.layers import DropPath
from nora.layers import MLP
from nora.layers import ShapeSpec
from nora.layers import weight_init
from nora.layers import add_decomposed_rel_pos
from nora.layers import get_abs_pos
from nora.layers import get_norm
from nora.layers import window_partition
from nora.layers import window_unpartition
from .base import Backbone

__all__ = ["ViT"]


class Attention(nn.Module):
    """
    Multi-head Attention block with relative position embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            embed_dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()

        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class ResBottleneckBlock(CNNBlock):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        norm: Union[str, Dict] = "LN",
        activation: Union[str, Dict] = "GELU",
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or dict): normalization for all conv layers.
            activation (str or dict): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, kernel_size=1, norm=norm, activation=activation)
        self.conv2 = Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, norm=norm, activation=activation)
        self.conv3 = Conv2d(bottleneck_channels, out_channels, kernel_size=1, norm=norm)

        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        weight_init.c2_msra_fill(self.conv2)

        weight_init.constant_fill(self.conv1.norm, val=0, bias=0)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        out = identity + x
        return out


class Block(nn.Module):
    """
    Transformer blocks with support of window attention and residual propagation blocks
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_channels: int,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm: Union[str, Dict] = "LN",
        activation: Union[str, Dict] = "GELU",
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        use_residual_block: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            embed_dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            feedforward_channels (int): hidden dim of MLP.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm (str or dict): Normalization layer.
            activation (str or dict): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()

        self.norm1 = get_norm(norm, embed_dim)
        self.attn = Attention(
            embed_dim,
            num_heads,
            qkv_bias,
            use_rel_pos,
            rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = get_norm(norm, embed_dim)
        self.mlp = MLP(in_features=embed_dim, hidden_features=feedforward_channels, activation=activation)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                bottleneck_channels=embed_dim // 2,
                norm="LN",
                activation=activation,
            )

    def forward(self, x):
        shortcut = x

        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernel_size: Union[int, Tuple[int, int]] = 16,
        stride: Union[int, Tuple[int, int]] = 16,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class ViT(Backbone):
    """
    support ViT proposed in ViTDet (https://arxiv.org/abs/2203.16527)
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        feedforward_channels: int,
        image_size: int = 1024,
        input_shape: Optional[ShapeSpec] = None,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        norm: Union[str, Dict] = "LN",
        activation: Union[str, Dict] = "GELU",
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        window_block_indexes: Optional[List[int]]= None,
        residual_block_indexes: Optional[List[int]] = None,
        use_checkpoint: bool = False,
        pretrain_image_size: bool = 224,
        pretrain_use_cls_token: bool = True,
        out_feature="last_feat",
    ):
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.image_size = image_size

        self.patch_embed = PatchEmbed(
            in_channels=input_shape.channels,
            embed_dim=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        if use_abs_pos:
            pretrain_image_size = _pair(pretrain_image_size)
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_image_size[0] // patch_size) * (pretrain_image_size[1] // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_dim=embed_dim,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm=norm,
                activation=activation,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(image_size // patch_size, image_size // patch_size),
            )
            if use_checkpoint:
                from fairscale.nn.checkpoint import checkpoint_wrapper

                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": 0,
            "square_size": self.image_size,
        }

    @staticmethod
    def build(name, **kwargs):
        predefined = {
            "ViT-tiny-p16": vit_tiny_p16,
            "ViT-small-p16": vit_small_p16,
            "ViT-base-p16": vit_base_p16,
            "ViT-large-p16": vit_large_p16,
            "ViT-huge-p14": vit_huge_p14,
            "ViT-giant-p14": vit_giant_p14,
            "ViT-gigiant-p14": vit_gigiant_p14,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for ViT.")

        return predefined[name](**kwargs)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2]))

        for block in self.blocks:
            x = block(x)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.

    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.

    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def vit_tiny_p16(**kwargs):
    return ViT(
        embed_dim=192,
        depth=12,
        num_heads=3,
        patch_size=16,
        feedforward_channels=768,
        drop_path_rate=0.1,
        **kwargs,
    )


def vit_small_p16(**kwargs):
    return ViT(
        embed_dim=384,
        depth=12,
        num_heads=6,
        patch_size=16,
        feedforward_channels=1536,
        drop_path_rate=0.1,
        **kwargs,
    )


def vit_base_p16(**kwargs):
    return ViT(
        embed_dim=768,
        depth=12,
        num_heads=12,
        patch_size=16,
        feedforward_channels=3072,
        drop_path_rate=0.1,
        **kwargs,
    )


def vit_large_p16(**kwargs):
    return ViT(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        patch_size=16,
        feedforward_channels=4096,
        drop_path_rate=0.1,
        **kwargs,
    )


def vit_huge_p14(**kwargs):
    return ViT(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        patch_size=14,
        feedforward_channels=5120,
        drop_path_rate=0.1,
        **kwargs,
    )


def vit_giant_p14(**kwargs):
    return ViT(
        embed_dim=1408,
        depth=40,
        num_heads=16,
        patch_size=14,
        feedforward_channels=6144,
        drop_path_rate=0.1,
        **kwargs,
    )


def vit_gigiant_p14(**kwargs):
    return ViT(
        embed_dim=1664,
        depth=48,
        num_heads=16,
        patch_size=14,
        feedforward_channels=8192,
        drop_path_rate=0.1,
        **kwargs,
    )
