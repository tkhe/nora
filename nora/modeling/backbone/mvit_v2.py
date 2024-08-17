# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from nora.layers import DropPath
from nora.layers import MLP
from nora.layers import ShapeSpec
from nora.layers import add_decomposed_rel_pos
from nora.layers import get_abs_pos
from nora.layers import get_norm
from nora.layers import window_partition
from nora.layers import window_unpartition
from .base import Backbone

__all__ = ["MViTv2"]


def attention_pool(x, pool, norm=None):
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H1, W1) -> (B, H1, W1, C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttetion(nn.Module):
    """
    Multiscale Multi-head Attention block.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        qkv_bias: bool = True,
        norm: Union[str, Dict] = "LN",
        pool_kernel: Tuple[int, int] = (3, 3),
        stride_q: int = 1,
        stride_kv: int = 1,
        residual_pooling: bool = True,
        window_size: int = 0,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[int] = None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            norm (str, dict): Normalization layer type. Default: "LN"
            pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
        """
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        # qkv pooling
        pool_padding = [(k - 1) // 2 for k in _pair(pool_kernel)]
        dim_conv = dim_out // num_heads
        self.pool_q = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_q,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_q = get_norm(norm, dim_conv)
        self.pool_k = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_kv,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_k = get_norm(norm, dim_conv)
        self.pool_v = nn.Conv2d(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_kv,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_v = get_norm(norm, dim_conv)

        self.window_size = window_size
        if window_size:
            self.q_win_size = window_size // stride_q
            self.kv_win_size = window_size // stride_kv

        self.residual_pooling = residual_pooling

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            assert input_size[0] == input_size[1]

            size = input_size[0]
            rel_dim = 2 * max(size // stride_q, size // stride_kv) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_dim, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        # qkv with shape (3, B, nHead, H, W, C)
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, -1).permute(3, 0, 4, 1, 2, 5)
        # q, k, v with shape (B, nHead, H, W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H, W, -1).unbind(0)

        q = attention_pool(q, self.pool_q, self.norm_q)
        k = attention_pool(k, self.pool_k, self.norm_k)
        v = attention_pool(v, self.pool_v, self.norm_v)

        ori_q = q
        if self.window_size:
            q, q_hw_pad = window_partition(q, self.q_win_size)
            k, kv_hw_pad = window_partition(k, self.kv_win_size)
            v, _ = window_partition(v, self.kv_win_size)
            q_hw = (self.q_win_size, self.q_win_size)
            kv_hw = (self.kv_win_size, self.kv_win_size)
        else:
            q_hw = q.shape[1:3]
            kv_hw = k.shape[1:3]

        q = q.view(q.shape[0], np.prod(q_hw), -1)
        k = k.view(k.shape[0], np.prod(kv_hw), -1)
        v = v.view(v.shape[0], np.prod(kv_hw), -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, q_hw, kv_hw)

        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.view(x.shape[0], q_hw[0], q_hw[1], -1)

        if self.window_size:
            x = window_unpartition(x, self.q_win_size, q_hw_pad, ori_q.shape[1:3])

        if self.residual_pooling:
            x += ori_q

        H, W = x.shape[1], x.shape[2]
        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    """
    MultiScale Transformer block.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm: Union[str, Dict] = "LN",
        actiation: Union[str, Dict] = "GELU",
        qkv_pool_kernel: Tuple[int, int] = (3, 3),
        stride_q: int = 1,
        stride_kv: int = 1,
        residual_pooling: bool = True,
        window_size: int = 0,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[int] = None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads in the MViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm (str, dict): Normalization layer.
            activation (str, dict): Activation layer.
            qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
        """
        super().__init__()

        self.norm1 = get_norm(norm, dim)
        self.attn = MultiScaleAttetion(
            dim,
            dim_out,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm=norm,
            pool_kernel=qkv_pool_kernel,
            stride_q=stride_q,
            stride_kv=stride_kv,
            residual_pooling=residual_pooling,
            window_size=window_size,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
        )

        self.drop_path = DropPath(drop_path)
        self.norm2 = get_norm(norm, dim_out)
        self.mlp = MLP(
            in_features=dim_out,
            hidden_features=int(dim_out * mlp_ratio),
            out_features=dim_out,
            activation=actiation,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
        else:
            self.proj = None

        if stride_q > 1:
            kernel_skip = stride_q + 1
            padding_skip = int(kernel_skip // 2)
            self.pool_skip = nn.MaxPool2d(kernel_skip, stride_q, padding_skip, ceil_mode=False)
        else:
            self.pool_skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        x_block = self.attn(x_norm)

        if self.proj is not None:
            x = self.proj(x_norm)

        if self.pool_skip is not None:
            x = attention_pool(x, self.pool_skip)

        x = x + self.drop_path(x_block)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
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


class MViTv2(Backbone):
    """
    Support Multi-Scale Vision Transformer v2 (MViTv2) backbone (https://arxiv.org/abs/2104.11227).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depths: List[int],
        drop_path_rate: float,
        patch_kernel: Union[int, Tuple[int, int]] = 7,
        patch_stride: Union[int, Tuple[int, int]] = 4,
        patch_padding: Union[int, Tuple[int, int]] = 3,
        input_shape: Optional[ShapeSpec] = None,
        qkv_pool_kernel: Union[int, Tuple[int, int]] = 3,
        adaptive_kv_stride: int = 4,
        adaptive_window_size: int = 56,
        residual_pooling: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm: Union[str, Dict] = "LN",
        activation: Union[str, Dict] = "GELU",
        use_abs_pos: bool = False,
        use_rel_pos: bool = True,
        rel_pos_zero_init: bool = True,
        use_checkpoint: bool = False,
        img_size: Union[int, Tuple[int, int]] = 224,
        pretrain_img_size: Union[int, Tuple[int, int]] = 224,
        pretrain_use_cls_token: bool = True,
        out_features: Optional[List[str]] = None,
        freeze_at: int = -1,
    ):
        """
        Args:
            embed_dim (int): patch embedding dimension.
            num_heads (int): mumber of base attention heads in each MViT block.
            depths (list[int]): depth of each stage.
            drop_path_rate (float): stochastic depth rate.
            patch_kernel (int or tuple): kernel size for patch embedding.
            patch_stride (int or tuple): stride size for patch embedding.
            patch_padding (int or tuple): padding size for patch embedding.
            input_shape (ShapeSpec): input shape.
            qkv_pool_kernel (int or tuple): kernel size for qkv pooling layers.
            adaptive_kv_stride (int): stride size for kv pooling layers in adaptive window attention.
            adaptive_window_size (int): adaptive window size for window attention blocks.
            residual_pooling (bool): If true, enable residual pooling.
            mlp_ratio (float): ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm (str, dict): Normalization layer.
            activation (str, dict): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embedding.
            use_rel_pos (bool): If True, add relative postional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            use_checkpoint (bool): If True, use checkpointing to save memory.
            img_size (int or tuple): input image size.
            pretrain_img_size (int or tuple): input image size for pretraining.
            pretrain_use_cls_token (bool): If True, use cls token for pretraining.
            out_features (list[str]): name of the layers whose outputs should be returned in forward.
            freeze_at (int): Stages to be frozen (stop grad and set eval mode).
                -1 means not freezing any parameters.
        """
        super().__init__()

        if input_shape is None:
            input_shape = ShapeSpec(channels=3)

        patch_stride = _pair(patch_stride)
        pretrain_img_size = _pair(pretrain_img_size)
        img_size = _pair(img_size)

        self.patch_embed = PatchEmbed(
            in_channels=input_shape.channels,
            embed_dim=embed_dim,
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
        )

        if use_abs_pos:
            # Initialize absoluate positional embedding with pretrain image size.
            num_patches = (pretrain_img_size[0] // patch_stride[0]) * (pretrain_img_size[1] // patch_stride[1])
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        dim_out = embed_dim
        stride_kv = adaptive_kv_stride
        window_size = adaptive_window_size
        input_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        stride = patch_stride[0]

        self.use_checkpoint = use_checkpoint
        self.pretrain_use_cls_token = pretrain_use_cls_token

        if out_features is not None:
            num_stages = max([{"stage2": 1, "stage3": 2, "stage4": 3, "stage5": 4}.get(f, 0) for f in out_features])
            depths = depths[:num_stages]

        self.stages = []
        for idx, num_block in enumerate(depths):
            name = f"stage{idx + 2}"
            stage = []
            window_size_ = 0 if idx > 1 else window_size
            for i in range(num_block):
                # Multiply stride_kv by 2 if it's the last block of stage2 and stage3.
                if (idx == 1 or idx == 2) and (i == num_block - 1):
                    stride_kv_ = stride_kv * 2
                else:
                    stride_kv_ = stride_kv
                block = MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[sum(depths[:idx]) + i],
                    norm=norm,
                    actiation=activation,
                    qkv_pool_kernel=qkv_pool_kernel,
                    stride_q=2 if (idx > 0 and i == 0) else 1,
                    stride_kv=stride_kv_,
                    residual_pooling=residual_pooling,
                    window_size=window_size_,
                    use_rel_pos=use_rel_pos,
                    rel_pos_zero_init=rel_pos_zero_init,
                    input_size=input_size,
                )
                if use_checkpoint:
                    from fairscale.nn.checkpoint import checkpoint_wrapper

                    block = checkpoint_wrapper(block)

                embed_dim = dim_out
                stage.append(block)

                if idx > 0 and i == 0:
                    window_size = window_size // 2
                    input_size = [s // 2 for s in input_size]

            self.add_module(name, nn.ModuleList(stage))
            if name in out_features:
                self.add_module(f"norm{idx + 2}", get_norm(norm, dim_out))
            self.stages.append(name)

            self._out_feature_channels[name] = dim_out
            self._out_feature_strides[name] = stride

            dim_out *= 2
            num_heads *= 2
            stride_kv = max(stride_kv // 2, 1)
            stride *= 2

        self._out_features = out_features
        assert len(self._out_features)

        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, f"Available children: {', '.join(children)}"

        self.freeze(freeze_at)

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

    @staticmethod
    def build(name, **kwargs):
        predefined = {
            "MViT-v2-tiny": mvit_v2_tiny,
            "MViT-v2-small": mvit_v2_small,
            "MViT-v2-base": mvit_v2_base,
            "MViT-v2-large": mvit_v2_large,
            "MViT-v2-huge": mvit_v2_huge,
        }

        if name not in predefined:
            raise ValueError(f"`{name}` is not predefined for MViT-v2.")

        return predefined[name](**kwargs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, x.shape[1:3])

        outputs = {}

        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            for block in stage:
                x = block(x)

            if stage_name in self._out_features:
                x_out = getattr(self, f"norm{i + 2}")(x)
                outputs[stage_name] = x_out.permute(0, 3, 1, 2)

        return outputs

    def freeze(self, freeze_at: int = -1):
        if freeze_at >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if freeze_at >= 1 and self.pos_embed is not None:
            self.pos_embed.requires_grad = False

        if freeze_at >= 2:
            for i in range(0, freeze_at - 1):
                stage = getattr(self, self.stages[i])
                stage.eval()
                for param in stage.parameters():
                    param.requires_grad = False


def mvit_v2_tiny(**kwargs):
    return MViTv2(
        embed_dim=96,
        num_heads=1,
        depths=[1, 2, 5, 2],
        drop_path_rate=0.2,
        **kwargs,
    )


def mvit_v2_small(**kwargs):
    return MViTv2(
        embed_dim=96,
        num_heads=1,
        depths=[1, 2, 11, 2],
        drop_path_rate=0.2,
        **kwargs,
    )


def mvit_v2_base(**kwargs):
    return MViTv2(
        embed_dim=96,
        num_heads=1,
        depths=[2, 3, 16, 3],
        drop_path_rate=0.4,
        **kwargs,
    )


def mvit_v2_large(**kwargs):
    return MViTv2(
        embed_dim=144,
        num_heads=2,
        depths=[2, 6, 36, 4],
        drop_path_rate=0.5,
        **kwargs,
    )


def mvit_v2_huge(**kwargs):
    return MViTv2(
        embed_dim=192,
        num_heads=3,
        depths=[4, 8, 60, 8],
        drop_path_rate=0.6,
        **kwargs,
    )
