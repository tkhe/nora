from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn

from nora.layers import FFN
from nora.layers import MultiHeadAttention
from nora.layers import get_norm

__all__ = [
    "DETRTransformer",
    "DETRTransformerDecoder",
    "DETRTransformerEncoder",
]


class DETRTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        norm: Union[str, Dict] = "LN",
        activation: Union[str, Dict] = "ReLU",
        batch_first: bool = True,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_dropout,
            batch_first=batch_first,
        )
        self.norm1 = get_norm(norm, embed_dim)
        self.ffn = FFN(
            in_features=embed_dim,
            hidden_features=feedforward_channels,
            out_features=embed_dim,
            activation=activation,
            drop=ffn_dropout,
        )
        self.norm2 = get_norm(norm, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ):
        query = self.attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
        )
        query = self.norm1(query)
        query = self.ffn(query)
        query = self.norm2(query)
        return query


class DETRTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        norm: Union[str, Dict] = "LN",
        activation: Union[str, Dict] = "ReLU",
        batch_first: bool = True,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_dropout,
            batch_first=batch_first,
        )
        self.norm1 = get_norm(norm, embed_dim)
        self.cross_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_dropout,
            batch_first=batch_first,
        )
        self.norm2 = get_norm(norm, embed_dim)
        self.ffn = FFN(
            in_features=embed_dim,
            hidden_features=feedforward_channels,
            out_features=embed_dim,
            activation=activation,
            drop=ffn_dropout,
        )
        self.norm3 = get_norm(norm, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
        )
        query = self.norm1(query)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
        )
        query = self.norm2(query)
        query = self.ffn(query)
        query = self.norm3(query)
        return query


class DETRTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        norm: Union[str, Dict] = "LN",
        activation: Union[str, Dict] = "ReLU",
        batch_first: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        layers = [
            DETRTransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                feedforward_channels=feedforward_channels,
                ffn_dropout=ffn_dropout,
                norm=norm,
                activation=activation,
                batch_first=batch_first,
            )
            for _ in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ):
        for layer in self.layers:
            query = layer(query, query_pos, key_padding_mask)
        return query


class DETRTransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        feedforward_channels: int = 1024,
        ffn_dropout: float = 0.0,
        norm: Union[str, Dict] = "LN",
        activation: Union[str, Dict] = "ReLU",
        batch_first: bool = True,
        return_intermediate: bool = True,
        post_norm: Union[str, Dict] = "LN",
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.return_intermediate = return_intermediate

        layers = [
            DETRTransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                feedforward_channels=feedforward_channels,
                ffn_dropout=ffn_dropout,
                norm=norm,
                activation=activation,
                batch_first=batch_first,
            )
            for _ in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.post_norm = get_norm(post_norm, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ):
        if self.return_intermediate:
            intermediate = []
            for layer in self.layers:
                query = layer(
                    query,
                    key=key,
                    value=value,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    key_padding_mask=key_padding_mask,
                )
                if self.return_intermediate:
                    if self.post_norm is not None:
                        intermediate.append(self.post_norm(query))
                    else:
                        intermediate.append(query)

            return torch.stack(intermediate)
        else:
            for layer in self.layers:
                query = layer(
                    query,
                    key=key,
                    value=value,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    key_padding_mask=key_padding_mask,
                )

            if self.post_norm is not None:
                query = self.post_norm(query)

            return query


class DETRTransformer(nn.Module):
    def __init__(
        self,
        in_features: List[str],
        encoder: DETRTransformerEncoder,
        decoder: DETRTransformerDecoder,
    ):
        super().__init__()

        assert len(in_features) == 1, f"DETR only use single level feature, but got len(in_features)={len(in_features)}`"

        self.encoder = encoder
        self.decoder = decoder

        self.embed_dim = self.encoder.embed_dim
        self.in_features = in_features

    def forward(self, features, mask, query_embed, pos_embed):
        x = features[self.in_features[0]]

        N, C, H, W = x.shape
        # [N, C, H, W] -> [N, HxW, C]
        x = x.view(N, C, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(N, C, -1).permute(0, 2, 1)
        query_embed = query_embed.unsqueeze(0).repeat(N, 1, 1)  # [num_query, dim] -> [N, num_query, dim]
        mask = mask.view(N, -1)  # [N, H, W] -> [N, HxW]
        memory = self.encoder(
            query=x,
            query_pos=pos_embed,
            key_padding_mask=mask,
        )
        target = torch.zeros_like(query_embed)
        decoder_output = self.decoder(
            query=target,
            key=memory,
            value=memory,
            query_pos=query_embed,
            key_pos=pos_embed,
            key_padding_mask=mask,
        )
        memory = memory.permute(0, 2, 1).reshape(N, C, H, W)
        return decoder_output, memory
