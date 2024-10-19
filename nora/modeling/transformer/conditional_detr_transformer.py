from typing import Dict
from typing import List
from typing import Union

import torch
import torch.nn as nn

from nora.layers import ConditionalCrossAttention
from nora.layers import ConditionalSelfAttention
from nora.layers import FFN
from nora.layers import MLP
from nora.layers import get_norm
from nora.layers import coordinate_to_embedding
from .detr_transformer import DETRTransformerEncoder

__all__ = [
    "ConditionalDETRTransformer",
    "ConditionalDETRTransformerDecoder",
    "ConditionalDETRTransformerEncoder",
]


class ConditionalDETRTransformerEncoder(DETRTransformerEncoder):
    """
    Conditional uses the same encoder as DETR
    """
    pass


class ConditionalDETRTransformerDecoderLayer(nn.Module):
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
        is_first: bool = False,
    ):
        super().__init__()

        self.self_attn = ConditionalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_dropout,
            batch_first=batch_first,
        )
        self.norm1 = get_norm(norm, embed_dim)
        self.cross_attn = ConditionalCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_dropout,
            batch_first=batch_first,
            is_first=is_first,
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
        query,
        key,
        query_pos,
        key_pos,
        key_padding_mask,
        query_sine_embed,
        is_first,
    ):
        query = self.self_attn(
            query,
            key=query,
            query_pos=query_pos,
            key_pos=query_pos,
        )
        query = self.norm1(query)
        query = self.cross_attn(
            query,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
            key_padding_mask=key_padding_mask,
            query_sine_embed=query_sine_embed,
            is_first=is_first,
        )
        query = self.norm2(query)
        query = self.ffn(query)
        query = self.norm3(query)
        return query


class ConditionalDETRTransformerDecoder(nn.Module):
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
            ConditionalDETRTransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                feedforward_channels=feedforward_channels,
                ffn_dropout=ffn_dropout,
                norm=norm,
                activation=activation,
                batch_first=batch_first,
                is_first=(i == 0),
            )
            for i in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.post_norm = get_norm(post_norm, embed_dim)
        self.query_scale = MLP(
            in_features=self.embed_dim,
            hidden_features=self.embed_dim,
            out_features=self.embed_dim,
            activation=activation,
        )
        self.ref_point_head = MLP(
            in_features=self.embed_dim,
            hidden_features=self.embed_dim,
            out_features=2,
            activation=activation,
        )

    def forward(
        self,
        query,
        key,
        query_pos,
        key_pos,
        key_padding_mask,
    ):
        # [num_queries, N, 2]
        reference_unsigmoid = self.ref_point_head(query_pos)
        reference = reference_unsigmoid.sigmoid()
        reference_xy = reference[..., :2]
        intermediate = []
        for idx, layer in enumerate(self.layers):
            is_first = idx == 0

            if is_first:
                position_transform = 1
            else:
                position_transform = self.query_scale(query)

            # get sine embedding for the query vector
            query_sine_embed = coordinate_to_embedding(reference_xy)
            # apply position transform
            query_sine_embed = query_sine_embed[..., :self.embed_dim] * position_transform

            query = layer(
                query,
                key=key,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                query_sine_embed=query_sine_embed,
                is_first=is_first,
            )

            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate), reference

        if self.post_norm is not None:
            query = self.post_norm(query)

        return query.unsqueeze(0), reference


class ConditionalDETRTransformer(nn.Module):
    def __init__(
        self,
        in_features: List[str],
        encoder: ConditionalDETRTransformerEncoder,
        decoder: ConditionalDETRTransformerDecoder,
    ):
        super().__init__()

        assert len(in_features) == 1, f"Conditional DETR only uses single level feature, but got len(in_features)={len(in_features)}`"

        self.encoder = encoder
        self.decoder = decoder

        self.embed_dim = self.encoder.embed_dim
        self.in_features = in_features

        for coder in [self.encoder, self.decoder]:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, features, mask, query_embed, pos_embed):
        x = features[self.in_features[0]]

        N, C, H, W = x.shape
        x = x.view(N, C, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(N, C, -1).permute(0, 2, 1)
        query_embed = query_embed.unsqueeze(0).repeat(N, 1, 1)
        mask = mask.view(N, -1)

        memory = self.encoder(
            query=x,
            query_pos=pos_embed,
            key_padding_mask=mask,
        )
        target = torch.zeros_like(query_embed)
        hidden_states, references = self.decoder(
            query=target,
            key=memory,
            query_pos=query_embed,
            key_pos=pos_embed,
            key_padding_mask=mask,
        )
        return hidden_states, references
