# Adapted from Hugging Face implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import warnings

from transformers.models.falcon.modeling_falcon import (
    FalconLinear,
    FalconRotaryEmbedding, 
    FalconLinearScalingRotaryEmbedding, 
    FalconDynamicNTKScalingRotaryEmbedding, 
    rotate_half
    #apply_rotary_pos_emb,
)

class Annotated_Falcon(nn.Module):
    def __init__(self, model, layers=None):
        super().__init__()
        self.model = model
        self.layers_to_collect = layers

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        int_values = dict()
        for i, l in enumerate(self.model.transformer.h):
            if self.layers_to_collect is not None and i not in self.layers_to_collect:
                continue
            int_values[i] = l.self_attention.int_values
        return output, int_values

    def toggle_layer(self, layer_to_annotate):
        for i, l in enumerate(self.model.transformer.h):
            if self.layers_to_collect is not None and i not in self.layers_to_collect:
                continue
            if i == layer_to_annotate:
                l.self_attention.collect = True
            else:
                l.self_attention.collect = False


def get_annotated_falcon(model, layers=None):
    config = model.config
    for i, l in enumerate(model.transformer.h):
        if layers is not None and i not in layers:
            continue
        new_attn = FalconAttention(config)
        
        new_attn.load_state_dict(l.self_attention.state_dict())
        
        l.self_attention = new_attn
    return Annotated_Falcon(model, layers)


class FalconAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.is_causal = True

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # self.maybe_rotary = self._init_rope() if config.rotary else lambda q, p : q
        self.maybe_rotary = self._init_rope() if config.rotary else lambda q, k, t, p: (q, k)
        if not config.rotary:
            print(f"WARNING: unintended results may stem from not using RoPe in this repository's current state.")

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        if config.new_decoder_architecture:
            qkv_out_dim = (config.num_kv_heads * 2 + config.num_attention_heads) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        self.query_key_value = FalconLinear(self.hidden_size, qkv_out_dim, bias=config.bias)
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        self.dense = FalconLinear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv_heads = config.num_kv_heads if (self.new_decoder_architecture or not self.multi_query) else 1
        self.collect = True

    # adding temporary override for 4.35.2 RoPE for Falcon
    def _init_rope(self):
        if self.config.rope_scaling is None:
            rotary_emb = OldFalconRotaryEmbeddings(
                self.head_dim,
                base=self.config.rope_theta,
                max_position_embeddings=self.config.max_position_embeddings,
            )
            return rotary_emb
        else:
            raise NotImplementedError("Non-RoPE scaling is not properly implemented in this repository's current state. "  + \
                                        "This is essentially all due to needing to version bump to `transformers` v4.44.2 from " + \
                                        "v4.35.2." \
            )

        if self.config.rope_scaling is None:
            rotary_emb = FalconRotaryEmbedding(
                self.head_dim,
                base=self.config.rope_theta,
                max_position_embeddings=self.config.max_position_embeddings,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                rotary_emb = FalconLinearScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.config.rope_theta,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                rotary_emb = FalconDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.config.rope_theta,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
        return rotary_emb

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        if self.new_decoder_architecture:
            batch, seq_len, _ = fused_qkv.shape
            qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]
            key = torch.broadcast_to(key, query.shape)
            value = torch.broadcast_to(value, query.shape)

            query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
            return query, key, value
        elif not self.multi_query:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        else:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads + 2, self.head_dim)
            return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]

    # Copied from transformers.models.bloom.modeling_bloom.BloomAttention._merge_heads
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        int_values = dict()
        if self.collect:
            int_values['x_t'] = hidden_states.cpu().detach()

        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(
            batch_size * num_kv_heads,
            query_length,
            self.head_dim,
        )
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * num_kv_heads, query_length, self.head_dim)

        past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
        
        # updated for transformer version bump
        # kv_seq_len = key_layer.shape[-2]
        # if layer_past is not None:
        #     kv_seq_len += layer_past[0].shape[-2]
        # cos, sin = self.maybe_rotary(value_layer, seq_len=kv_seq_len)
        # query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)
        # query_layer = query_layer.squeeze(0)
        # key_layer = key_layer.squeeze(0)

        query_layer, key_layer = self.maybe_rotary(query_layer, key_layer, past_kv_length, position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, kv_length, head_dim]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, kv_length, _ = key_layer.shape
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, num_kv_heads, -1, self.head_dim)

        if self.collect:
            int_values['Q'] = query_layer_.cpu().detach()
            int_values['K'] = key_layer_.cpu().detach()
            int_values['V'] = value_layer_.cpu().detach()

        if alibi is None:
            # override, do NOT use sdpa from torch
            if not output_attentions:
                # TODO: deprecate this once we add FA2 support in Falcon
                # logger.warning_once(
                #     "The current implementation of Falcon calls `torch.scaled_dot_product_attention` directly, this will be deprecated in the"
                #     " future in favor of the `BetterTransformer` API. Please install the latest optimum library with `pip install -U optimum` and call "
                #     "`model.to_bettertransformer()` to benefit from `torch.scaled_dot_product_attention` and future performance optimizations."
                # )

                attn_output = F.scaled_dot_product_attention(
                    query_layer_, key_layer_, value_layer_, attention_mask, 0.0, is_causal=self.is_causal and attention_mask is None and query_length > 1
                )
                attention_scores = None
            else:
                attention_scores = query_layer_ @ key_layer_.transpose(-1, -2)
                attention_scores /= math.sqrt(self.head_dim)

                if attention_mask is None:
                    attention_mask = torch.triu(torch.ones((1, 1, query_length, query_length)), diagonal=1).to(attention_scores.device) * -1e3

                attention_scores = F.softmax(attention_scores + attention_mask, dim=-1, dtype=hidden_states.dtype)
                attn_output = attention_scores @ value_layer_

            attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

            output_tensor = self.dense(attn_output)

            if self.collect:
                int_values['O'] = output_tensor.cpu().detach()

            self.int_values = int_values

            if output_attentions:
                return output_tensor, present, attention_scores
            else:
                return output_tensor, present

        else:
            matmul_result = query_layer_ @ key_layer_.transpose(-1, -2)

            # change view to [batch_size, num_heads, q_length, kv_length]
            attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)

            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            input_dtype = attention_scores.dtype
            # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                attention_scores = attention_scores.to(torch.float32)
            # Matt (HF) note: We could possibly use F.scaled_dot_product_attention here too, by
            # adding (alibi * self.inv_norm_factor) to attention_mask. I think this would be mathematically
            # equivalent and more performant, but there might be a numerical difference. If you're reading this
            # and you'd like to experiment and maybe file a PR, feel free!
            attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)
            attention_logits *= self.inv_norm_factor
            attention_probs = F.softmax(attention_logits + attention_mask, dim=-1, dtype=hidden_states.dtype)
            # [batch_size, num_heads, q_length, kv_length]
            attention_probs = self.attention_dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # change view [batch_size, num_heads, q_length, kv_length]
            attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)

            # matmul: [batch_size * num_heads, q_length, head_dim]
            context_layer = (attention_probs_reshaped @ value_layer_).flatten(0, 1)

            # change view [batch_size, q_length, num_heads * head_dim]
            context_layer = self._merge_heads(context_layer)

            output_tensor = self.dense(context_layer)
            if self.collect:
                int_values['O'] = output_tensor.cpu().detach()
            
            self.int_values = int_values

            if output_attentions:
                return output_tensor, present, attention_probs
            else:
                return output_tensor, present

class OldFalconRotaryEmbeddings(nn.Module):
    """Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is designed to operate on queries and keys that are compatible with `[batch_size,
    n_heads_per_partition, seq_len, head_dim]` (e.g. MinGPTAttention format).
    """

    def __init__(self, head_dim: int, base=10000, max_position_embeddings=2048):
        super().__init__()
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.seq_len_cached = -1
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device).to(dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)

        if dtype in [torch.float16, torch.bfloat16]:
            emb = emb.float()

        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

        self.cos_cached = self.cos_cached.type(dtype)
        self.sin_cached = self.sin_cached.type(dtype)

    def cos_sin(
        self, seq_len: int, past_key_values_length: int, position_ids: torch.Tensor, device="cpu", dtype=torch.bfloat16
    ) -> torch.Tensor:
        total_length = seq_len + past_key_values_length
        if total_length > self.seq_len_cached:
            self._set_cos_sin_cache(total_length, device, dtype)

        # the cached tensors need to update their devices (for example, after we change the model's device)
        self.cos_cached = self.cos_cached.to(device)
        self.sin_cached = self.sin_cached.to(device)

        # Gather cos, sin at the designated position ids
        cos = self.cos_cached[position_ids]  # [bs, seq_len, dim]
        sin = self.sin_cached[position_ids]  # [bs, seq_len, dim]
        return cos, sin

    def forward(self, query, key, past_key_values_length, position_ids):
        _, seq_len, _ = query.shape
        cos, sin = self.cos_sin(seq_len, past_key_values_length, position_ids, query.device, query.dtype)
        # Query and key's shapes are [bs * num_heads, seq_len, dim], might need manual expansion. Ifs and elses used to
        # avoid unnecessary repeat_interleave operations.
        query_expansion_factor = int(query.shape[0] / cos.shape[0])
        if query_expansion_factor > 1:
            query_cos = torch.repeat_interleave(cos, query_expansion_factor, dim=0)
            query_sin = torch.repeat_interleave(sin, query_expansion_factor, dim=0)
        else:
            query_cos, query_sin = cos, sin

        key_expansion_factor = int(key.shape[0] / cos.shape[0])
        if key_expansion_factor > 1:
            if key_expansion_factor != query_expansion_factor:
                key_cos = torch.repeat_interleave(cos, key_expansion_factor, dim=0)
                key_sin = torch.repeat_interleave(sin, key_expansion_factor, dim=0)
            else:
                key_cos, key_sin = query_cos, query_sin
        else:
            key_cos, key_sin = cos, sin

        return (query * query_cos) + (rotate_half(query) * query_sin), (key * key_cos) + (rotate_half(key) * key_sin)
