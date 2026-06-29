import torch
from torch import nn
from typing import Optional, Tuple, List
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
import torch.nn.functional as F
from quantize.du_norm import DuQwen3RMSNorm

from collections import OrderedDict
import math

import pdb
import copy
from models.transformation import *

from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding,apply_rotary_pos_emb,Qwen3RMSNorm,repeat_kv
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
class QuantQwen3MLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        args=None,
    ):
        super().__init__()
        
        self.gate_proj = QuantLinear(org_module.gate_proj,
                                           args.gate_weight_quant_params,
                                           args.gate_act_quant_params)
        self.down_proj = QuantLinear(org_module.down_proj,
                                           args.down_weight_quant_params,
                                           args.down_act_quant_params)
        self.up_proj = QuantLinear(org_module.up_proj,
                                           args.up_weight_quant_params,
                                           args.up_act_quant_params)
        self.act_fn = ACT2FN[hidden_act]
        self.init_duquant_params = torch.tensor(0) if args.gate_weight_quant_params['quant_method'] == 'duquant' else torch.tensor(1)

    def forward(self, x):
        if not self.init_duquant_params:
            self.init_duquant_params = torch.tensor(1)
            act = self.act_fn(self.gate_proj(x))
            self.up_proj.copy_quantizers_duquant_params(self.gate_proj)
            mul = act * self.up_proj(x)
            return self.down_proj(mul)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class QuantQwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, org_module: nn.Module, config: Qwen3Config, layer_idx: int ,args = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.k_proj = QuantLinear(
            org_module.k_proj,
            args.k_weight_quant_params,
            args.k_act_quant_params,
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
            args.v_weight_quant_params,
            args.v_act_quant_params,
        )
        self.q_proj = QuantLinear(
            org_module.q_proj,
            args.q_weight_quant_params,
            args.q_act_quant_params,
        )
        self.o_proj = QuantLinear(
            org_module.o_proj, args.o_weight_quant_params, args.o_act_quant_params
        )
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul, rotate=None
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul, rotate=None
        )

        self.q_norm = DuQwen3RMSNorm(org_module.q_norm,eps=org_module.q_norm.variance_epsilon)
        self.k_norm = DuQwen3RMSNorm(org_module.k_norm,eps=org_module.k_norm.variance_epsilon)
        self.sliding_window = org_module.sliding_window

        self.use_weight_quant = False
        self.use_act_quant = False


        self.init_duquant_params = torch.tensor(0) if args.gate_weight_quant_params['quant_method'] == 'duquant' else torch.tensor(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim) # [bsz,token,num_head,head_dim]

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2) # [bsz, num_head, token, head_dim]
        if not self.init_duquant_params:
            self.k_proj.copy_quantizers_duquant_params(self.q_proj)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2) # [bsz,num_head,token,head_dim]
        if not self.init_duquant_params:
            self.v_proj.copy_quantizers_duquant_params(self.q_proj)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        kv_seq_len = key_states.shape[-2] # token

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            kv_seq_len += past_key_value[0].shape[-2]
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        past_key_value = (key_states, value_states) if use_cache else None
        
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) # [bsz,num_head,token,head_dim] @ [bsz,num_head,head_dim,kv_len] => [bsz,num_head,token,kv_len]


        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) # [bsz,num_head,q_len,kv_len]
        attn_weights = self.pv_matmul.quant_x1(attn_weights) # [bsz,num_head,q_len,kv_len]
        value_states = self.pv_matmul.quant_x2(value_states) # [bsz,num_head,kv_len,head_dim]
        attn_output = self.pv_matmul(attn_weights, value_states) # [bsz, num_head, q_len, kv_len] @ [bsz, num_head,kv_len, head_dim] => [bsz, num_head,q_len,head_dim]

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2) # [bsz,q_len,num_head,head_dim]
        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)

        attn_output = self.o_proj(attn_output)

        self.init_duquant_params = torch.tensor(1)

        return attn_output, attn_weights

    

class QuantQwen3DecoderLayer(nn.Module):
    def __init__(self, 
                 config: Qwen3Config,
                 ori_layer,
                 layer_idx : int,
                 args):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantQwen3Attention(
            org_module=ori_layer.self_attn,
            config=config,
            layer_idx=layer_idx,
            args=args,
            )
        self.mlp = QuantQwen3MLP(
            org_module=ori_layer.mlp,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            args=args,
        )
        self.input_layernorm = DuQwen3RMSNorm(ori_layer.input_layernorm,eps=ori_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = DuQwen3RMSNorm(ori_layer.post_attention_layernorm,eps=ori_layer.post_attention_layernorm.variance_epsilon)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC)
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states.to(self.mlp.up_proj.weight.device)).to(residual.device)
        hidden_states = residual + hidden_states
        return hidden_states

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)

    def smooth_and_quant_temporary(self):
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_temporary(self.input_layernorm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_temporary(self.post_attention_layernorm,[self.mlp.up_proj,self.mlp.gate_proj],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_temporary(self.self_attn.v_proj,self.self_attn.o_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_temporary(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
            self.mlp.down_proj.temp_weight = self.mlp.down_proj.weight
        else:
            for name, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight
        # quant
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter=True

    def clear_temp_variable(self):
       for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
            smooth_ln_fcs_inplace(self.input_layernorm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_inplace(self.post_attention_layernorm,[self.mlp.up_proj,self.mlp.gate_proj],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_inplace(self.self_attn.v_proj,self.self_attn.o_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter=False

    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)  

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)  

    def duquant_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)  
    
    def duquant_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination
    
    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()
    
    def register_duquant_params(self):        
        for name, module in self.named_modules():
            if isinstance(module, QuantQwen3MLP) or isinstance(module, QuantQwen3Attention):
                delattr(module, 'init_duquant_params')
                module.register_buffer('init_duquant_params', torch.tensor(1))
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_duquant_params()
                module.act_quantizer.register_duquant_params()
    
    def load_duquant_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('R') > -1 or k.find('permutation_list') > -1 or k.find('init_duquant_params') > -1:
                exec(f'self.{k} = v.to(device)')
    
    def load_smooth_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('smooth') > -1:
                # exec(f'self.{k} = v')
                self.register_parameter(k, torch.nn.Parameter(v.to(device), requires_grad=False))
    
    def load_post_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('post') > -1:
                # exec(f'self.{k} = v')
                rg = False if k.find('down') > -1 else True
                self.register_parameter(k, torch.nn.Parameter(v.to(device), requires_grad=rg))

    def load_lwc_params(self, state_dict, device):
        for k, v in state_dict.items():
            if k.find('bound_factor') > -1:
                v = torch.nn.Parameter(v.to(device))
                exec(f'self.{k} = v.to(device)')