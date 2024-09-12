import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from masks import get_A_mask, get_h2o_mask, get_lambda_mask_sparse, get_eviction_kv_indices

'''
    Baseline implementation from LESS augmented with several extra gate-decay features.
    Currently works with:
    - constant lambda gating
    - learned constant lambda gating per layer
    - learned constant lambda gating per layer and attention head
    - time-dependent lambda gating
    - time-data-dependent lambda gating, dependent on FLA (in progress)
'''

class KernelizedHeadAttention(nn.Module):
    def __init__(self, dim_head, dim_hid, dim_ker, num_heads, dropout, multi_query=False, lambda_gating=None):
        super().__init__()
        self.dim_ker = dim_ker
        self.num_heads = num_heads
        self.dim_hid = dim_hid
        self.dim_head = dim_head
        self.lambda_gating = lambda_gating
        self.multi_query = multi_query
        print(f"Number of heads: {self.num_heads}")
        print(f"Hidden dimension: {self.dim_hid}")
        print(f"Head dimension: {self.dim_head}")
        print(f"Kernel dimension: {self.dim_ker}")
        
        # MLP Layer 1
        a = math.sqrt(6/(dim_head + dim_hid))
        self.kernel_q_mat1 = torch.nn.init.uniform_(torch.empty(num_heads, dim_head, dim_hid), a=-a, b=a)
        self.kernel_q_mat1 = nn.Parameter(self.kernel_q_mat1, requires_grad=True)
        if not self.multi_query:
            self.kernel_k_mat1 = torch.nn.init.uniform_(torch.empty(num_heads, dim_head, dim_hid), a=-a, b=a)
            self.kernel_k_mat1 = nn.Parameter(self.kernel_k_mat1, requires_grad=True)
        else:
            self.kernel_k_mat1 = nn.Linear(dim_head, dim_hid, bias=False)
        
        # MLP Layer 2
        a = math.sqrt(6/(dim_ker + dim_hid))
        self.kernel_q_mat2 = torch.nn.init.uniform_(torch.empty(num_heads, dim_hid, dim_ker), a=-a, b=a)
        self.kernel_q_mat2 = nn.Parameter(self.kernel_q_mat2, requires_grad=True)
        if not self.multi_query:
            self.kernel_k_mat2 = torch.nn.init.uniform_(torch.empty(num_heads, dim_hid, dim_ker), a=-a, b=a)
            self.kernel_k_mat2 = nn.Parameter(self.kernel_k_mat2, requires_grad=True)
        else:
            self.kernel_k_mat2 = nn.Linear(dim_hid, dim_ker, bias=False)
        

        # MLP Layer 3 (keys only)
        a = math.sqrt(6/(2 * dim_ker))
        if not self.multi_query:
            self.interaction_k = torch.nn.init.uniform_(torch.empty(num_heads, dim_ker, dim_ker), a=-a, b=a)
            self.interaction_k = nn.Parameter(self.interaction_k, requires_grad=True)
        else:
            self.interaction_k = nn.Linear(dim_ker, dim_ker, bias=False)
        
        
        if self.multi_query:
            num_heads = 1
        self.scalingD = nn.Parameter(torch.ones(1, num_heads, 1, dim_ker) * 1e-4, requires_grad=True)
        self.scalingD2 = nn.Parameter(torch.ones(1, num_heads, 1, dim_ker) * 1e-4, requires_grad=True)

        # initialize at 5.0 for all learned lambdas that are still time-independent
        # initialization at 5.0 results in decay post-sigmoid of around 0.99 without being too deep
        # into extremely low gradient territory, seems neutral enough
        if self.lambda_gating == "learned-constant":
            self.W_lambda = nn.Parameter(5.0 * torch.ones(1), requires_grad=True)
        elif self.lambda_gating == "learned-constant-head":
            assert not self.multi_query, "Multi-query attention (MQA) not supported by per-head lambdas, defeats the purpose of MQA."
            self.W_lambda = nn.Parameter(5.0 * torch.ones(self.num_heads), requires_grad=True)
        elif self.lambda_gating == "time-dependent":
            # based on Mamba-2 formulation of G_t
            # init alpha to large negative value for slow start
            self.alpha = nn.Parameter(-10.0 * torch.ones(1), requires_grad=True)
            self.W_lambda = nn.Linear(self.num_heads * self.dim_head, 1, bias=False)
        elif self.lambda_gating == "time-data-dependent":
            try:
                import fla
            except ImportError:
                raise ImportError("Time and data-dependent lambda gating requires FLA to be installed for " \
                                  + "efficient training of expanded, recurrent form.")

            # low rank W_lambda set up with hardcoded constant based on Gated Linear Attention
            # NOTE: this can be implemented per-head with a rearrange step or just per layer, we
            #       are implement per head following GLA
            if self.multi_query:
                self.W_lambda = nn.Sequential(
                    nn.Linear(self.num_heads * self.dim_head, 16, bias=False),
                    nn.Linear(16, self.dim_ker, bias=False)
                )
            else:
                self.W_lambda = nn.Sequential(
                    nn.Linear(self.num_heads * self.dim_head, 16, bias=False),
                    nn.Linear(16, self.num_heads * self.dim_ker, bias=False)
                )

        self.multi_query = multi_query
        self.act = F.gelu
        self.kernel_f = F.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t, q, k, v, lr_attn_mask, sparse_norms_lse, sparse_attn_weights, lambda_constant, half_precision=False):
        B, S, D = q.shape
        H = self.num_heads

        # addition of lambda gating-based decay for sparse mask
        decay_mask = None
        if self.lambda_gating == "constant":
            decay_mask = get_lambda_mask_sparse(lr_attn_mask, lambda_constant)
        elif self.lambda_gating == "learned-constant" or self.lambda_gating == "learned-constant-head":
            decay_mask = get_lambda_mask_sparse(lr_attn_mask, F.sigmoid(self.W_lambda))
        elif self.lambda_gating == "time-dependent":
            # here, self.W_lambda is a nn.Linear instead of being parameters we apply sigmoid to
            lambda_t = torch.exp(-1.0 * F.softplus(self.W_lambda(x_t)) * torch.exp(self.alpha))
            decay_mask = get_lambda_mask_sparse(lr_attn_mask, lambda_t)
        elif self.lambda_gating == "time-data-dependent":
            # implementation is identical to GLA, but they never exponentiate out of
            # logarithmic space for some reason?
            lambda_t_data = F.logsigmoid(self.W_lambda(x_t)) / 16

        q = q.reshape(B, S, self.num_heads, D // self.num_heads).transpose(1, 2)
        if not self.multi_query:
            k = k.reshape(B, S, self.num_heads, D // self.num_heads).transpose(1, 2)
            v = v.reshape(B, S, self.num_heads, D // self.num_heads).transpose(1, 2)
        else:
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
        
        q = self.act(self.dropout(torch.einsum('bhsd,hde->bhse', q, self.kernel_q_mat1)))
        q = self.kernel_f(torch.einsum('bhsd,hde->bhse', q, self.kernel_q_mat2))
        
        if not self.multi_query:
            k = self.act(self.dropout(torch.einsum('bhsd,hde->bhse', k, self.kernel_k_mat1)))
            k = torch.abs(self.scalingD) * self.kernel_f(torch.einsum('bhsd,hde->bhse', k, self.kernel_k_mat2))
            k = k + torch.einsum('bhsd,hde->bhse', k, self.interaction_k) * self.scalingD2
        else:
            k = self.act(self.dropout(self.kernel_k_mat1(k)))
            k = torch.abs(self.scalingD) * self.kernel_f(self.kernel_k_mat2(k))
            k = k + self.interaction_k(k) * self.scalingD2

        # ALTERNATE forward path to accomodate unique needs for FLA kernels and
        # non-QK^T manifestation for time-data-dependent lambda gating use
        if self.lambda_gating == "time-data-dependent":
            # weird kernel function from LESS, no abs until right now
            q = q.abs()
            k = k.abs()

            eviction_kv_indices = get_eviction_kv_indices(lr_attn_mask)

            out_linear, norm, offset = self.time_data_dep_forward(
                q=q,
                k=k,
                v=v,
                lambda_t_data=lambda_t_data,
                eviction_kv_indices=eviction_kv_indices,
                half_precision=half_precision,
                fused_recurrent_override=True,
            )
         
            lr_norms_lse = torch.log(norm + 1e-6)
            lr_norms_lse = torch.cat((float('-inf') * torch.ones(B, H, offset, 1, device=lr_norms_lse.device), lr_norms_lse), -2)
            norm_factor_lse = torch.logaddexp(lr_norms_lse, sparse_norms_lse)
          
            out_linear = torch.cat((float('-inf') * torch.ones(B, H, offset, D // H, device=out_linear.device), out_linear), -2)

            # order shouldn't matter here, we should be able to construct post-matmul with v output before div
            out = torch.matmul(~lr_attn_mask * sparse_attn_weights, v) + out_linear
            out = torch.exp(out - norm_factor_lse)
            out = out.transpose(1, 2).flatten(2)

        # STANDARD forward path, also uses decay_mask logic when applicable
        else:
            out = torch.matmul(q.abs(), k.abs().transpose(2, 3)) # B, H, S, S
            
            if decay_mask is not None:
                out = decay_mask * out
            else:
                out = lr_attn_mask * out

            lr_norms_lse = torch.log(out.sum(dim=-1, keepdim=True) + 1e-6)
            norm_factor_lse = torch.logaddexp(lr_norms_lse, sparse_norms_lse)
            out = (torch.log(out + 1e-6) * lr_attn_mask) + ((~lr_attn_mask) * sparse_attn_weights)
            out = torch.exp(out - norm_factor_lse)
            out = torch.matmul(out, v).transpose(1, 2).flatten(2)

        return out
    

    def time_data_dep_forward(self, q, k, v, lambda_t_data, eviction_kv_indices, half_precision=False, fused_recurrent_override=False):
        B, H, S_q, D = q.shape
        _, H_k, S_k, D_k = k.shape
        _, _, _, D_v = v.shape 

        assert eviction_kv_indices.size(-1) % H_k == 0, \
            "Even division of triggered token length by attn. heads required. "        + \
            "Eviction decisions are not allowed to be dependent on attn. heads, "      + \
            "i.e. the same number of tokens must be evicted per-head. Non-factor for " + \
            "multi-query attention."

        # B x (S_reduct. * H_k) => B x S_reduct. x H_k, where S_reduct. also accounts for offset for query
        eviction_kv_indices = eviction_kv_indices.unsqueeze(1).view(B, H_k, eviction_kv_indices.size(-1) // H_k)
        _, _, S_reduct = eviction_kv_indices.shape 

        
        '''
            Necessary to avoid mismatch with reordered keys and values later
            NOTE: This does NOT work for extremely adaptive eviction policies,
                  i.e. ones that might not evict at every time-step. To account
                  for such cases, a secondary tensor containing query time-steps
                  where evictions occurred would be necessary.
        '''
        S_offset = S_q - S_reduct
        q = q[..., S_offset:, :]
        lambda_t_data = lambda_t_data[:, S_offset:, :]

        
        # reorder keys and values according to gathered indices ordered
        # by eviction events, check related test to verify correctness of this 
        # behavior if in doubt
        eviction_k_indices = eviction_kv_indices.unsqueeze(-1).expand(-1, -1, -1, D_k) # B x S_reduct. x H_k => B x S_reduct. x H x D_k_
        eviction_v_indices = eviction_kv_indices.unsqueeze(-1).expand(-1, -1, -1, D_v) # B x S_reduct. x H_k => B x S_reduct. x H x D_k_
        k = k.gather(-2, eviction_k_indices)
        v = v.gather(-2, eviction_v_indices)
        
        if self.multi_query:
            lambda_t_data = lambda_t_data.view(B, 1, S_reduct, D_k)
        else:
            lambda_t_data = lambda_t_data.view(B, H_k, S_reduct, D_k)

        
        # FLA forward pass, using chunk_gla for now due to issues with triton
        try:
            from fla.layers.gla import chunk_gla, fused_recurrent_gla
        except ImportError:
            raise ImportError("Time and data-dependent lambda gating requires FLA to be installed for " \
                              + "efficient training of expanded, recurrent form.")
      
        # weird case where amp doesn't autocast before GLA kernels, where it doesn't catch
        # that it needs to autocast
        if half_precision:
            q = q.half()
            k = k.half()
            v = v.half()
            lambda_t_data = lambda_t_data.half()

        # GLA kernels, not unexpectedly, do not account for GQA or MQA
        if self.multi_query:
            k = k.expand(-1, H, -1, -1)
            v = v.expand(-1, H, -1, -1)
            lambda_t_data = lambda_t_data.expand(-1, H, -1, -1)

        # Triton is slower for single tokens, here for extensibility to inference
        # code later on
        if S_q == 1 or fused_recurrent_override:
            output, _ = fused_recurrent_gla(q, k, v, lambda_t_data, initial_state=None)
        else:
            output, _ = chunk_gla(q, k, v, lambda_t_data, initial_state=None)


        # GLA kernels don't return normalization tensors, as they use LayerNorm instead,
        # so we need to construct them ourselves
        lambda_t_shifted = torch.ones_like(lambda_t_data)
        lambda_t_shifted[:, :, 1:, :] = torch.exp(lambda_t_data[:, :, :-1, :])
        k_for_norm = lambda_t_shifted * k

        # scaling factor included here, GLA kernels do it by default
        k_cum_T = torch.cumsum(k_for_norm, dim=-2).transpose(2, 3)
        norm = torch.matmul(q * D_k ** -0.5, k_cum_T)
        norm = torch.diagonal(norm, dim1=-2, dim2=-1).unsqueeze(-1)

        return output, norm, S_offset
        
