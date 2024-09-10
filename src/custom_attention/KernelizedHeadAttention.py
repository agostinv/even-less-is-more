import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from masks import get_A_mask, get_h2o_mask, get_lambda_mask_sparse

class KernelizedHeadAttention(nn.Module):
    def __init__(self, dim_head, dim_hid, dim_ker, num_heads, dropout, multi_query=False, lambda_gating=None):
        super().__init__()
        self.dim_ker = dim_ker
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.lambda_gating = lambda_gating
        print(f"Number of heads: {self.num_heads}")
        
        # MLP Layer 1
        a = math.sqrt(6/(dim_head + dim_hid))
        self.kernel_q_mat1 = torch.nn.init.uniform_(torch.empty(num_heads, dim_head, dim_hid), a=-a, b=a)
        self.kernel_q_mat1 = nn.Parameter(self.kernel_q_mat1, requires_grad=True)
        if not multi_query:
            self.kernel_k_mat1 = torch.nn.init.uniform_(torch.empty(num_heads, dim_head, dim_hid), a=-a, b=a)
            self.kernel_k_mat1 = nn.Parameter(self.kernel_k_mat1, requires_grad=True)
        else:
            self.kernel_k_mat1 = nn.Linear(dim_head, dim_hid, bias=False)
        
        # MLP Layer 2
        a = math.sqrt(6/(dim_ker + dim_hid))
        self.kernel_q_mat2 = torch.nn.init.uniform_(torch.empty(num_heads, dim_hid, dim_ker), a=-a, b=a)
        self.kernel_q_mat2 = nn.Parameter(self.kernel_q_mat2, requires_grad=True)
        if not multi_query:
            self.kernel_k_mat2 = torch.nn.init.uniform_(torch.empty(num_heads, dim_hid, dim_ker), a=-a, b=a)
            self.kernel_k_mat2 = nn.Parameter(self.kernel_k_mat2, requires_grad=True)
        else:
            self.kernel_k_mat2 = nn.Linear(dim_hid, dim_ker, bias=False)

        

        # MLP Layer 3 (keys only)
        a = math.sqrt(6/(2 * dim_ker))
        if not multi_query:
            self.interaction_k = torch.nn.init.uniform_(torch.empty(num_heads, dim_ker, dim_ker), a=-a, b=a)
            self.interaction_k = nn.Parameter(self.interaction_k, requires_grad=True)
        else:
            self.interaction_k = nn.Linear(dim_ker, dim_ker, bias=False)
        
        
        if multi_query:
            num_heads = 1
        self.scalingD = nn.Parameter(torch.ones(1, num_heads, 1, dim_ker) * 1e-4, requires_grad=True)
        self.scalingD2 = nn.Parameter(torch.ones(1, num_heads, 1, dim_ker) * 1e-4, requires_grad=True)

        # initialize at 5.0 for all learned lambdas that are still time-independent
        # initialization at 5.0 results in decay post-sigmoid of around 0.99 without being too deep
        # into extremely low gradient territory, seems neutral enough
        if self.lambda_gating == "learned-constant":
            self.W_lambda = nn.Parameter(5.0 * torch.ones(1), requires_grad=True)
        elif self.lambda_gating == "learned-constant-head":
            assert not multi_query, "Multi-query attention (MQA) not supported by per-head lambdas, defeats the purpose of MQA."
            self.W_lambda = nn.Parameter(5.0 * torch.ones(num_heads), requires_grad=True)
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
            self.W_lambda = nn.Sequential(
                nn.Linear(dim_hid, 16, bias=False),
                nn.Linear(16, dim_ker, bias=False)
            )

        self.multi_query = multi_query
        self.act = F.gelu
        self.kernel_f = F.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t, q, k, v, lr_attn_mask, sparse_norms_lse, sparse_attn_weights, lambda_constant):
        B, S, D = q.shape
            
        # addition of lambda gating-based decay for sparse mask
        decay_mask = None
        if self.lambda_gating == "constant":
            decay_mask = get_lambda_mask_sparse(lr_attn_mask, lambda_constant)
        elif self.lambda_gating == "learned-constant" or lambda_gating == "learned-constant-head":
            decay_mask = get_lambda_mask_sparse(lr_attn_mask, F.sigmoid(self.W_lambda))
        elif self.lambda_gating == "time-dependent":
            # here, self.W_lambda is a nn.Linear instead of being parameters we apply sigmoid to
            lambda_t = torch.exp(-1.0 * F.softplus(self.W_lambda(x_t)) * torch.exp(self.alpha))
            decay_mask = get_lambda_mask_sparse(lr_attn_mask, lambda_t)
        elif self.lambda_gating == "time-data-dependent":
            raise NotImplementedError("Time-data-dependent lambda gating not yet implemented.")

        q = q.reshape(B, S, self.num_heads, D // self.num_heads).transpose(1, 2)
        if not multi_query:
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