import unittest
import torch
from custom_attention import KernelizedHeadAttention

import triton
import triton.language as tl

import torch.nn.functional as F

'''
    Logic in KernelizedHeadAttention for time-data-dependent lambda gating is
    non-trivial and basically requires a test to try and test for weird behavior.
    At a minimum, this test should ensure FLA is available to be imported and that
    the results of the FLA forward pass are close to a reference pass implemented
    inefficiently, but with expected logic.

    Unlike some other tests, this test should be run on a GPU compatible with CUDA.
'''

@triton.jit
def precision_error(lambda_in, lambda_out, stride):
    m = tl.program_id(0)

    lambda_in = lambda_in + m * stride
    lambda_out = lambda_out + m * stride

    l_tl = tl.load(lambda_in)
    l_tl_exp = tl.exp(l_tl)
    tl.store(lambda_out, l_tl_exp)


class TestKHAForTDDwFLA(unittest.TestCase):
    def setUp(self):
        assert torch.cuda.is_available(), "This test expects a CUDA-compatible GPU when run."
        
        # toggle to check between implementations
        self.multi_query = True

        # minimum size is expected of 16 of self.S and self.D
        self.B = 2
        self.S = 2048
        self.H = 32
        self.D = 16

        self.H_k = 1 if self.multi_query else H
        self.D_v = 128

        self.q = torch.randn(self.B, self.H, self.S, self.D, device="cuda")
        self.k = torch.randn(self.B, self.H_k, self.S, self.D, device="cuda")
        self.v = torch.randn(self.B, self.H_k, self.S, self.D_v, device="cuda")
        self.lambda_t_data = F.logsigmoid(torch.randn(self.B, self.S, self.H_k * self.D, device="cuda")) / 16

        self.eviction_kv_indices = torch.arange(0, self.S, device="cuda")[torch.randperm(self.S)].expand(self.B, -1).repeat(1, self.H_k) # B x (H x S)

        self.q = self.q.abs()
        self.k = self.k.abs()

        self.kha = KernelizedHeadAttention(
            dim_head=self.D,
            dim_hid=self.H * self.D,
            dim_ker=self.D,
            num_heads=self.H,
            dropout=0.0,
            multi_query=self.multi_query,
            lambda_gating="time-data-dependent",
        )


    def equivalent_fwd_pass_logic(self, q, k, v, lambda_t_data, eviction_kv_indices, multi_query=False):
        B, H, S, D = q.size()
        _, H_k, _, _ = k.size()
        _, _, _, D_v = v.size()

        # GLA kernels execute on this no matter what we do, we would have to multiply
        # the query by this factor to remove the scaling
        q = q * q.size(-1) ** -0.5


        assert eviction_kv_indices.size(-1) % H_k == 0, \
            "Even division of triggered token length by attn. heads required. "        + \
            "Eviction decisions are not allowed to be dependent on attn. heads, "      + \
            "i.e. the same number of tokens must be evicted per-head. Non-factor for " + \
            "multi-query attention."

        # B x (S_reduct. * H) => B x S_reduct. x H, where S_reduct. also accounts for offset for query
        eviction_kv_indices = eviction_kv_indices.unsqueeze(1).view(B, H_k, eviction_kv_indices.size(-1) // H_k)
        
        
        # reorder keys and values according to gathered indices ordered
        # by eviction events, check related test to verify correctness of this 
        # behavior if in doubt
        eviction_k_indices = eviction_kv_indices.unsqueeze(-1).expand(-1, -1, -1, D) # B x S_reduct. x H => B x S_reduct. x H x D_k
        eviction_v_indices = eviction_kv_indices.unsqueeze(-1).expand(-1, -1, -1, D_v) # B x S_reduct. x H => B x S_reduct. x H x D_v
        k = k.gather(-2, eviction_k_indices)
        v = v.gather(-2, eviction_v_indices)

        # not testing multi_query here, so simple view
        lambda_t_data = lambda_t_data.view(B, H_k, S, D)
        
        if self.multi_query:
            k = k.expand(-1, H, -1, -1)
            v = v.expand(-1, H, -1, -1)
            lambda_t_data = lambda_t_data.expand(-1, H, -1, -1)

        # initialize and begin inefficient linear attention
        recurrent_state = torch.zeros(B, H, D, D_v, device=q.device)
        out = torch.zeros(B, H, S, D_v, device=q.device)
        for i in range(S):
            q_temp = q[:, :, i, :].unsqueeze(-2)
            k_temp = k[:, :, i, :].unsqueeze(-2)
            v_temp = v[:, :, i, :].unsqueeze(-2)

            # account for summation tensor in case of norm
            if v_temp.dim() < 4:
                v_vemp = v_temp.unsqueeze(-1)

            lambda_i = lambda_t_data[:, :, i, :].unsqueeze(-1).exp() # B x H x D_k x 1

            recurrent_state = (lambda_i * recurrent_state) + torch.matmul(k_temp.transpose(2, 3), v_temp)
            out[:, :, i, :] = torch.matmul(q_temp, recurrent_state).squeeze(2)

        return out


    # tests exp precision
    # then checks output equality
    # then checks norm equality
    def test_KHA_for_TDD_w_FLA(self):
        
        # double check for no weird tl precision issues
        lambda_tl = torch.zeros_like(self.lambda_t_data)
        precision_error[(self.B * self.S * self.H_k * self.D,)](self.lambda_t_data, lambda_tl, 1)
        lambda_base = torch.exp(self.lambda_t_data)

        self.assertTrue(torch.allclose(lambda_tl, lambda_base, atol=1e-6),
                        msg=f'\nExpected \n{lambda_base}, \n\ngot \n{lambda_tl}')
      
        # generate output and norm
        output, norm, _ = self.kha.time_data_dep_forward(
            self.q, self.k, self.v, self.lambda_t_data, self.eviction_kv_indices, fused_recurrent_override=True
        )

        summation_value = torch.ones(self.B, self.H_k, self.S, 1, device=self.q.device)
        output_ref = self.equivalent_fwd_pass_logic(
            self.q, self.k, self.v, self.lambda_t_data, self.eviction_kv_indices, multi_query=self.multi_query
        )
        norm_ref = self.equivalent_fwd_pass_logic(
            self.q, self.k, summation_value, self.lambda_t_data, self.eviction_kv_indices, multi_query=self.multi_query
        )

        # Ensure that the output and normalization tensors are close
        self.assertTrue(torch.allclose(output, output_ref, atol=1e-4),
                        msg=f'\nExpected \n{output_ref}, \n\ngot \n{output}')
        self.assertTrue(torch.allclose(norm, norm_ref, atol=1e-4),
                        msg=f'\nExpected \n{norm_ref}, \n\ngot \n{norm}')


if __name__ == '__main__':
    unittest.main()
