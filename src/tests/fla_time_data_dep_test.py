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

        # minimum size is expected of 16 of self.S and self.D
        self.B = 2
        self.S = 16
        self.H = 2
        self.D = 16

        self.q = torch.randn(self.B, self.S, self.H, self.D, device="cuda")
        self.k = torch.randn(self.B, self.S, self.H, self.D, device="cuda")
        self.v = torch.randn(self.B, self.S, self.H, self.D, device="cuda")
        self.lambda_t_data = F.logsigmoid(torch.randn(self.B, self.S, self.H * self.D, device="cuda")) / 16

        self.eviction_kv_indices = torch.arange(0, self.S, device="cuda")[torch.randperm(self.S)]

        self.q = self.q.abs()
        self.k = self.k.abs()

        self.kha = KernelizedHeadAttention(
            dim_head=self.D,
            dim_hid=self.H * self.D,
            dim_ker=self.D,
            num_heads=self.H,
            dropout=0.0,
            multi_query=False,
            lambda_gating="time-data-dependent",
        )


    def equivalent_fwd_pass_logic(self, q, k, v, lambda_t_data, eviction_kv_indices, multi_query=False):
        B, S, H, D = q.size()
      
        # GLA kernels execute on this no matter what we do, we would have to multiply
        # the query by this factor to remove the scaling
        q = q * q.size(-1) ** -0.5

        # transpose for consistency with FLA expectations, even though
        # we don't use it here
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)        

        # reorder keys and values according to gathered indices ordered
        # by eviction events
        k = k[:, :, self.eviction_kv_indices, :]
        v = v[:, :, self.eviction_kv_indices, :]

        # not testing multi_query here, so simple view
        lambda_t_data = lambda_t_data.view(B, H, S, D)

        # initialize and begin inefficient linear attention
        recurrent_state = torch.zeros(B, H, D, D, device=q.device)
        out = torch.zeros_like(q)
        for i in range(S):
            q_temp = q[:, :, i, :].unsqueeze(2)
            k_temp = k[:, :, i, :].unsqueeze(2)
            v_temp = v[:, :, i, :].unsqueeze(2)
            lambda_i = lambda_t_data[:, :, i, :].unsqueeze(2).exp().transpose(2, 3) # B x H x D_k x 1

            recurrent_state = (lambda_i * recurrent_state) + torch.matmul(k_temp.transpose(2, 3), v_temp)
            out[:, :, i, :] = torch.matmul(q_temp, recurrent_state).squeeze(2)

        # GLA kernels don't return normalization tensors, as LA usually uses LayerNorm instead,
        # so we need to construct them ourselves
        lambda_t_shifted = torch.ones_like(lambda_t_data)
        lambda_t_shifted[:, :, 1:, :] = torch.exp(lambda_t_data[:, :, :-1, :])
        k_for_norm = lambda_t_shifted * k

        k_cum_T = torch.cumsum(k_for_norm, dim=-2).transpose(2, 3)
        norm = torch.matmul(q, k_cum_T)
        norm = torch.diagonal(norm, dim1=-2, dim2=-1).unsqueeze(-1)

        return out, norm


    # tests exp precision
    # then checks output equality
    # then checks norm equality
    def test_KHA_for_TDD_w_FLA(self):
        
        # double check for no weird tl precision issues
        lambda_tl = torch.zeros_like(self.lambda_t_data)
        precision_error[(self.B * self.S * self.H * self.D,)](self.lambda_t_data, lambda_tl, 1)
        lambda_base = torch.exp(self.lambda_t_data)

        self.assertTrue(torch.allclose(lambda_tl, lambda_base, atol=1e-6),
                        msg=f'\nExpected \n{lambda_base}, \n\ngot \n{lambda_tl}')
      
        # generate output and norm
        output, norm = self.kha.time_data_dep_forward(
            self.q, self.k, self.v, self.lambda_t_data, self.eviction_kv_indices, recurrent_override=True
        )

        output_ref, norm_ref = self.equivalent_fwd_pass_logic(
            self.q, self.k, self.v, self.lambda_t_data, self.eviction_kv_indices
        )

        # Ensure that the output and normalization tensors are close
        self.assertTrue(torch.allclose(output, output_ref, atol=1e-4),
                        msg=f'\nExpected \n{output_ref}, \n\ngot \n{output}')
        self.assertTrue(torch.allclose(norm, norm_ref, atol=1e-4),
                        msg=f'\nExpected \n{norm_ref}, \n\ngot \n{norm}')


if __name__ == '__main__':
    unittest.main()
