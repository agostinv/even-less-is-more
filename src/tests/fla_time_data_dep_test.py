import unittest
import torch
from custom_attention import KernelizedHeadAttention

'''
    Logic in KernelizedHeadAttention for time-data-dependent lambda gating is
    non-trivial and basically requires a test to try and test for weird behavior.
    At a minimum, this test should ensure FLA is available to be imported and that
    the results of the FLA forward pass are close to a reference pass implemented
    inefficiently, but with expected logic.

    Unlike some other tests, this test should be run on a GPU compatible with CUDA.
'''
class TestKHAForTDDwFLA(unittest.TestCase):
    def setUp(self):
        self.B = 2  # Batch size
        self.S = 3  # Sequence length
        self.H = 2  # Number of heads
        self.D = 4  # Dimension per head

        self.q = torch.randn(self.B, self.S, self.H, self.D)
        self.k = torch.randn(self.B, self.S, self.H, self.D)
        self.v = torch.randn(self.B, self.S, self.H, self.D)
        self.lambda_t_data = torch.randn(self.B, self.S, self.H * self.D)
        self.eviction_kv_indices = torch.arange(0, self.S, 1)[torch.randperm(self.S)]

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
            lambda_i = lambda_t_data[:, :, i, :].repeat(1, 1, D).view(B, H, D, D)
            recurrent_state = lambda_i * recurrent_state + torch.matmul(k[:, :, i, :].transpose(2, 3), v[:, :, i, :])
            out[:, :, i, :] = torch.matmul(q[:, :, i, :], recurrent_state)

        # GLA kernels don't return normalization tensors, as they use LayerNorm instead,
        # so we need to construct them ourselves
        k_cum_T = torch.cumsum(k, dim=-2).transpose(2, 3)
        norm = torch.matmul(q, k_cum_T)
        norm = torch.diagonal(norm, dim1=-2, dim2=-1).unsqueeze(-1)

        return out, norm


    def test_KHA_for_TDD_w_FLA(self):
        # Call the method
        output, norm = self.kha.time_data_dep_forward(
            self.q, self.k, self.v, self.lambda_t_data, self.eviction_kv_indices
        )

        output_ref, norm_ref = self.equivalent_fwd_pass_logic(
            self.q, self.k, self.v, self.lambda_t_data, self.eviction_kv_indices
        )

        # Ensure that the output and normalization tensors are close
        self.assertTrue(torch.allclose(output, output_ref, atol=1e-6),
                        msg=f'\nExpected \n{output_ref}, \n\ngot \n{output}')
        self.assertTrue(torch.allclose(norm, norm_ref, atol=1e-6),
                        msg=f'\nExpected \n{norm_ref}, \n\ngot \n{norm}')


if __name__ == '__main__':
    unittest.main()