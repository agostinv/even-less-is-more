import unittest
import torch
from masks import get_lambda_mask_sparse

class TestGetLambdaMaskSparse(unittest.TestCase):
    def setUp(self):
        # Set up some sample inputs, masks should be of dimensions B x H x N x N
        # these samples are 1 x 1 x 6 x 6
        self.attn_mask = torch.tensor([[[
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1]
        ]]], dtype=torch.float32)

        self.sparse_mask = torch.tensor([[[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 1, 0]
        ]]], dtype=torch.float32)

        self.lambda_val = 0.5
        self.decay = list(map(lambda x: self.lambda_val**x, range(self.sparse_mask.size(-1))))

    def test_lambda_mask_sparse(self):
        # Call the function
        lambda_mask = get_lambda_mask_sparse(self.sparse_mask, self.lambda_val)
        lambda_mask = torch.logical_xor(self.attn_mask, self.sparse_mask).to(torch.float32) + lambda_mask

        # Check that the mask decays on triggering events (i.e. cache eviction)
        # The exact values will depend on the implementation, but we expect a decaying pattern
        # appearing dynamically, self.decay[0] is "no decay" in this instance but effectively begins
        # a timer across the rows for decay to appear in this test
        expected_mask = torch.tensor([[[
            [1.000,         0.000,         0.000,         0.000, 0.000,         0.000],
            [1.000,         1.000,         0.000,         0.000, 0.000,         0.000],
            [self.decay[0], 1.000,         1.000,         0.000, 0.000,         0.000],
            [self.decay[1], 1.000,         self.decay[0], 1.000, 0.000,         0.000],
            [self.decay[2], self.decay[0], self.decay[1], 1.000, 1.000,         0.000],
            [self.decay[3], self.decay[1], self.decay[2], 1.000, self.decay[0], 1.000]
        ]]], dtype=torch.float32)

        self.assertTrue(torch.allclose(lambda_mask, expected_mask, atol=1e-6), 
                        msg=f'\nExpected \n{expected_mask}, \n\ngot \n{lambda_mask}')

if __name__ == '__main__':
    unittest.main()