import unittest
import torch
from masks import get_eviction_kv_indices

class TestKVEvictionIndices(unittest.TestCase):
    def setUp(self):
        # Set up some sample inputs, masks should be of dimensions B x H x N x N
        # these samples are 1 x 1 x 6 x 6
        self.sparse_mask = torch.tensor([[[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 1, 0]
        ]]], dtype=torch.float32)

    def test_kv_eviction_indices(self):
        # Call the function
        indices = get_eviction_kv_indices(self.sparse_mask)

        # ensure that the result is effectively just a 1D tensor of indices
        # for KV reordering
        expected_indices = torch.tensor([0, 2, 1, 4], dtype=torch.long)

        self.assertTrue(torch.allclose(expected_indices, indices, atol=1e-6), 
                        msg=f'\nExpected \n{expected_indices}, \n\ngot \n{indices}')

if __name__ == '__main__':
    unittest.main()