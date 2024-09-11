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

        self.k = torch.randn(6, 4)

    def test_kv_eviction_indices(self):
        # Call the function
        indices = get_eviction_kv_indices(self.sparse_mask)

        # ensure that the result is effectively just a 1D tensor of indices
        # for KV reordering
        expected_indices = torch.tensor([0, 2, 1, 4], dtype=torch.long)

        # check for appropriate behavior with KV reordering, ideally
        # automatically deals with size difference for indexing
        k_expected = torch.zeros(4, 4)
        for i in range(expected_indices.size(0)):
            k_expected[i, :] = self.k[expected_indices[i], :]

        self.assertTrue(torch.allclose(expected_indices, indices, atol=1e-6), 
                        msg=f'\nExpected \n{expected_indices}, \n\ngot \n{indices}')
        
        self.assertTrue(torch.allclose(k_expected, self.k[indices], atol=1e-6), 
                        msg=f'\nExpected \n{expected_indices}, \n\ngot \n{indices}')

if __name__ == '__main__':
    unittest.main()
