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
        ]], 
        [[    [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 1, 1, 0, 0, 0],
              [0, 1, 1, 1, 0, 0],
              [1, 1, 1, 1, 0, 0]
        ]]], dtype=torch.float32)

        self.k = torch.randn(2, 6, 4)

    def test_kv_eviction_indices(self):
        # Call the function
        indices = get_eviction_kv_indices(self.sparse_mask)

        # ensure that the result is effectively just a 1D tensor of indices
        # for KV reordering
        expected_indices = torch.tensor([[0, 2, 1, 4], [1, 2, 3, 0]])

        self.assertTrue(torch.allclose(expected_indices, indices, atol=1e-6), 
                        msg=f'\nExpected \n{expected_indices}, \n\ngot \n{indices}')
        
        
        # check for appropriate behavior with KV reordering, ideally
        # automatically deals with size difference for indexing
        k_expected = torch.zeros(2, 4, 4)
        temp_evict = expected_indices
        for i in range(temp_evict.size(0)):
            for j in range(temp_evict.size(1)):
                k_expected[i, j, :] = self.k[i, expected_indices[i, j], :]
        
        indices = indices.unsqueeze(-1).expand(-1, -1, self.k.size(2))
        self.k = self.k.gather(1, indices)
        
        self.assertTrue(torch.allclose(k_expected, self.k, atol=1e-6), 
                        msg=f'\nExpected \n{k_expected}, \n\ngot \n{self.k}')

if __name__ == '__main__':
    unittest.main()
