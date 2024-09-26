import unittest
import torch
from masks import get_eviction_kv_indices

class TestKVEvictionIndices(unittest.TestCase):
    def setUp(self):
        # Set up some sample inputs, masks should be of dimensions B x H x N x N
        # these samples are 2 x 2 x 6 x 6
        self.sparse_mask = torch.tensor([[[
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 0],
              [1, 1, 1, 0, 0, 0],
              [1, 1, 1, 0, 1, 0]
        ],
        [     [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 1, 1, 0, 0, 0],
              [0, 1, 1, 1, 0, 0],
              [1, 1, 1, 1, 0, 0]
        ]], 
        [[    [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 1, 1, 0, 0, 0],
              [0, 1, 1, 1, 0, 0],
              [1, 1, 1, 1, 0, 0]
        ],
        [     [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 0],
              [1, 1, 1, 0, 0, 0],
              [1, 1, 1, 0, 1, 0]
        ]]], dtype=torch.float32)

        self.k = torch.randn(2, 2, 6, 4) # B x H x N x D_k

    def test_kv_eviction_indices(self):
        # Call the function
        indices = get_eviction_kv_indices(self.sparse_mask)
        
        B, H, _, _ = self.k.shape
        indices = indices.unsqueeze(1).view(B, H, indices.size(-1) // H)

        # ensure that the result is effectively just a 1D tensor of indices
        # for KV reordering
        expected_indices = torch.tensor([[   [0, 2, 1, 4],
                                             [1, 2, 3, 0]],
                                            [[1, 2, 3, 0],
                                             [0, 2, 1, 4]
                                        ]])
        self.assertTrue(torch.allclose(expected_indices, indices, atol=1e-6), 
                        msg=f'\nExpected \n{expected_indices}, \n\ngot \n{indices}')

        
        # check for appropriate behavior with KV reordering, ideally
        # automatically deals with size difference for indexing
        k_expected = torch.zeros(2, 2, 4, 4)
        temp_evict = expected_indices
        for i in range(temp_evict.size(0)):
            for j in range(temp_evict.size(1)):
                for k in range(temp_evict.size(2)):
                    k_expected[i, j, k, :] = self.k[i, j, expected_indices[i, j, k], :]
        
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, self.k.size(3))
        self.k = self.k.gather(2, indices)
        
        self.assertTrue(torch.allclose(k_expected, self.k, atol=1e-6), 
                        msg=f'\nExpected \n{k_expected}, \n\ngot \n{self.k}')

if __name__ == '__main__':
    unittest.main()
