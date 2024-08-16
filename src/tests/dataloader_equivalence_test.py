import torch
import unittest
from utils.train_utils import mem_eff_get_activations_layer, get_activations_layer

class TestActivationsLayer(unittest.TestCase):
    def setUp(self):
        # Set up a simple model and dataloader for testing, values chosen arbitrarily 
        self.batches = 2
        self.bsz = 4
        self.num_heads = 2
        self.seq_len = 5
        self.head_dim = 3

        self.model = DUT(self.num_heads, self.head_dim)
        
        # Create dummy data
        self.dataloader = []
        for i in range(self.batches):
            inputs = torch.randn(self.bsz, self.num_heads, self.seq_len, self.head_dim)
            labels = None
            self.dataloader.append((inputs, labels))

    def test_logical_equivalency(self):
        # Get activations from both functions
        qs_all_mem_eff, ks_all_mem_eff, vs_all_mem_eff, os_all_mem_eff = mem_eff_get_activations_layer(
            model=self.model,
            layer=0,
            dataloader=self.dataloader,
            batches=self.batches,
            bsz=self.bsz,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            permute=False,
	    half_precision=False,
	    frag_factor=2,
        )
        
        qs_all, ks_all, vs_all, os_all = get_activations_layer(
            model=self.model,
            layer=0,
            dataloader=self.dataloader,
            batches=self.batches,
            permute=False,
        )

        # Check for logical equivalency
        self.assertTrue(torch.allclose(qs_all_mem_eff, qs_all))
        self.assertTrue(torch.allclose(ks_all_mem_eff, ks_all))
        self.assertTrue(torch.allclose(vs_all_mem_eff, vs_all))
        self.assertTrue(torch.allclose(os_all_mem_eff, os_all))

# extremely simple DUT for low-resource device testing
class DUT(torch.nn.Module):
    def __init__(self, num_heads, head_dim):
        super(DUT, self).__init__()
        self.model = torch.nn.Linear(head_dim, head_dim)
        self.model.device = 'cpu'
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x):
        # Simulate the output structure
        output = self.model(x)
        batch_int_values = {
            0: {
                'Q': self.model(x),
                'K': self.model(x),
                'V': self.model(x),
                'O': self.model(x).view(output.size(0), 1, output.size(2), self.num_heads * self.head_dim).squeeze(1)
            }
        }
        return output, batch_int_values

if __name__ == '__main__':
    unittest.main()
