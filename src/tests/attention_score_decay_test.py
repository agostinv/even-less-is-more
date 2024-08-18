import unittest
import torch
import math
from masks import local_heavy_hitter_mask_nonoverlap
import torch.nn as nn

def reference_local_heavy_hitter_mask_nonoverlap(attn_weights, heavy_budget, recent_budget, no_padding_seq_length=None, multi_query=False):

    # attn_weights (BS, head, query, keys)
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]
    if no_padding_seq_length is None:
        padding_length = 0
    else:
        raise NotImplementedError("Padding not yet implemented")
        padding_length = seq_length - no_padding_seq_length

    offset = torch.finfo(attn_weights.dtype).min
    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

    accumulated_attention_score = torch.sum(tmp_attn[:,:,padding_length:heavy_budget+recent_budget+padding_length,:], dim=-2) #(head, keys)
    accumulated_attention_score[:,:,heavy_budget+recent_budget+padding_length:] = 0
    accumulated_attention_score[:,:,:padding_length] = 0

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    if multi_query:
        mask_bottom = mask_bottom[:,0].unsqueeze(1) #B1SS
        accumulated_attention_score = accumulated_attention_score.sum(dim=1, keepdim=True) #B1S
    mask_bottom[:,:, padding_length:heavy_budget+recent_budget+padding_length, padding_length:heavy_budget+recent_budget+padding_length] = True

    for token_index in range(heavy_budget+recent_budget+padding_length, seq_length):
        
        tmp_attn_index = nn.functional.softmax(attn_weights[:,:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)

        if multi_query:
            tmp_attn_index = tmp_attn_index.sum(dim=1, keepdim=True) #B1S
        _, tmp_topk_index = accumulated_attention_score[..., :token_index-recent_budget].topk(k=heavy_budget, dim=-1)
        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)
        
        mask_bottom_index[:, : , token_index-recent_budget:token_index+1] = True

        mask_bottom[:,:,token_index,:] = mask_bottom_index
        accumulated_attention_score += tmp_attn_index
        accumulated_attention_score = accumulated_attention_score * mask_bottom_index
    
    return mask_bottom


class TestLocalHeavyHitterMaskNonoverlap(unittest.TestCase):
    def setUp(self):
        # Sample input, decided arbitrarily, needs to be of shape B x H x N x N,
        # here as 1 x 1 x 4 x 4
        self.attn_weights = torch.tensor([[[
            [3.0,    -999.0,    -999.0,     -999.0,     -999.0,     -999.0],
            [10.0,   10.0,      -999.0,     -999.0,     -999.0,     -999.0],
            [10.0,   10.0,      12.0,       -999.0,     -999.0,     -999.0],
            [10.0,   10.0,      13.0,       16.0,       -999.0,     -999.0],
            [10.0,   10.0,      14.0,       17.0,       19.0,       -999.0],
            [25.0,   11.0,      15.0,       18.0,       20.0,       21.0],
        ]]], dtype=torch.float32)
        self.heavy_budget = 2
        self.recent_budget = 1

    def test_attention_decay_mask(self):
        # grabbing end h2o matrix with specified decays, first is a control call with a reference function
        attention_h2o_mask_no_decay = local_heavy_hitter_mask_nonoverlap(self.attn_weights, self.heavy_budget, self.recent_budget, attention_score_decay=1.0)
        control_mask = reference_local_heavy_hitter_mask_nonoverlap(self.attn_weights, self.heavy_budget, self.recent_budget)
        self.assertTrue(torch.equal(attention_h2o_mask_no_decay, control_mask),
                        f"Control masks don't match: \n{attention_h2o_mask_no_decay}\n\n{control_mask}\n\n")

        # mild amounts of decay
        attention_h2o_mask_mild_decayed = local_heavy_hitter_mask_nonoverlap(self.attn_weights, self.heavy_budget, self.recent_budget, attention_score_decay=0.8)
        mild_decay_mask = torch.tensor([[[
            [ True,  True,  True, False, False, False],
            [ True,  True,  True, False, False, False],
            [ True,  True,  True, False, False, False],
            [ True,  True,  True,  True, False, False],
            [ True, False,  True,  True,  True, False],
            [ True, False, False,  True,  True,  True]
        ]]])
        self.assertTrue(torch.equal(attention_h2o_mask_mild_decayed, mild_decay_mask),
                        f"Mild decay masks don't match: \n{attention_h2o_mask_mild_decayed}\n\n{mild_decay_mask}\n\n")

        # extreme decay, expect extreme diagonal behavior
        attention_h2o_mask_extr_decayed = local_heavy_hitter_mask_nonoverlap(self.attn_weights, self.heavy_budget, self.recent_budget, attention_score_decay=0.1)
        extr_decay_mask = torch.tensor([[[
            [ True,  True,  True, False, False, False],
            [ True,  True,  True, False, False, False],
            [ True,  True,  True, False, False, False],
            [ True,  True,  True,  True, False, False],
            [False,  True,  True,  True,  True, False],
            [False, False,  True,  True,  True,  True]
        ]]])
        self.assertTrue(torch.equal(attention_h2o_mask_extr_decayed, extr_decay_mask),
                        f"Extreme decay masks don't match: \n{attention_h2o_mask_extr_decayed}\n\n{extr_decay_mask}\n\n")

if __name__ == '__main__':
    unittest.main()