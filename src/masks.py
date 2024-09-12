# Adapted from https://github.com/FMInference/H2O/blob/main/h2o_hf/utils_lm_eval/modify_opt.py

import torch
import torch.nn as nn

def local_heavy_hitter_mask_nonoverlap(attn_weights, heavy_budget, recent_budget, no_padding_seq_length=None, multi_query=False, attention_score_decay=1.0):

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

    # does nothing while attention_score_decay isn't set to a custom value, otherwise applies decay along the query axis to 
    # try and mitigate the power law distribution native to H2O
    # needs to be added later as well as the lower mask is removed before the below `for loop`
    # 1e-6 mask is needed to prevent boolean switch from flipping valid values during control flows (i.e. decay = 1.0)
    # implementation is a little clunky, could be revisited but should work
    attention_decay_mask = torch.tril(attention_score_decay * torch.ones_like(attn_weights, dtype=dtype_attn_weights))
    attention_decay_mask[attention_decay_mask == 0] = 1
    attention_decay_mask = attention_decay_mask.cumprod(dim=-2) + torch.tril(1e-6 * torch.ones_like(attn_weights, dtype=dtype_attn_weights))
    attention_decay_mask[attention_decay_mask == 1] = 0
    attention_decay_mask = attention_decay_mask / attention_score_decay

    accumulated_attention_score = torch.cumsum(tmp_attn, dim=-2) #(head, keys)
    accumulated_attention_score[:,:,:,heavy_budget+recent_budget+padding_length:] = 0
    accumulated_attention_score[:,:,:,:padding_length] = 0

    # decay accumulated scores so far
    accumulated_attention_score = accumulated_attention_score * attention_decay_mask

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    if multi_query:
        mask_bottom = mask_bottom[:,0].unsqueeze(1) #B1SS
        accumulated_attention_score = accumulated_attention_score.sum(dim=1, keepdim=True) #B1S
    mask_bottom[:,:, padding_length:heavy_budget+recent_budget+padding_length, padding_length:heavy_budget+recent_budget+padding_length] = True

    for token_index in range(heavy_budget+recent_budget+padding_length, seq_length):
        
        tmp_attn_index = nn.functional.softmax(attn_weights[:,:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)

        if multi_query:
            tmp_attn_index = tmp_attn_index.sum(dim=1, keepdim=True) #B1S
        _, tmp_topk_index = accumulated_attention_score[..., token_index - 1, :token_index-recent_budget].topk(k=heavy_budget, dim=-1)
        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)
        
        mask_bottom_index[:, :, token_index-recent_budget:token_index+1] = True

        mask_bottom[:,:,token_index,:] = mask_bottom_index
        accumulated_attention_score[..., token_index, :] = tmp_attn_index + attention_score_decay * accumulated_attention_score[..., token_index - 1, :]
        accumulated_attention_score[..., token_index, :] = accumulated_attention_score[..., token_index, :] * mask_bottom_index
    
    return mask_bottom


def get_h2o_mask(attn_weights, heavy_budget, recent_budget, multi_query, attention_score_decay):
    if heavy_budget > 0:
        mask_bottom = local_heavy_hitter_mask_nonoverlap(attn_weights, heavy_budget, recent_budget, multi_query=multi_query, attention_score_decay=attention_score_decay) # Default: No padding applied to input
    else:
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    if multi_query:
        ones = torch.ones_like(mask_bottom, dtype=torch.bool)
    else:
        ones = torch.ones_like(attn_weights, dtype=torch.bool)
    ones = torch.triu(ones, diagonal=-recent_budget)
    mask_bottom = torch.logical_or(mask_bottom, ones)

    mask_bottom = torch.tril(mask_bottom, diagonal=0)

    return mask_bottom

def get_A_mask(attn_weights, heavy_budget, recent_budget):
    A_mask = torch.ones_like(attn_weights, dtype=torch.bool)
    A_mask = torch.triu(A_mask, diagonal=-recent_budget)
    A_mask[..., :heavy_budget] = 1
    A_mask = torch.tril(A_mask, diagonal=0)
    return A_mask

'''
    Series of simple mask construction for global cache decay based on
    constant or time-dependent scalar decay values. Intended to be applied
    during training when an N x N matrix is being constructed anyways for the
    attention portion modeling inference sparsity.

    These only work for actual, consistent progression across time. Not 
    functional for dynamic sparsity, but present as a reference.
'''

# assume purely causal architecture, of course
# assumes that lambda_val ranges from [0, 1]
def get_lambda_mask_nosparse(attn_weights, lambda_val: float):
    len = attn_weights.size(0)
    lambda_mask = torch.arange(len).unsqueeze(0) - torch.arange(len).unsqueeze(1)
    lambda_mask = torch.tril(lambda_mask, diagonal=0)
    lambda_mask = lambda_val ** lambda_mask
    return lambda_mask

# lambda_t_val MUST be broadcastable, i.e. it must be N x 1
def get_lambda_t_mask_nosparse(attn_weights, lambda_t_val: torch.Tensor):
    len = attn_weights.size(0)
    lambda_mask = torch.tril(torch.ones(len, len), diagonal=0)
    lambda_t_val = lambda_t_val.unsqueeze(0).expand(len, len)
    lambda_mask = torch.cumprod(lambda_mask * lambda_t_val, dim=1)
    return lambda_mask


'''
    What is actually required is a parallelizable method of modeling WHEN states
    are added to the local cache, at which point we want to apply decay to ALL previous
    states. One thought is shifting!

    Example below provides basic idea with an inverted, dynamic map (~sparse_map). When a
    bit flips it represents ejection. Note that end product is lower_tril * (~sparse_map)
    so no need to worry about upper triangular portion. Afterward, we XOR to produce...

    e.g.  #############     #############       ############# 
          # 0 - - - - #     # 0 - - - - #  XOR  # 0 - - - - #
          # 1 0 - - - #  => # 0 0 - - - #   =>  # 1 0 - - - #
          # 1 0 0 - - #     # 1 0 0 - - #       # 0 0 0 - - #
          # 1 0 1 0 - #     # 1 0 0 0 - #       # 0 0 1 0 - #
          # 1 0 1 0 0 #     # 1 0 1 0 0 #       # 0 0 0 0 0 #
          #############     #############       #############

    From here, we can sum and multiply the output by some constant lambda and elementwise 
    multiply with the first mask. This produces the following with `g` representing our 
    constant lambda.

    e.g. #############     #####     #####         #############
         # 0 - - - - #     # 0 #     # 0 #  mult   # 0 - - - - #
         # 1 0 - - - #  => # 1 #  => # g #  =>     # g 0 - - - #
         # 0 0 0 - - #     # 0 #     # 0 #         # 1 0 0 - - #
         # 0 0 1 0 - #     # 1 #     # g #         # g 0 g 0 - #
         # 0 0 0 0 0 #     # 0 #     # 0 #         # 1 0 1 0 0 #
         #############     #####     #####         #############

    This produces a matrix that we can then simply cumprod across the key dimension to get 
    something close to what we want. Dividing it by `g` results in exact inference behavior!
    `h` represents g^2 to maintain "prettiness" here.

    e.g. #############           #############            #############
         # 0 - - - - #           # 0 - - - - #            # 0 - - - - #
         # g 0 - - - #  cumprod  # g 0 - - - #   divide   # 1 0 - - - #
         # 1 0 0 - - #    =>     # g 0 0 - - #      =>    # 1 0 0 - - #
         # g 0 g 0 - #           # h 0 g 0 - #            # g 0 1 0 - #
         # 1 0 1 0 0 #           # h 0 g 0 0 #            # g 0 1 0 0 #
         #############           #############            #############

         
    This also makes a time-dependent version somewhat easier, but still not
    trivial to execute on. The first non-zero value in the row-wise sum intermediate 
    matrix up above represents the "decoding" time-step that triggers the cache eviction. 
    That means that any map of time-dependent lambda_t values can be masked up to that time-step.
    At that point, instead of naively masking by `g` we can multiply by the masked_lambda_t values
    (may have to transpose or something for the keys?). 

    TODO: some time-dependent implementation
    
''' 

# currently also takes care of tril mask behavior, may be unnecessary and should be tested
# as it is inefficient to constantly use it
# expects sparse_mask of dimensions B x H x N x N
# originally, lambda_val was just a float but we should now expect potential tensors of size H
def get_lambda_mask_sparse(sparse_mask, lambda_val):
    xor_sparse_mask = get_eviction_triggered(sparse_mask)

    # NOTE: final bool() and int() casts cover block-wise cases for evictions,
    #       this ensures likely compatibility with methods like paged attention
    evict_triggered = torch.sum(xor_sparse_mask, dim=-1).unsqueeze(-1).bool().int()
    
    # check for case where lambda_val is dependent on a dimension
    if torch.is_tensor(lambda_val) and (sum(lambda_val.size()) / lambda_val.dim()) > 1:
        # then check for attention head dependent or time dependent
        if lambda_val.dim() == 1:
            lambda_evict = lambda_val.view(1, lambda_val.size(-1), 1, 1) * evict_triggered
        else:
            # in this case, lambda_val should be B x N x 1 coming in, so need to view
            # with that in mind
            lambda_evict = lambda_val.view(lambda_val.size(-3), 1, lambda_val.size(-2), lambda_val.size(-1)) * evict_triggered
    else:
        lambda_evict = lambda_val * evict_triggered
    lambda_mask = lambda_evict * sparse_mask

    # need to switch zeros to ones for cumprod then switch back before dividing out a
    # lambda_val to respect initial evictions
    lambda_mask[lambda_mask == 0] = 1
    lambda_mask = torch.cumprod(lambda_mask, dim=-2) + (1e-6 * sparse_mask)
    lambda_mask[lambda_mask == 1]  = 0

    # check for case where lambda_val is dependent on a dimension
    if torch.is_tensor(lambda_val) and (sum(lambda_val.size()) / lambda_val.dim()) > 1:
        # then check for attention head dependent or time dependent
        if lambda_val.dim() == 1:
            return lambda_mask / lambda_val.view(1, lambda_val.size(-1), 1, 1)
        else:
            # need to ensure there are no non-zero values in lambda_evict
            # when using it for final division of initial decay term
            lambda_evict_div = lambda_evict
            lambda_evict_div[lambda_evict_div == 0] = 1
            return lambda_mask / lambda_evict_div
    else:
        return lambda_mask / lambda_val


# sole purpose is to provide indices of keys/values as they are evicted across
# time so that they can be reordered during time-data-dependent forward passes
def get_eviction_kv_indices(sparse_mask):
    B, H, _, _ = sparse_mask.shape
    evict_triggered = get_eviction_triggered(sparse_mask)


    # produces a series of two-element tensors acting as indices, then we split to
    # just get key indices for evictions in order across time
    evict_indices = torch.nonzero(evict_triggered, as_tuple=False)
    
    # last column post-split should contain reordered key indices, squeeze to
    # remove extra dimension post-split 
    evict_indices = evict_indices.split(1, dim=-1)[-1].squeeze()

    assert evict_indices.size(0) % B == 0, \
            "Even division of triggered token length by batch size required. " + \
            "Eviction decisions are not allowed to be dependent on batch, "    + \
            "i.e. the same number of tokens must be evicted per-batch."
    evict_indices = evict_indices.view(B, evict_indices.size(0) // B)

    return evict_indices # B x (H * S_evicted)

# provides the eviction events as what amounts to a binary mask
def get_eviction_triggered(sparse_mask):
    shifted_down_sparse_mask = torch.zeros_like(sparse_mask)
    shifted_down_sparse_mask[:, :, 1:, :] = sparse_mask[:, :, :-1, :]

    # note: non-zeros are all True, only zeros are False as baseline before xor
    #       need to add back identity matrix to account for missing diagonal
    evict_triggered_xor = torch.logical_xor(
                        sparse_mask.to(torch.int),
                        shifted_down_sparse_mask.to(torch.int), 
    ).to(torch.int)
    
    return evict_triggered_xor
