import sys
from data_processing import get_c4, get_wikitext2
from annotated_models.llama import get_annotated_llama
from annotated_models.falcon import get_annotated_falcon
from masks import get_A_mask, get_h2o_mask, get_lambda_mask_sparse
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

import math

def mem_eff_get_activations_layer(model, layer, dataloader, batches, bsz, num_heads, seq_len, head_dim, permute=True):
    '''
    Collect Q, K, V, and O for a particular layer across a number of batches.

    This implementation permutes the data before concatenating and pre-allocates tensors to avoid
    excessive RAM consumption.
    '''

    # dataloader should be a list based on data_processing.py
    # permutation only intended to be turned off for tests
    if permute:
        random.shuffle(dataloader)

    model.eval() 
    
    inputs, _ = next(iter(dataloader))
    total_size = batches * bsz
    
    qs_all = torch.empty((total_size, num_heads, seq_len, head_dim), device=model.model.device)
    ks_all = torch.empty((total_size, num_heads, seq_len, head_dim), device=model.model.device)
    vs_all = torch.empty((total_size, num_heads, seq_len, head_dim), device=model.model.device)
    os_all = torch.empty((total_size, num_heads, seq_len, head_dim), device=model.model.device)

    for i, (inputs, labels) in enumerate(dataloader):
        if i == batches:
            break
        print(f'Layer {layer}, collecting batch {i+1} / {batches}')
        
        with torch.inference_mode(): 
            
            inputs = inputs.to(model.model.device)
            outputs, batch_int_values = model(inputs)
            qs_all[i * bsz:(i + 1) * bsz, :, :, :] = batch_int_values[layer]['Q']
            ks_all[i * bsz:(i + 1) * bsz, :, :, :] = batch_int_values[layer]['K']
            vs_all[i * bsz:(i + 1) * bsz, :, :, :] = batch_int_values[layer]['V']
            os_all[i * bsz:(i + 1) * bsz, :, :, :] = batch_int_values[layer]['O']
            
        del inputs, labels, outputs
        torch.cuda.empty_cache()

    # still have to reshape, could be done earlier but these just view the data a opposed to copying
    qs_all = qs_all.transpose(1, 2).flatten(2)
    ks_all = ks_all.transpose(1, 2).flatten(2)
    vs_all = vs_all.transpose(1, 2).flatten(2)

    datasize = (sys.getsizeof(qs_all.storage()) + sys.getsizeof(ks_all.storage()) + sys.getsizeof(vs_all.storage()) + sys.getsizeof(os_all.storage())) / 1024**3
    print('Data size: {:.3f}GB'.format(datasize))

    return qs_all, ks_all, vs_all, os_all

# inefficient use of RAM due to torch.cat(...)
def get_activations_layer(model, layer, dataloader, batches, permute=True):
    '''
    Collet Q, K, V, and O for a particular layer across a number of batches.
    '''
    model.eval() 
    qs_all = []
    ks_all = []
    vs_all = []
    os_all = []
    
    for i, (inputs, labels) in enumerate(dataloader):
        if i == batches:
            break
        print(f'Layer {layer}, collecting batch {i+1} / {batches}')
        
        with torch.inference_mode(): 
            
            inputs = inputs.to(model.model.device)
            outputs, batch_int_values = model(inputs)
            print(batch_int_values[layer]['Q'].shape)
            qs_all.append(batch_int_values[layer]['Q'])
            ks_all.append(batch_int_values[layer]['K'])
            vs_all.append(batch_int_values[layer]['V'])
            os_all.append(batch_int_values[layer]['O'])
            
        del inputs, labels, outputs
        torch.cuda.empty_cache()

    qs_all = torch.cat(qs_all).transpose(1, 2).flatten(2)
    print(qs_all.shape)
    ks_all = torch.cat(ks_all).transpose(1, 2).flatten(2)
    vs_all = torch.cat(vs_all).transpose(1, 2).flatten(2)
    os_all = torch.cat(os_all)

    datasize = (sys.getsizeof(qs_all.storage()) + sys.getsizeof(ks_all.storage()) + sys.getsizeof(vs_all.storage()) + sys.getsizeof(os_all.storage())) / 1024**3
    print('Data size: {:.3f}GB'.format(datasize))

    if permute:
        rand_perm = torch.randperm(len(qs_all))
        qs_all = qs_all[rand_perm]
        ks_all = ks_all[rand_perm]
        vs_all = vs_all[rand_perm]
        os_all = os_all[rand_perm]

    return qs_all, ks_all, vs_all, os_all


class QKVODataset(Dataset):
    '''
    Simple PyTorch dataset of Q, K, V, and O.
    '''
    def __init__(self, Q, K, V, O):
        self.Q = Q
        self.K = K
        self.V = V
        self.O = O

    def __len__(self):
        return len(self.Q)

    def __getitem__(self, idx):
        return self.Q[idx].float(), self.K[idx].float(), self.V[idx].float(), self.O[idx].float()

def get_target_attn(config, q, k, attn_mask):
    target_attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(config.hidden_size // config.num_attention_heads)
    
    attn_weights = (attn_mask * target_attn) + ((~attn_mask) * torch.finfo(target_attn.dtype).min)
    target_attn = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    return target_attn, attn_weights

def get_target_attn_out(config, q, k, v, attn_mask, multi_query):
    B, S, D = q.shape
    q = q.reshape(B, S, config.num_attention_heads, D // config.num_attention_heads).transpose(1, 2)
    if not multi_query:
        k = k.reshape(B, S, config.num_attention_heads, D // config.num_attention_heads).transpose(1, 2)
        v = v.reshape(B, S, config.num_attention_heads, D // config.num_attention_heads).transpose(1, 2)
    else:
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
    
    attn, attn_weights = get_target_attn(config, q, k, attn_mask)
    
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).flatten(2)
    return out, attn, attn_weights

def h2o_attn_out(config, q, k, v, attn_weights, heavy_budget, recent_budget, multi_query):
    attn_mask = get_h2o_mask(attn_weights, heavy_budget, recent_budget, multi_query=multi_query)
    out, _, attn_weights = get_target_attn_out(config, q, k, v, attn_mask,  multi_query=multi_query)
    return out, torch.logsumexp(attn_weights, -1, keepdim=True), attn_mask

def A_attn_out(config, q, k, v, attn_weights, heavy_budget, recent_budget, multi_query):
    attn_mask = get_A_mask(attn_weights, heavy_budget, recent_budget)
    out, _, attn_weights = get_target_attn_out(config, q, k, v, attn_mask, multi_query=multi_query)
    return out, torch.logsumexp(attn_weights, -1, keepdim=True), attn_mask


def h2o_attn_weights(attn_weights, heavy_budget, recent_budget, multi_query, lambda_gating, lambda_constant):
    attn_mask = get_h2o_mask(attn_weights, heavy_budget, recent_budget, multi_query=multi_query)
    
    # addition of lambda gating-based decay for sparse mask
    if lambda_gating == "constant":
        sparse_mask = get_lambda_mask_sparse(attn_mask, lambda_constant)
    elif lambda_gating == "time-dependent":
        raise NotImplementedError("Time-dependent lambda gating not implemented yet.")
    
    attn_weights = (attn_weights * sparse_mask) + ((~attn_mask) * torch.finfo(attn_weights.dtype).min)
    return attn_weights, torch.logsumexp(attn_weights, -1, keepdim=True), attn_mask

def A_attn_weights(attn_weights, heavy_budget, recent_budget, lambda_gating, lambda_constant):
    attn_mask = get_A_mask(attn_weights, heavy_budget, recent_budget)
    
    # addition of lambda gating-based decay for sparse mask
    if lambda_gating == "constant":
        sparse_mask = get_lambda_mask_sparse(attn_mask, lambda_constant)
    elif lambda_gating == "time-dependent":
        raise NotImplementedError("Time-dependent lambda gating not implemented yet.")

    attn_weights = (attn_weights * sparse_mask) + ((~attn_mask) * torch.finfo(attn_weights.dtype).min)
    return attn_weights, torch.logsumexp(attn_weights, -1, keepdim=True), attn_mask