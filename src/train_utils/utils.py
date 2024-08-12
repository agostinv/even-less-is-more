import sys
from data_processing import get_c4, get_wikitext2
from annotated_models.llama import get_annotated_llama
from annotated_models.falcon import get_annotated_falcon
from masks import get_A_mask, get_h2o_mask
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from train_utils.utils import *

import math

def get_activations_layer(model, layer, dataloader, batches):
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
            qs_all.append(batch_int_values[layer]['Q'])
            ks_all.append(batch_int_values[layer]['K'])
            vs_all.append(batch_int_values[layer]['V'])
            os_all.append(batch_int_values[layer]['O'])
            
        del inputs, labels, outputs
        torch.cuda.empty_cache()

    qs_all = torch.cat(qs_all).transpose(1, 2).flatten(2)
    ks_all = torch.cat(ks_all).transpose(1, 2).flatten(2)
    vs_all = torch.cat(vs_all).transpose(1, 2).flatten(2)
    os_all = torch.cat(os_all)

    datasize = (sys.getsizeof(qs_all.storage()) + sys.getsizeof(ks_all.storage()) + sys.getsizeof(vs_all.storage()) + sys.getsizeof(os_all.storage())) / 1024**3
    print('Data size: {:.3f}GB'.format(datasize))

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


def h2o_attn_weights(attn_weights, heavy_budget, recent_budget, multi_query):
    attn_mask = get_h2o_mask(attn_weights, heavy_budget, recent_budget, multi_query=multi_query)
    attn_weights = (attn_weights * attn_mask) + ((~attn_mask) * torch.finfo(attn_weights.dtype).min)
    #print(attn_mask.shape)
    return attn_weights, torch.logsumexp(attn_weights, -1, keepdim=True), attn_mask

def A_attn_weights(attn_weights, heavy_budget, recent_budget):
    attn_mask = get_A_mask(attn_weights, heavy_budget, recent_budget)
    attn_weights = (attn_weights * attn_mask) + ((~attn_mask) * torch.finfo(attn_weights.dtype).min)
    return attn_weights, torch.logsumexp(attn_weights, -1, keepdim=True), attn_mask