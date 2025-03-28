import os
import argparse
import numpy as np
import random
import gc
import yaml
from tqdm import tqdm

from data_processing import get_c4, get_wikitext2
from annotated_models.llama import get_annotated_llama
from annotated_models.falcon import get_annotated_falcon
from custom_attention import KernelizedHeadAttention

from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM

import torch
from torch import optim
from utils.train_utils import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def reset_seed(offset: int):
    torch.manual_seed(0 + offset)
    np.random.seed(0 + offset)
    random.seed(0 + offset)
    set_seed(42 + offset)


models_sizes_dict = {
    'llama2': ['7b', '13b', '70b'],
    'falcon': ['7b', '40b'],
}

hugging_name_dict = {
    'llama2': lambda x: f'meta-llama/Llama-2-{x}-hf', 
    'falcon': lambda x: f'tiiuae/falcon-{x}',
}

o_proj_dict = {
    'llama2': lambda l: l.self_attn.o_proj, 
    'falcon': lambda l: l.self_attention.dense,
}

annotated_functions = {
    'llama2': get_annotated_llama,
    'falcon': get_annotated_falcon,
}

get_annotated_layers_functions = {
    'llama2': lambda m: m.model.model.layers,
    'falcon': lambda m: m.model.transformer.h,
}

mq_models = ['falcon']

criterion_mse = torch.nn.MSELoss()
scaler = torch.amp.GradScaler()


def train(net, config, trainloader, optimizer, attn_mask, heavy_budget, recent_budget, fix_heavy_to_initial_tokens, o_proj, multi_query, lambda_constant, attention_score_decay=1.0, half_precision=False):

    net.train()
    train_loss = 0.0
    
    for j, (x_t, q, k, v, o) in enumerate(trainloader):
        x_t = x_t.to(device)
        q = q.to(device)
        k = k.to(device)
        v = v.to(device)
        o = o.to(device)

        with torch.amp.autocast(device_type=device):
            target_out, target_attn, target_attn_weights = get_target_attn_out(config, q, k, v, attn_mask, multi_query)

            if fix_heavy_to_initial_tokens:
                sparse_attn_weights, sparse_norms_lse, sparse_mask = A_attn_weights(target_attn_weights, heavy_budget, recent_budget)
            else:
                sparse_attn_weights, sparse_norms_lse, sparse_mask = h2o_attn_weights(target_attn_weights, heavy_budget, recent_budget, multi_query, attention_score_decay)
            
            lr_mask = attn_mask * (~sparse_mask)
            
            pred_out = net(x_t, q, k, v, lr_mask, sparse_norms_lse, sparse_attn_weights, lambda_constant, half_precision, batch_idx=j)
            
            if o_proj is not None:
                loss_start = heavy_budget + recent_budget
                pred_out = o_proj(pred_out)[:, loss_start:]
                target_out = o[:, loss_start:]

            loss = criterion_mse(pred_out, target_out)
            
        train_loss += loss.item() 
        scaler.scale(loss).backward()

        # attempting to deal with untenable weight growth in certain configurations
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update() 
        optimizer.zero_grad()

    train_loss /= len(trainloader)
    return train_loss

  
def validate(net, config, valloader, attn_mask, heavy_budget, recent_budget, fix_heavy_to_initial_tokens, o_proj, multi_query, baseline_hh, baseline_recent, lambda_constant, attention_score_decay=1.0, half_precision=False):

    val_loss = 0.0
    baseline_val_loss = 0.0
    net.eval()
    for j, (x_t, q, k, v, o) in enumerate(valloader):
        with torch.inference_mode():
            x_t = x_t.to(device)
            q = q.to(device)
            k = k.to(device)
            v = v.to(device)
            o = o.to(device)

            target_out, target_attn, target_attn_weights = get_target_attn_out(config, q, k, v, attn_mask, multi_query)
            if fix_heavy_to_initial_tokens:
                sparse_attn_weights, sparse_norms_lse, sparse_mask = A_attn_weights(target_attn_weights, heavy_budget, recent_budget)
                baseline_sparse_out, _, _ = A_attn_out(config, q, k, v, target_attn_weights, baseline_hh, baseline_recent, multi_query)
            else:
                sparse_attn_weights, sparse_norms_lse, sparse_mask = h2o_attn_weights(target_attn_weights, heavy_budget, recent_budget, multi_query, attention_score_decay)
                baseline_sparse_out, _, _ = h2o_attn_out(config, q, k, v, target_attn_weights, baseline_hh, baseline_recent, multi_query, attention_score_decay)

            lr_mask = attn_mask * (~sparse_mask)
            
            pred_out = net(x_t, q, k, v, lr_mask, sparse_norms_lse, sparse_attn_weights, lambda_constant)
            
            loss_start = heavy_budget + recent_budget
            pred_out = o_proj(pred_out)[:, loss_start:]
            target_out = o[:, loss_start:]
            baseline_out = o_proj(baseline_sparse_out)[:, loss_start:]
            
            loss = criterion_mse(pred_out, target_out)
            baseline_loss = criterion_mse(baseline_out, target_out)
            val_loss += loss.item() 
            baseline_val_loss += baseline_loss.item() 
            
    val_loss /= len(valloader) 
    baseline_val_loss /= len(valloader) 
    return val_loss, baseline_val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    # Basic Configs
    parser.add_argument('--save_dir', type=str, default='../checkpoints')
    parser.add_argument('--model_name', type=str, default='llama2')
    parser.add_argument('--model_size', type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--random-seed-offset", type=int, default=0)
     
    # Data Collection 
    parser.add_argument('--seq_len', type=int, default=-1) 
    parser.add_argument('--sampling_batch_size', type=int, default=1)
    parser.add_argument('--seqs_to_collect', type=int, default=512)
    parser.add_argument('--half_precision', action='store_true') # QKVO will be collect in half precision if this is toggled
    
    # Sparse Cache 
    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)
    parser.add_argument("--budget-config", type=str, default=None) # for budget config file, overrides other options
    parser.add_argument("--fix_heavy_to_initial_tokens", action='store_true') # for Lambda masking

    # Kernels
    parser.add_argument("--ker_dim", type=int, default=8)
    parser.add_argument("--ker_hid", type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Training
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2)

    # Training Parallelism
    parser.add_argument('--from_layer', type=int, default=0)
    parser.add_argument('--to_layer', type=int, default=9999)

    parser.add_argument('--debug', action='store_true') # for wikitext versus c4 loading

    parser.add_argument('--lambda-gating', type=str, choices=[None, 'constant', 'learned-constant', 'learned-constant-head', 'time-dependent', 'time-data-dependent'], default=None)
    parser.add_argument('--lambda-constant', type=float, default=1.0)

    parser.add_argument('--attention-score-decay', type=float, default=1.0) # enables attention accumulation decay

    parser.add_argument('--kernel-fn', type=str, choices=["LESS", "Hedgehog", "Dijiang"], default="LESS")

    args = parser.parse_args()

    save_dir = args.save_dir
    model_name = args.model_name
    model_size = args.model_size
    
    seq_len = args.seq_len
    sampling_batch_size = args.sampling_batch_size
    batches_to_collect = args.seqs_to_collect // sampling_batch_size
    
    heavy_ratio = args.heavy_ratio
    recent_ratio = args.recent_ratio
    fix_heavy_to_initial_tokens = args.fix_heavy_to_initial_tokens

    ker_dim = args.ker_dim
    ker_hid = args.ker_hid
    dropout = args.dropout

    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size

    layer_start = args.from_layer
    layer_end = args.to_layer
    
    debug = args.debug   
 
    device = args.device

    lambda_gating = args.lambda_gating
    lambda_constant = args.lambda_constant

    attention_score_decay = args.attention_score_decay
    random_seed_offset = args.random_seed_offset

    kernel_fn = args.kernel_fn

    assert heavy_ratio >= 0 and heavy_ratio <= 1 and recent_ratio >= 0 and recent_ratio <= 1

    try:
        os.makedirs(save_dir)
    except FileExistsError:
        # directory already exists
        pass
    

    model_size_name = models_sizes_dict[model_name][model_size]
    
    print("Using ", hugging_name_dict[model_name](model_size_name))
    tokenizer = AutoTokenizer.from_pretrained(hugging_name_dict[model_name](model_size_name))
    model = AutoModelForCausalLM.from_pretrained(hugging_name_dict[model_name](model_size_name))
    print('Model loaded.')

    for p in model.parameters():
        p.requires_grad = False

    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads
    if seq_len == -1:
        seq_len = config.max_position_embeddings
    print(f"Sanity check for transformer version update: seq_len is set to {seq_len}")

    # debugging is faster with wt
    if not debug:
        trainloader = get_c4(nsamples=batches_to_collect, seed=(0 + random_seed_offset), seqlen=seq_len, tokenizer=tokenizer, batch_size=sampling_batch_size)
    else:
        trainloader = get_wikitext2(nsamples=batches_to_collect, seed=(0 + random_seed_offset), seqlen=seq_len, tokenizer=tokenizer, batch_size=sampling_batch_size)
    
    model = annotated_functions[model_name](model,[i for i in range(layer_start, layer_end)])

    attn_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len)), diagonal=0).bool().to(device)

    multi_query = model_name in mq_models

    if args.half_precision:
        model.half()
    
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_gb = (param_size + buffer_size) / 1024**3
    print('Model size: {:.3f}GB'.format(size_all_gb))

    heavy_budget = int(heavy_ratio * config.max_position_embeddings)
    recent_budget = int(recent_ratio * config.max_position_embeddings)
    if fix_heavy_to_initial_tokens:
        # Lambda
        baseline_hh = heavy_budget
        baseline_recent = int((recent_ratio + 0.5 * (ker_dim / seq_len)) * config.max_position_embeddings)
    else:
        # H2O    
        baseline_hh = int((heavy_ratio + 0.25 * (ker_dim / seq_len)) * config.max_position_embeddings)
        baseline_recent = int((recent_ratio + 0.25 * (ker_dim / seq_len)) * config.max_position_embeddings)
    

    
    for li, l in enumerate(get_annotated_layers_functions[model_name](model)):
        if li < layer_start or li >= layer_end:
            continue
        print(f'STARTING LAYER {li}')
        reset_seed(random_seed_offset)

        model.toggle_layer(li)

        # slightly inefficient to open file over and over again, but it's relatively small so shouldn't be a big deal
        if args.budget_config is not None:
            print(f"WARNING: Budget config file {args.budget_config} is being used. Any otherwise specified budget values will be ignored.")
            with open(args.budget_config, 'r') as file:
                budget_data = yaml.load(file, Loader=yaml.FullLoader)

                assert model_name.lower() in budget_data['model'], f"Model {model_name} doesn't match contents of config."
                assert model_size_name.lower() in budget_data['model'], f"Model size set to {model_size_name} doesn't match contents of config."
                assert f'layer_{li}' in budget_data['layers'].keys(), f"Didn't find layer {li} in budget config when it was expected. " \
                    + "Double check config and expected number of layers for model."

                fixed_budget = int(budget_data['layers'][f'layer_{li}']['fixed_budget'])
                heavy_budget = int(budget_data['layers'][f'layer_{li}']['heavy_budget'])
                recent_budget = int(budget_data['layers'][f'layer_{li}']['recent_budget'])
                print(f"Using fixed budget of fixed budget of {fixed_budget}, heavy budget of {heavy_budget}, and recent budget of {recent_budget} for layer {li}")
                
                fixed_budget = fixed_budget * config.max_position_embeddings 
                heavy_budget = heavy_budget * config.max_position_embeddings
                recent_budget = recent_budget * config.max_position_embeddings
        
        train_mses = torch.zeros(epochs)
        val_mses = torch.zeros_like(train_mses)

        model = model.to(device)
        xs, qs, ks, vs, os_ = mem_eff_get_activations_layer(model, li, trainloader, batches_to_collect, batch_size, num_heads, seq_len, head_dim, permute=True, half_precision=args.half_precision, multi_query=multi_query)
        model = model.cpu()
        torch.cuda.empty_cache()

        samples = len(qs)        

        # attempt to reduce size of batches, need to test against baseline
        if args.half_precision:
            xs.half()
            qs.half()
            ks.half()
            vs.half()
            os_.half()

        train_data = xQKVODataset(xs[:int(0.9 * samples)], qs[:int(0.9 * samples)], ks[:int(0.9 * samples)], vs[:int(0.9 * samples)], os_[:int(0.9 * samples)])
        val_data = xQKVODataset(xs[int(0.9 * samples):], qs[int(0.9 * samples):], ks[int(0.9 * samples):], vs[int(0.9 * samples):], os_[int(0.9 * samples):])
        print("Train samples:", len(train_data), " Val samples:", len(val_data))

        trainloader_net = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
        valloader_net = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1)

        net = KernelizedHeadAttention(
                head_dim, 
                ker_hid, 
                ker_dim, 
                num_heads, 
                dropout, 
                multi_query, 
                lambda_gating,
                kernel_fn
        )
        net.to(device).float()

        o_proj = o_proj_dict[model_name](l).float().to(device)
        #if args.half_precision:
            #o_proj.half()
            #o_proj_module = o_proj_dict[model_name](l)
            #o_proj_module = o_proj_module.half()
        
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        best = float('inf')
        for epoch in tqdm(range(epochs)):
            train_loss = train(
                net, 
                config, 
                trainloader_net, 
                optimizer, 
                attn_mask, 
                heavy_budget, 
                recent_budget, 
                fix_heavy_to_initial_tokens, 
                o_proj,
                multi_query,
                lambda_constant,
                attention_score_decay,
                half_precision=args.half_precision,
            )
            val_loss, baseline_loss = validate(
                net, 
                config, 
                valloader_net, 
                attn_mask, 
                heavy_budget, 
                recent_budget, 
                fix_heavy_to_initial_tokens, 
                o_proj,
                multi_query,
                baseline_hh, 
                baseline_recent,
                lambda_constant,
                attention_score_decay,
                half_precision=args.half_precision,
            )
            
            train_mses[epoch] = train_loss
            val_mses[epoch] = val_loss
            
            scheduler.step()

            if val_loss < best:
                best = val_loss
                net = net.cpu()
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'train_mses': train_mses, 
                    'val_mses': val_mses, 
                    'config': vars(args),
                    },
                f'{save_dir}/layer_{li}.pth')
            
                net = net.to(device)

            print("EPOCH", epoch + 1)
            print("Train loss:", train_loss)
            print("Val loss: ", val_loss)
            print("Best val loss: ", best)
            print("Baseline+ val loss: ", baseline_loss)

        if lambda_gating == "learned-constant" or lambda_gating == "learned-constant-head":
            print(f"Final learned gate decay values: {net.lambda_val}")

        if args.half_precision:
            o_proj_module = o_proj_dict[model_name](l)
            o_proj_module = o_proj_module.half()
        
        # forceful attempt at garbage collection to manage memory consumption
        del qs, ks, vs, os_
        torch.cuda.empty_cache()
        gc.collect()
        
