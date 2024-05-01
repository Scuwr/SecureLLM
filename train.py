import os
import random
import sys
import datetime
import argparse
import functools
from collections import OrderedDict
import json
import logging
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

import uuid
import numpy as np
from tqdm import tqdm

import wandb

import db_manager as my_db_lib
import env
import datasets
import model_selector
from utils import ParseKwargs



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # turn off tensorflow logging
logger = logging.getLogger(__name__)

my_val_db = my_db_lib.DBManager(db_path=env.db_val_path)

VALIDATION_MAX_GEN_LEN = 480  # TODO make this an argument

class MyTorchDistributed:
    '''Wrapper class for all my Torch Distributed needs to train on multiple GPUs'''
    @staticmethod
    def spawn(main_fn, world_size, addr='localhost', port='12346', backend='nccl'):
        # wrap to do init/destroy before/after user defined function. Spawn will pass rank (0,1,... for process counter) and world_size (constant)
        _barrier_seed = random.randint(0, 2**16-1)
        wrapped = functools.partial(MyTorchDistributed._wrap_main_fn, main_fn=main_fn, world_size=world_size, _barrier_seed=_barrier_seed, addr=addr, port=port, backend=backend)
        try:
            torch.multiprocessing.spawn(wrapped, nprocs=world_size, daemon=True)
        except KeyboardInterrupt as e:
            # print(e)
            MyTorchDistributed._destroy()

    @staticmethod
    def _wrap_main_fn(rank, **kwargs):
        main_fn = kwargs.pop('main_fn')
        MyTorchDistributed._init(rank=rank, **kwargs)
        main_fn(rank)
        MyTorchDistributed._destroy()

    @staticmethod
    def _init(rank, world_size, _barrier_seed, addr, port, backend, try_limit=10):
        MyTorchDistributed.barrier_seeder = random.Random(_barrier_seed)
        os.environ['MASTER_ADDR'] = addr
        os.environ['MASTER_PORT'] = port
        for _ in range(try_limit):
            try:
                torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
                break
            except RuntimeError as e:
                os.environ['MASTER_PORT'] = str(int(os.environ['MASTER_PORT']) + 1)
                print('Failed to init, trying port', os.environ['MASTER_PORT'], 'Error:', e)
        else:
            raise RuntimeError(f'Could not init process group, tried ports {port}-{int(port)+try_limit-1}')
    @staticmethod
    def _destroy():
        torch.distributed.destroy_process_group()
    @staticmethod
    def wrap_model(model, rank_gpu, find_unused_parameters=False):
        # model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        model = DistributedDataParallel(model, device_ids=[rank_gpu], output_device=rank_gpu, find_unused_parameters=find_unused_parameters, gradient_as_bucket_view=True)
        return model
    barrier_seeder: random.Random = None
    @staticmethod
    def my_barrier(file_dir='./scripts/temp/barriers/'):
        '''TORCH.DISTRIBUTED.BARRIER IS TERRIBLE AND CAUSES OOM ERRORS AFTER THE ENTIRE VALIDATION PROCESS FINISHED PERFECTLY. IM 100% SURE IT IS CAUSED BY BARRIER AND NOT SOMETHING ELSE.'''
        # three phases, draw phase, standby phase, main phase (think of all threads being in two neighbouring phases at the same time)
        # DRAW phase
        # a process enters draw phase by deleteing its .standby file and creating a .draw file (cannot delete .main as slow siblings might be looking for it from prev run)
        # a process cannot exit draw phase unless there are 8 .draw files
        # STANDBY phase
        # a process enters standby phase by deleting its .main file and creating a .standby file
        # a process cannot exit standby phase unless there are 8 .standby files, as soon as it exists it deletes its .draw file
        # MAIN phase
        # a process enters main phase by creating a .main file
        # a process cannot exit main phase unless there are 8 .main files, as soon as it exists it deletes its .standby file
        seed = MyTorchDistributed.barrier_seeder.randint(0, 10**6)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        port = os.environ['MASTER_PORT']  # to make sure this code doesn't break when multiple instances are running
        file_dir = Path(file_dir)
        file_dir.mkdir(parents=True, exist_ok=True)
        draw_file = file_dir / f'b_{rank}_{port}_{seed}.draw'
        standby_file = file_dir / f'b_{rank}_{port}_{seed}.standby'
        main_file = file_dir / f'b_{rank}_{port}_{seed}.main'
        if os.path.exists(draw_file): 
            logger.warning(f'rank {rank} tried to enter draw phase but draw file already exists, SHOULD NOT HAPPEN and could have causes a deadlock')
            os.remove(draw_file)
        # make sure no .draw files exist due to a previous crashed run or whatever
        if os.path.exists(standby_file):
            os.remove(standby_file)
        # DRAW phase start
        open(draw_file, 'a').close()
        while len([f for f in os.listdir(file_dir) if f.endswith(f'_{port}_{seed}.draw')]) < world_size:
            time.sleep(0.1)
        # DRAW phase end
        # STANDBY phase start
        if os.path.exists(main_file):
            os.remove(main_file)
        open(standby_file, 'a').close()
        while len([f for f in os.listdir(file_dir) if f.endswith(f'_{port}_{seed}.standby')]) < world_size:
            time.sleep(0.1)
        os.remove(draw_file)
        # STANDBY phase end
        # MAIN phase start
        open(main_file, 'a').close()
        while len([f for f in os.listdir(file_dir) if f.endswith(f'_{port}_{seed}.main')]) < world_size:
            time.sleep(0.1)
        # MAIN phase end
        # DONE, cannot delete main file as others are looking for it, can clean up standby file
        os.remove(standby_file)
        
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    # not sure if these are needed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)

def get_scheduler(optimizer, cos_period, warmup_steps, last_step=-1, T_mult=1, eta_min_factor=0):
    max_lr = optimizer.param_groups[0]['initial_lr'] if 'initial_lr' in optimizer.param_groups[0] else optimizer.param_groups[0]['lr']
    # cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cos_period, T_mult=T_mult, eta_min=eta_min_factor*max_lr)
    def warmup(current_step: int):
        r = float(current_step+1) / warmup_steps
        return r if r <= 1 else 1
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cos_scheduler], [warmup_steps])
    scheduler = warmup_scheduler
    # skip to last step
    for _ in range(last_step):
        scheduler.step()
    return scheduler

def get_logits_from_lm_output(hidden_state, input_ids, pad_token_id, delta=0):
    r''' delta=0 means we use the last token, delta=1 means we use the second last token, etc
    '''
    num_tokens = torch.ne(input_ids, pad_token_id).sum(-1) # 1d int tensor, number of tokens in each sentence
    sequence_lengths = (num_tokens - 1 - delta).to(hidden_state.device)
    last_token_hidden_states = hidden_state[..., torch.arange(hidden_state.shape[-3], device=hidden_state.device), sequence_lengths, :]
    # logits = lmhead(last_token_hidden_states.to(lmhead.weight.device))
    logits = last_token_hidden_states
    return logits

def get_lm_loss_from_logits(hidden_states, labels):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
    """
    # lm_logits = lmhead(hidden_states.to(lmhead.weight.device))
    lm_logits = hidden_states
    if not lm_logits.requires_grad: # error with tensor_parallel
        lm_logits.requires_grad = True
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    if torch.isnan(loss).any():
        print("Output")
        print(shift_logits.view(-1, shift_logits.size(-1)))
        print("Labels")
        print(shift_labels.view(-1))
        raise ValueError("Loss value is NaN")

    return loss

def _decode_tokens(token_ids, tokenizer):
    """
        Decode a list of tokens. Useful because llama tokenizer's decode method doesn't add spaces between tokens if decoded individually.
        Args:
            token_ids (list or tensor): A list of token ids
            tokenizer: A tokenizer that can decode token ids
        Example:
            >>> s = 'This is a test sentence'
            >>> token_ids = torch.tensor(tokenizer.encode(s), dtype=torch.long) # or token_ids = tokenizer.encode(s)
            >>> result = _decode_tokens(token_ids, tokenizer)
            >>> ''.join(result) == s
    """
    if not isinstance(token_ids, list):
        token_ids = token_ids.tolist()
    result = [None] * len(token_ids)
    result[0] = tokenizer.decode(token_ids[0])
    BACKRANGE = 2  # how many tokens to look back when decoding, 1 gave incorrect results, 2 seems to work
    for i in range(1, len(token_ids)):
        start_idx = max(0, i-BACKRANGE)
        prev         = tokenizer.decode(token_ids[start_idx:i])
        cur_and_prev = tokenizer.decode(token_ids[start_idx:i+1])
        if not cur_and_prev.startswith(prev):
            logger.warning(f'DECODING TOKENS cur_and_prev does not start with prev: {cur_and_prev} {prev}')
        result[i] = cur_and_prev[len(prev):]
        # _t = "'"
        # print(f"{_t+cur_and_prev+_t:<12} - {_t+prev+_t:<12} = {_t+result[i]+_t:<12}")
    return result

def llm_generate(model, tokenizer, dataset: datasets.DynamicBatchLoader, temprature=0.75, gen_max_len=50, gem_max_token_len=None, pbar=None, use_kv=False):
    device = next(model.parameters()).device
    model.eval()

    # clora_size = model.peft_config['clora'].n
    #clora_size = 1

    i = 0
    past_kv = None
    for batch in dataset.for_generation(tokenizer):
        max_len = max(len(d['tokens']) for d in batch)
        if pbar is not None and isinstance(pbar, tqdm):
            old_desc = pbar.desc if pbar.desc is not None else ''
            old_desc = old_desc if '|||' not in old_desc else old_desc[:old_desc.find('|||')]
            pbar.set_description(old_desc + f'||| max_len={max_len} gen_iter={i}')
            i += 1
            pbar.refresh()
        text_input = torch.full((len(batch), max_len), tokenizer.pad_id, dtype=torch.long, device=device)
        for k, d in enumerate(batch):
            text_input[k, : len(d['tokens'])] = torch.tensor(d['tokens'], dtype=torch.long, device=device)

        with torch.no_grad():
            if use_kv: 
                assert len(batch) == 1, "batch size MUST BE 1!! use_kv is set to true but batch size is not 1, padding breaks the kv cache so batch size must be 1"
                sample = batch[0]
                if 'kv_cache' in sample and sample['kv_cache'] is not None:
                    past_kv = sample['kv_cache']
                    # move to device
                    # past_kv = tuple([tuple([p.to(device) for p in kv]) for kv in past_kv])
                    text_input = torch.tensor(sample['last_token'], device=device).reshape(1, 1)
                    out = model(text_input, past_key_values=past_kv) 
                else:  # no past_kv, first iteration
                    out = model(text_input)
                assert hasattr(out, 'past_key_values'), "use_kv set to true while the model doesn't have past_key_values return (are you sure this is a hugging face model?)" 
                kv_cache = out.past_key_values
                # detach
                # kv_cache = tuple([tuple([p.detach().cpu() for p in kv]) for kv in kv_cache])
                sample['kv_cache'] = kv_cache
            else:
                out = model(text_input) 
            if hasattr(out, 'logits'):  # huggingface model
                out = out.logits

            logits = get_logits_from_lm_output(out.detach(), text_input, tokenizer.pad_id, delta=0)

            if dataset.suppressed_tokens is not None:
                    logit_mask = np.array([d['suppressed_tokens'] for d in batch])
                    logit_mask = torch.from_numpy(logit_mask).to(device)
                    logits = logits * logit_mask

            if temprature > 0:
                logits /= temprature
                probs = torch.softmax(logits, dim=1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1).to('cpu').detach()
            else:
                next_token = torch.argmax(logits, dim=1).to('cpu').detach()

        for sample, next_token in zip(batch, next_token):
            # print('260', sample['tokens'], [next_token.item()])
            new_text = tokenizer.decode(sample['tokens'] + [next_token.item()])
            sample['gen_text'] = new_text
            sample['tokens'] = tokenizer.encode(new_text, add_special_tokens=False)
            sample['last_token'] = next_token.item()
            sample['gen_len'] += 1
            if sample['sample'].is_generation_done(new_text, tokens=sample['tokens']):  # done with this generation
                sample['done'] = True
                sample['limit_hit'] = False
                yield sample
            elif sample['gen_len'] >= gen_max_len:  # sample not done, but hit gen_max_len
                sample['done'] = True
                sample['limit_hit'] = True
                yield sample
            elif gem_max_token_len is not None and len(sample['tokens']) >= gem_max_token_len:  # sample not done, but hit max_len
                sample['done'] = True
                sample['limit_hit'] = True
                yield sample
            
            # clean up cache
            if sample['done'] and use_kv:
                sample['kv_cache'] = None

def _left_padd_cache(batch_size, past_key_values, device):
    seq_length = 1
    adapter = past_key_values[0][0][0].shape[0]
    dtype = past_key_values[0][0][0].dtype

    lens_kv = [x[0][0].shape[-2] for x in past_key_values]
    # print('kv lens', lens_kv)
    past_key_values_length = max(lens_kv)
    seq_length_with_past = seq_length + past_key_values_length
    # concat accross batch dim
    concatted_kv = []
    for i in range(len(past_key_values[0])):
        concatted_kv.append([])
        for j in range(len(past_key_values[0][i])):
            shape: torch.Size = past_key_values[0][i][j].shape  # (adapter, batch_size, num_heads, sequence_length, embed_size_per_head))
            shape_list: list = list(shape)
            shape_list[1] = batch_size
            shape_list[3] = past_key_values_length
            zeros = torch.zeros((*shape_list, ), dtype=dtype, device=device)
            for k in range(batch_size):
                # print(zeros.shape, past_key_values[k][i][j].shape)
                assert zeros[:, k:k+1, :, -lens_kv[k]:, :].shape == past_key_values[k][i][j].shape
                zeros[:, k:k+1, :, -lens_kv[k]:, :] = past_key_values[k][i][j]

            concatted_kv[i].append(zeros)


    attention_mask = torch.ones(
        (batch_size, seq_length_with_past), dtype=torch.bool, device=device
    )
    # padding on the left
    for k in range(batch_size):
        pad_len = past_key_values_length - lens_kv[k]
        attention_mask[k, :pad_len] = False
    
    return concatted_kv, attention_mask

def llm_generate_kv_cache(model, tokenizer, dataset: datasets.DynamicBatchLoader, temprature=0.75, gen_max_len=50, gem_max_token_len=None, pbar=None, completion_percent=1):
    device = next(model.parameters()).device
    model.eval()

    i = 0
    for batch in dataset.for_generation(tokenizer):
        max_len = max(len(d['tokens']) for d in batch)
        if pbar is not None and isinstance(pbar, tqdm):
            old_desc = pbar.desc if pbar.desc is not None else ''
            old_desc = old_desc if '|||' not in old_desc else old_desc[:old_desc.find('|||')]
            pbar.set_description(old_desc + f'||| max_len={max_len} gen_iter={i}')
            i += 1
            pbar.refresh()

        concatted_kv = None
        while sum(sample['done'] for sample in batch)/len(batch) < completion_percent:  # generate until entire batch is done
            with torch.no_grad():
                if concatted_kv is None:  # first iteration
                    out = [model(torch.tensor(d['tokens'], device=device).unsqueeze(0)) for d in batch]
                    kv = [x.past_key_values for x in out]
                    position_ids = torch.tensor([len(d['tokens']) for d in batch], device=device)  # careful with this, new token was not added to d['tokens'] yet
                    concatted_kv, attention_mask = _left_padd_cache(batch_size=len(batch), past_key_values=kv, device=device)
                    logits = torch.stack([x.logits[0, -1] for x in out], dim=0)
                else:
                    new_tokens = torch.tensor([d['last_token'] for d in batch], device=device)
                    out = model(new_tokens.reshape(-1, 1), past_key_values=concatted_kv, attention_mask=attention_mask, position_ids=position_ids)
                    concatted_kv = out.past_key_values
                    position_ids += 1
                    attention_mask = torch.cat([attention_mask, torch.ones((*attention_mask.shape[:-1], 1), dtype=torch.bool, device=device)], dim=-1)
                    if hasattr(out, 'logits'):  # huggingface model
                        out = out.logits
                    logits = out[:, -1, :]  # left padding, so we want the last token

                if dataset.suppressed_tokens is not None:
                    logit_mask = np.array([d['suppressed_tokens'] for d in batch])
                    logit_mask = torch.from_numpy(logit_mask).to(device)
                    # print(logit_mask)
                    # print(logits[:,29900:29940])

                    logits = logits * logit_mask

                    # print(logits[:,29900:29940])
                
                # new_tokens = torch.argmax(out.logits[:, -1], dim=-1)
                # tokens = [t + [new_tokens[i].item()] for i, t in enumerate(tokens)]

                if temprature > 0:
                    logits /= temprature
                    probs = torch.softmax(logits, dim=1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1).to('cpu').detach()
                else:
                    next_token = torch.argmax(logits, dim=1).to('cpu').detach()

            for sample, next_token in zip(batch, next_token):
                if sample['done']: 
                    continue
                new_text = tokenizer.decode(sample['tokens'] + [next_token.item()])
                sample['gen_text'] = new_text
                sample['tokens'] = tokenizer.encode(new_text)
                sample['last_token'] = next_token.item()
                sample['gen_len'] += 1
                if sample['sample'].is_generation_done(new_text, tokens=sample['tokens']):  # done with this generation
                    sample['done'] = True
                    sample['limit_hit'] = False
                    yield sample
                elif sample['gen_len'] >= gen_max_len:  # sample not done, but hit gen_max_len
                    sample['done'] = True
                    sample['limit_hit'] = True
                    yield sample
                elif gem_max_token_len is not None and len(sample['tokens']) >= gem_max_token_len:  # sample not done, but hit max_len
                    sample['done'] = True
                    sample['limit_hit'] = True
                    yield sample

def clip_norm(model):
    _clip_ret = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type='2')
    # print(_clip_ret)
    # print('max', max([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]))

def single_epoch(model, tokenizer, dataset, epoch, optim=None, scheduler=None, is_train=True, device='cpu', verbose=True, sample_limit=None):
    log_prefix = 'train' if is_train else 'val'
    model.train(is_train)
    cur_sample_count = 0
    sample_count = len(dataset) if sample_limit is None else sample_limit
    pbar = tqdm(total=sample_count, dynamic_ncols=True) if verbose else None
    metrics = {'losses': []}
    for batch_idx, batch in enumerate(dataset):
        text_list = [text.get_text() for text in batch]
        token_list = [tokenizer.encode(text) if isinstance(text, str) else text for text in text_list]
        max_len = max(len(i) for i in token_list)
        text_input = torch.full((len(token_list), max_len), tokenizer.pad_id, dtype=torch.long, device=device)
        for k, txt in enumerate(token_list):
            text_input[k, : len(txt)] = torch.tensor(txt, dtype=torch.long, device=device)
        attention_mask = torch.ne(text_input, tokenizer.pad_id)

        labels = datasets.SingleSample.batch_get_labels(batch, text_input, attention_mask, tokenizer)
        with torch.set_grad_enabled(is_train):
            output_pos = model.forward(text_input)
            if hasattr(output_pos, 'logits'):  # huggingface model
                output_pos = output_pos.logits

            bs, sq_len, _ = output_pos.shape
            _, lb_len = labels.shape

            pad_vector = torch.ones(bs, sq_len - lb_len, dtype=labels.dtype).to(device) * -100
            labels = torch.concat([pad_vector, labels], dim=-1)

            # Debugging
            # mask = (labels[0,1:] > 0)
            # outs = torch.argmax(output_pos, dim=-1).detach()
            # text_input = torch.concat([pad_vector, text_input], dim=-1)
            # print(tokenizer.decode(text_input[0,1:][mask]))
            # print("===")
            # print(tokenizer.decode(outs[0,:-1][mask]))
            # time.sleep(2)

            loss = get_lm_loss_from_logits(output_pos, labels)

        if optim is not None:
            assert is_train, 'optimizer provided but is_train is False'
            optim.zero_grad()
            loss.backward()
            clip_norm(model)
            optim.step()
            if scheduler is not None:
                scheduler.step()

        metrics['losses'].append(loss.item())
        if pbar is not None:
            cur_sample_count += len(batch)
            pbar.update(len(batch) if isinstance(dataset, datasets.DynamicBatchLoader) else 1)
            pbar.set_description(f'{log_prefix} Epoch: {epoch} loss {np.mean(metrics["losses"]):.4f}')
            pbar.refresh()
        if sample_limit is not None and cur_sample_count >= sample_limit:
            break
    if pbar is not None: pbar.close()
    if wandb.run is not None:
        wandb.log({
            log_prefix+'/loss': np.mean(metrics["losses"]), 
        }, commit=False)
    return metrics["losses"]

def validate(model_lm, tokenizer, dataloader, temprature=0, gen_max_len=VALIDATION_MAX_GEN_LEN, gem_max_token_len=None, callback=None, use_kv=True, verbose=True):
    print("Using KV Cache") if use_kv else print("Not using KV Cache")
    
    random.shuffle(dataloader.dataset)
    _all_dataset_names = set(s._metadata['dataset'] for s in dataloader.dataset if s._metadata is not None and 'dataset' in s._metadata)
    _all_dataset_names = sorted(list(_all_dataset_names))
    if verbose: print(_all_dataset_names + ['no_acc', 'unk'])
    # every dataset will have a key in memories, and the value will be a dict that the dataset manages. if 'to_print' is a key in the dict, it will be printed
    memories = OrderedDict((k, {}) for k in _all_dataset_names)
    memories['no_acc'] = {'count': 0}
    memories['unk'] = {}
    all_samples = []

    pbar = tqdm(total=len(dataloader), dynamic_ncols=True) if verbose else dataloader
    leftovers = 0

    generator = llm_generate_kv_cache if use_kv else llm_generate

    for done in generator(model_lm, tokenizer, dataloader, temprature=temprature, gen_max_len=gen_max_len, gem_max_token_len=gem_max_token_len, pbar=pbar):
        leftovers += 1 if done['limit_hit'] else 0
        if hasattr(done['sample'], 'get_accuracy') and callable(done['sample'].get_accuracy):
            dataset_name = done['sample']._metadata['dataset'] if (done['sample']._metadata is not None and 'dataset' in done['sample']._metadata) else 'unk'
            memories[dataset_name] = done['sample'].get_accuracy(done['gen_text'], memories[dataset_name])
        else:
            memories['no_acc']['count'] += 1
            memories['no_acc']['to_print'] = 'no_acc:' + str(memories['no_acc']['count'])
        all_samples.append(done)
        if verbose:
            pbar.update()
            _desc = str([v['to_print'] if 'to_print' in v else '-' for v in memories.values()])
            pbar.set_description(_desc)
            pbar.refresh()
        if callback is not None:
            callback(done, memories)
    if verbose:
        pbar.close()
        print({k:{**v, 'to_save': None} for k,v in memories.items()})  # print all memories except 'to_save'
        if leftovers > 0:
            print(f'Generation did not finish for ({leftovers}/{len(dataloader)}) samples. Consider increasing gen_max_len, currently {gen_max_len}')
    memories['all_samples'] = all_samples
    return memories

def save_validation_results_if_needed(val_results: dict[str, dict], save_info):
    to_save = {}
    save_info['save_date'] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    for key, val in val_results.items():
        if 'to_save' in val:
            to_save[key] = val['to_save']
            del val['to_save']
    if len(to_save) > 0:
        json_path = str(env.val_save_path)
        json_path = my_val_db.insert_model(json_path, json.dumps(save_info))
        assert not os.path.exists(json_path), f'file already exists: {json_path}'
        with open(json_path, 'w') as f:
            json.dump(to_save, f, indent=2)
    return val_results

def start_training(**kwargs):
    full_command = ' '.join(sys.argv)
    experiment_number, sub_experiment = kwargs['experiment'].split('.')
    model_args = kwargs['model_args']
    dataset_args = kwargs['dataset_args']
    train_args = kwargs['train_args']
    device = kwargs['device']
    rank = kwargs['rank']
    world_size = kwargs['world_size']
    seed = kwargs['seed']
    save_model = kwargs['save_model']
    
    total_epochs = int(train_args.get('epochs', 10))

    my_uuid = str(uuid.uuid4())
    if rank == 0:
        print(f'   ||| Running experiment {experiment_number}/{sub_experiment} on device {device} for {total_epochs} epochs. UUID: {my_uuid} |||')
        print(f'   ||| Command: {full_command} |||')
        if not save_model:
            print('WARNING: save_model is set to False')
    
    save_info = {
        'experiment': experiment_number,
        'sub_experiment': sub_experiment,
        'my_uuid': my_uuid,
        'sample_limit': kwargs['sample_limit'],
        'model_args': model_args,
        'dataset_args': dataset_args,
        'train_args': train_args,
        'seed': seed,
        'full_command': full_command,
    }


    # seed must be the same in all processes
    set_seed(seed)
    mymodel = model_selector.main_get_model(model_args=model_args, world_size=world_size, device=device, verbose=rank==0)
    # if world_size > 1:
        # mymodel.model = mymodel.model.to('cpu')
        # mymodel.model = MyTorchDistributed.wrap_model(mymodel.model, rank_gpu=rank)

    # seed must be the same in all processes
    set_seed(seed)
    dataset_mc_train = datasets.get_dataset(dataset_args, mymodel.tokenizer, is_train=True, verbose=rank==0)
    dataset_mc_valid = datasets.get_dataset(dataset_args, mymodel.tokenizer, is_train=False, verbose=rank==0)
    assert dataset_mc_train is not None or dataset_mc_valid is not None, 'No dataset loaded'
    if rank == 0: 
        print(f'Loaded {len(dataset_mc_train) if dataset_mc_train is not None else 0} training samples')
        print(f'Loaded {len(dataset_mc_valid) if dataset_mc_valid is not None else 0} validation samples')

    if dataset_mc_train is None:  # validation only
        if rank == 0: print('No training data requested. Validation only')
        val_results = validate(mymodel.model, mymodel.tokenizer, dataset_mc_valid, verbose=rank==0)
        val_results = save_validation_results_if_needed(val_results, {**save_info, 'epoch': -1})
        return # exit
    
    if save_model:
        if not callable(getattr(mymodel, 'save', None)):
            raise ValueError('Model does not have a save method. Either set save_model=False or implement a save method in the model')

    params_to_train = [(n,p) for n,p in mymodel.model.named_parameters() if p.requires_grad]  # already set which params to train in the model
    if rank == 0: print(f'sum of params to train:{sum([p.numel() for n,p in params_to_train]):,}')
    optim = torch.optim.AdamW([p for n,p in params_to_train], lr=float(train_args.lr), weight_decay=float(train_args.weight_decay), betas=(0.9, 0.95), eps=1e-5)
    # optim = torch.optim.Adam([p for n,p in params_to_train], lr=train_args.lr, weight_decay=train_args.weight_decay, betas=(0.9, 0.95))
    # optim = torch.optim.SGD([p for n,p in params_to_train], lr=train_args.lr)

    if 'scheduler' not in train_args or train_args.scheduler is None:
        scheduler = None
    elif train_args.scheduler == 'warmup':
        warmup_steps = int(train_args.warmup_steps)
        def warmup(current_step: int):
            r = float(current_step+1) / warmup_steps
            return r if r <= 1 else 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup)
    # elif train_args.scheduler == 'cosine':
    #     cos_period = train_args.cos_period
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, cos_period, T_mult=1, eta_min=0)
    else:
        raise ValueError(f'Unknown scheduler: {train_args.scheduler}')

    train_loss = []
    for epoch_count in range(1, total_epochs+1):
        if rank == 0: print('Epoch', epoch_count)
        cur_train_loss = single_epoch(mymodel.model, mymodel.tokenizer, dataset_mc_train, epoch=epoch_count, optim=optim, scheduler=scheduler,
                                      is_train=True, device=device, verbose=rank==0, sample_limit=kwargs['sample_limit'],)
        train_loss.append([round(i, 5) for i in cur_train_loss])

        val_results = None
        if dataset_mc_valid is not None:
            val_results = validate(mymodel.model, mymodel.tokenizer, dataset_mc_valid, verbose=rank==0)
            # remove 'all_samples', will error when saving, too much info to save
            del val_results['all_samples']
            val_results = save_validation_results_if_needed(val_results, {**save_info, 'epoch': epoch_count})
        
        if rank == 0 and save_model:
            mymodel.save({  # type: ignore ; save was checked to exist and be callable above
                            'info': {
                                'epoch': epoch_count,
                                'save_date': datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                                **save_info,
                                'metada': {
                                    'val': val_results,
                                    },
                            },
                            'hidden_info': {  # to not clutter the txt file that shows all info
                                'train_loss': train_loss,
                            }
                        }, save_path=kwargs['save_path'])

def main(rank, **kwargs):
    # set devices by incrementing on the single device given 
    first_device_name = kwargs['device']
    if first_device_name == 'cpu':
        kwargs['device'] = 'cpu'
    elif ',' not in first_device_name:
        device_idx = torch.cuda.device(first_device_name).idx
        kwargs['device'] = 'cuda:' + str(device_idx + rank)
    else:
        device_idx = torch.cuda.device(first_device_name.split(',')[rank]).idx
        kwargs['device'] = 'cuda:' + str(device_idx)
    kwargs['rank'] = rank
    return start_training(**kwargs)


if __name__ == '__main__':
    # print('-----DEBUGGING-----DEBUGGING-----DEBUGGING\n'*50)
    # sys.argv = """
    # train.py
    # --dataset_args
    #     train_str="schema_2:25000"
    #     val_str="schema_2_val:500"
    #     max_inp_matrix_size=8000
    # --model_args
    #     name=llama-2-7b
    #     max_seq_len=1024
    #     max_batch_size=32
    #     type=lora
    #     lora_r=8
    #     lora_alpha=32
    #     lora_dropout=0.1
    # --train_args
    #     lr=2e-4
    #     weight_decay=0.002
    # --experiment "4.2"
    # --world_size 1
    # --seed 2
    # --device cuda:7
    # """.split()  # for debugging
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_args', nargs='*', action=ParseKwargs, required=True, help='Arguments to pass to the dataset')
    parser.add_argument('--model_args', nargs='*', action=ParseKwargs, required=True, help='Arguments to pass to the model')
    parser.add_argument("--train_args", nargs='*', action=ParseKwargs, required=False, default=ParseKwargs.dotdict(), help="Arguments to pass to the training function")
    parser.add_argument("--device", required=True, help="GPU to use")
    parser.add_argument("--world_size", required=True, help="Number of GPUs")
    parser.add_argument("--save_model", default='True', choices=['True', 'False'], help="Save model weights after each epoch. Default: True")
    parser.add_argument("--seed", default=1, help="Seed to use")
    parser.add_argument("--samples_per_epoch", default=None, help="Number of samples per epoch. Default: All samples in dataset")
    parser.add_argument('--experiment', type=str, help='Experiment number example: "3.1" ')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save model')
    # parser.add_argument('--mapping', type=int, default=0, help='Select type of mapping (0 is no mapping)')
    # parser.add_argument('--pseudo', action='store_true', help='Use 6NF Pseudocode questions')
    # parser.add_argument('--sample_list', nargs='*', default=['1', '2', '3', 'union12', 'union13', 'union23', 'union123'], help='List of validation sets')
    # parser.add_argument('--train_sample_size', type=int, default=1000, help='Train Sample size')
    # parser.add_argument('--val_sample_size', type=int, default=100, help='Validate Sample size')

    args = parser.parse_args()

    # mapping_text = ""
    # if args.mapping > 0:
    #     mapping_text = f"~mapping={args.mapping}"

    # pseudo_text = ""
    # if args.pseudo:
    #     pseudo_text = "pseudo"

    # args.dataset_args.train_str = ','.join(f'schema{pseudo_text}_{i}{mapping_text}:{args.train_sample_size}' for i in args.sample_list)
    # args.dataset_args.val_str = ','.join(f'schema{pseudo_text}_{i}_val{mapping_text}:{args.val_sample_size}' for i in args.sample_list)

    main_args = {
        'dataset_args': args.dataset_args,
        'model_args': args.model_args,
        'train_args': args.train_args,
        'device': args.device,
        'world_size': int(args.world_size),
        'save_model': args.save_model.lower() == "true",
        'seed': int(args.seed),
        'sample_limit': int(args.samples_per_epoch) if args.samples_per_epoch is not None else None,
        'experiment': args.experiment,
        'save_path': args.save_path,
    }
    print('main_args', main_args)
    # make main_fn only accept parameters (rank), this is the function spawn wants
    main_fn = functools.partial(main, **main_args)
    MyTorchDistributed.spawn(main_fn, world_size=main_args['world_size'])

