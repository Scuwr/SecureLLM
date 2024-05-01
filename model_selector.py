from pathlib import Path
import os
import json
import time
import datetime
import sys

import torch
from transformers import AutoTokenizer

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

import env
import db_manager as my_db_lib
from utils import ParseKwargs
from modeling.llama import Llama, ModelArgs, Tokenizer, Transformer


my_db = my_db_lib.DBManager(db_path=env.db_path)


class MyModel:
    """Base class for models"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

def record_new_model(checkpoint, save_model=True, info_key='info', save_path=None):
    model_path = str(env.model_paths.save_model_path)
    model_path = my_db.insert_model(model_path, json.dumps(checkpoint[info_key]))
    print(f"Saving to location: {model_path}")

    if save_model:
        assert not os.path.exists(model_path), f'file already exists: {model_path}'
        torch.save(checkpoint, model_path)

    if save_path is not None:  # save to a different location
        if os.path.exists(save_path):
            print(f'overwriting: {save_path}')
        torch.save(checkpoint, save_path)
        print(f'also saved to: {save_path}')



def main_get_model(model_args, device, world_size, verbose=False) -> MyModel:
    """Returns a model and tokenizer based on model_args"""
    # Make sure to set frozen parameters to not require gradients, optimizer will only update parameters with .requires_grad = True
    if model_args.type == 'adapter':
        return _my_fair_llama.init(model_args, world_size=world_size, device=device, verbose=verbose)
    elif model_args.type == 'lora' or model_args.type == 'ia3' or model_args.type == 'prefix' or model_args.type == 'p_tuning':
        return _my_hf_peft_llama.init(model_args, world_size=world_size, device=device, verbose=verbose)
    else:
        raise ValueError(f'Unknown model_args.type: {model_args.type}')

class _my_fair_llama(MyModel):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def save(self, d, save_path=None):
        # all([k1==k2 and torch.equal(v1,v2) for (k1,v1),(k2,v2) in zip(mymodel1.model.state_dict().items(), mymodel1.model.named_parameters())])
        # BUG: state_dict() has requires_grad=False for all parameters, cant use it for saving
        d['parameters'] = {k:v for k,v in self.model.named_parameters() if v.requires_grad}
        record_new_model(d, save_model=True, info_key='info', save_path=save_path)

    @staticmethod
    def init(model_args, world_size, device, verbose=True):
        llama_dir = env.model_paths.vanilla_llama_dir
        tokenizer_path = str(env.model_paths.vanilla_tokenizer_path)

        model_parallel_size = world_size
        if not model_parallel_is_initialized():
            initialize_model_parallel(model_parallel_size)

        if device != 'cpu' and device != torch.device('cpu'):
            torch.cuda.set_device(device)
        # seed must be the same in all processes
        # set_seed(seed)  # already set in main.py

        start_time = time.time()
        ckpt_dir = llama_dir / model_args.name
        checkpoints = sorted(ckpt_dir.glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]

        try:
            checkpoint  # in case it's already loaded in a juptyer notebook
        except NameError:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        llama_args = ModelArgs(
            max_seq_len=int(model_args.max_seq_len),
            max_batch_size=int(model_args.max_batch_size),
            use_cache=False,
            use_adapter=True,
            adapter_len=int(model_args.adapter_len),
            adapter_layer=int(model_args.adapter_layer),
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        llama_args.vocab_size = tokenizer.n_words
        tokenizer.pad_id = tokenizer.eos_id
        print('Running float16 instead of bfloat16 to avoid Error "triu_tril_cuda_template" not implemented for BFloat16')
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # torch.set_default_dtype(torch.bfloat16)
        model = Transformer(llama_args)
        model.load_state_dict(checkpoint, strict=False)
        # freeze all parameters except adapter and output_linear
        for n, p in model.named_parameters():
            if "adapter" in n or "output_linear" in n or "lora" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        model.to(device)
        if verbose: print(f"Llama loaded and initialized in {time.time() - start_time:.2f} seconds")

        tokenizer._original_encode = tokenizer.encode
        tokenizer.encode = lambda x: tokenizer._original_encode(x, bos=False, eos=False)

        if hasattr(model_args, 'load') and model_args.load is not None:
            load_weights(model, model_args.load)

        return _my_fair_llama(model, tokenizer)

class _my_hf_peft_llama(MyModel):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def save(self, d, save_path=None):
        d['parameters'] = {k:v for k,v in self.model.named_parameters() if v.requires_grad}
        record_new_model(d, save_model=True, info_key='info', save_path=save_path)

    @staticmethod
    def init(model_args, world_size, device, verbose=True):
        from transformers import LlamaTokenizer, LlamaConfig
        from modeling import modeling_llama_hf

        if model_args.name == 'llama-2-7b':
            model_path = env.model_paths.llama_hf_7b
        elif model_args.name == 'llama-2-70b':
            model_path = env.model_paths.llama_hf_70b
        elif model_args.name == 'sqlcoder-7b-2':
            model_path = env.model_paths.sqlcoder_7b_2
        else:
            raise ValueError(f'Unknown model_args.name: {model_args.name}')

        # bfloat16 for training
        torch.set_default_dtype(torch.bfloat16)

        if model_args.name == 'llama-3-8b-chat':
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_id = tokenizer.eos_token_id
        else:
            tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=True)
            tokenizer.pad_id = 0 # TODO check if correct
            tokenizer._original_encode = tokenizer.encode
            tokenizer.encode = lambda x: tokenizer._original_encode(x, add_special_tokens=False)  # skip <s> added to the beginning decoded text
            # tokenizer._original_decode = tokenizer.decode
            # tokenizer.decode = lambda x: tokenizer._original_decode(x, skip_special_tokens=True)  # skip <s> added to the beginning decoded text

        config = LlamaConfig.from_pretrained(model_path)
        # print('START', device)
        model = modeling_llama_hf.LlamaForCausalLM.from_pretrained(model_path, config=config)
        # print('END', device)
        # set dtype to float16
        model.half()

        if world_size > 1 and hasattr(model_args, 'use_tp') and model_args.use_tp:
            path2add = Path('./tensor_parallel/src/').resolve()
            assert path2add.exists()
            if str(path2add) not in sys.path: sys.path.insert(0, str(path2add))
            import tensor_parallel as tp

            # In distrubited returns: Tuple[nn.Module, Collection[str]]: Shard and a set of modified parameter names after modification
            # tpmodel = tp.tensor_parallel(model, [device], distributed=True)
            # model = tpmodel[0]
            # tp_modified = tpmodel[1]

            model = tp.tensor_parallel(model, [i for i in range(int(model_args.use_tp))], distributed=False)
            tp_modified = None

            lora_target_modules = ['q_proj.tp_wrapped_module', 'v_proj.tp_wrapped_module']  # target modules when model is wrapped by tp
        else:
            tp_modified = None
            lora_target_modules = ['q_proj', 'v_proj']  # default for llama if target_modules=None, look at peft.utils.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

        if model_args.type == 'lora':
            if hasattr(model_args, 'lora_r') and model_args.lora_r is not None:
                from peft import get_peft_model, LoraConfig
                peft_config = LoraConfig(
                    # inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
                    inference_mode=False, r=int(model_args.lora_r), lora_alpha=int(model_args.lora_alpha), lora_dropout=float(model_args.lora_dropout),
                    target_modules=lora_target_modules,
                )
                model = get_peft_model(model, peft_config)
            else:
                for param in model.parameters():
                    param.requires_grad = False

        if model_args.type == 'ia3':
            if hasattr(model_args, 'ia3') and model_args.ia3 is not None:
                from peft import IA3Config, get_peft_model
                peft_config = IA3Config(
                    inference_mode=False,
                    target_modules=lora_target_modules,
                )
                model = get_peft_model(model, peft_config)

        if model_args.type == 'prefix':
            if hasattr(model_args, 'num_virtual_tokens') and model_args.num_virtual_tokens is not None:
                from peft import get_peft_model, PrefixTuningConfig
                peft_config = PrefixTuningConfig(
                    peft_type="PREFIX_TUNING",
                    task_type="CAUSAL_LM",
                    num_virtual_tokens=int(model_args.num_virtual_tokens),
                    inference_mode = False
                )

                model = get_peft_model(model, peft_config)
            else:
                for param in model.parameters():
                    param.requires_grad = False

        if model_args.type == 'p_tuning':
            if hasattr(model_args, 'num_virtual_tokens') and model_args.num_virtual_tokens is not None:
                from peft import get_peft_model, PromptEncoderConfig
                peft_config = PromptEncoderConfig(
                    peft_type="P_TUNING",
                    task_type="CAUSAL_LM",
                    num_virtual_tokens=int(model_args.num_virtual_tokens),
                    inference_mode = False
                )

                model = get_peft_model(model, peft_config)
            else:
                for param in model.parameters():
                    param.requires_grad = False

        model.to(device)
        if device != 'cpu' and device != torch.device('cpu'):
            torch.cuda.set_device(device)

        if hasattr(model_args, 'load') and model_args.load is not None:
            load_weights(model, model_args.load)

        result = _my_hf_peft_llama(model, tokenizer)
        if tp_modified is not None:
            result.tp_modified = tp_modified
        return result

def load_weights(model, load_path):
    checkpoint = torch.load(load_path, map_location="cpu")
    assert len(checkpoint['parameters']) > 0, f'no parameters found in {load_path}'
    r = model.load_state_dict(checkpoint['parameters'], strict=False)
    assert len(r.unexpected_keys) == 0, f'unexpected_keys: {r.unexpected_keys}'
    print(f'Loaded weights from {load_path}')

def fuse_adapters(sources):
    NUM_ADAPTER_LAYERS = 36
    adapter_filter_dict = lambda x: {k:v for k,v in x.items() if 'adapter' in k}
    get_dict = lambda m: m.state_dict() if isinstance(m, torch.nn.Module) else m  # to accept either model or state_dict
    sources = [adapter_filter_dict(get_dict(m)) for m in sources]
    # target = adapter_filter_dict(get_dict(target))
    all_keys = set(key for s in sources for key in s.keys())
    assert all([all_keys == set(s.keys()) for s in sources]), 'all sources must have the same keys'
    # assert set(target.keys()).issuperset(all_keys), 'target must have all keys'
    target = {}

    # get num of layers
    # for llama cant get it from gate shape, so get it from query shape
    adapter_lens = [int(v.shape[0]/NUM_ADAPTER_LAYERS) for source in sources for k,v in source.items() if k.endswith('adapter_query.weight')]
    # adapter_lens = [v.shape[3] for source in sources for k,v in source.items() if k.endswith('layers.0.attention.adapter_gate')]
    total_len = sum(adapter_lens)
    assert len(adapter_lens) == len(sources), 'all sources must have an adapter_len'
    # get embed dim
    adapter_dim = [v for k,v in sources[0].items() if k.endswith('adapter_query.weight')][0].shape[1]
    assert all([v.shape[1] == adapter_dim for source in sources for k,v in source.items() if k.endswith('adapter_query.weight')]), 'all adapter_querys must have the same embed dim'

    # print('lens:', adapter_lens)
    # print('total:', total_len)
    # print('dim:', adapter_dim)

    for k in all_keys:
        if k.endswith('.adapter_gate'):
            gate_list = [s[k] for s in sources]
            target[k] = torch.concat(gate_list, dim=3)
            # average them
            # target[k] = sum(gate_list) / len(gate_list)
            # take the first
            # target[k] = gate_list[0]
        elif k.endswith('adapter_query.weight'):
            sources_reshaped = [s[k].reshape(-1, 1, adap_len, adapter_dim) for s, adap_len in zip(sources, adapter_lens)]
            num_layers = sources_reshaped[0].shape[0]
            assert all([n.shape[0] == num_layers for n in sources_reshaped]), 'all sources must have the same num of layers'
            target[k] = torch.cat(sources_reshaped, dim=2).reshape(num_layers*total_len, adapter_dim)
        else:
            raise ValueError(f'unknown key: {k}')
    return target, total_len

def fuse_ia3(sources):
    assert all('base_model.model.model.layers' in k for source in sources for k in source.keys()), 'all sources must have keys base_model.model.model.layers'
    all_keys = set(key for s in sources for key in s.keys())
    assert all([all_keys == set(s.keys()) for s in sources]), 'all sources must have the same keys'
    # sum embeddings
    res = {}
    for k in all_keys:
        res[k] = sum([s[k] for s in sources])
    return res

# model that stores multiple models and can fuse them
class MultiModel():
    def __init__(self, base, pefts):
        self.base = base
        self.pefts = pefts
    
    def set_active(self, idx):
        self.active = idx
        self.base.load_state_dict(self.pefts[idx], strict=False)
    
    def to(self, device):
        self.base.to(device)
    
    def parameters(self):
        return self.base.parameters()

    def eval(self):
        self.base.eval()

    def forward(self, *args, **kwargs):
        # return self.base(*args, **kwargs)
        res = []
        for i in range(len(self.pefts)):
            self.set_active(i)
            out = self.base(*args, **kwargs)
            if hasattr(out, 'logits'):
                res.append(out.logits)
            else:
                res.append(out)
        return self.combine(res)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def combine(self, res):
        # max logit
        # return torch.max(torch.stack(res), dim=0)[0]
        # print('mixing')
        # return torch.max(*res)
        return torch.max(torch.stack([*res]), dim=0).values

