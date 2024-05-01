import env
import argparse
from train import set_seed
from transformers import LlamaTokenizer, LlamaConfig
from modeling import modeling_llama_hf

from modeling.secureLLM.slora import SloraSumConfig, SloraMaxDiffElemConfig, SloraLogitConfig, SloraLoraHubConfig
from modeling.secureLLM.slora_model import SloraBaseModel

import torch
from train import ParseKwargs

import random
from collections import OrderedDict

import datasets, train
from transformers import LlamaTokenizer

from tqdm.notebook import tqdm

import sys
import utils



def main(**kwargs):
    device = kwargs['device']
    sample_size = kwargs['sample_size']
    models = kwargs['models']
    negated_models = kwargs['negated']
    SloraConfig = kwargs['config']
    weights = kwargs['weights']
    mapping = kwargs['mapping']
    gpt = kwargs['gpt']
    pseudo = kwargs['pseudo']
    val_list = kwargs['val_str']
    model_root = kwargs.get('mroot', "./trained_models/v2/")

    world_size = 1
    model_name = 'llama-2-7b'
    seed = 1

    # model_root = "./trained_models/v2/"
    # model_root = "./trained_models/v2_animals_sql/"
    # model_root = "./trained_models/v2_animals_ps/"

    set_seed(seed)


    mapping_text = ""
    if mapping > 0:
        mapping_text = f"~mapping={mapping}"
    
    gpt_text = ""
    if gpt:
        gpt_text = "_gpt"

    pseudo_text = ""
    if pseudo:
        pseudo_text = "pseudo"

    class dataset_args:
        val_str = ','.join(f'schema{pseudo_text}_{i}_val{gpt_text}{mapping_text}:{sample_size}' for i in val_list)
        max_inp_matrix_size = 800
        batchify_len = 400
        max_batch_size = 128

    print()
    print("=== Inputs ===")
    print(f'GPU: {device}')
    print(f'Sample Size: {sample_size}')
    print(f'Model root: {model_root}')
    print(f'Models: {models}')
    print(f'Negated Models: {negated_models}')
    print(f'Config Type: {SloraConfig}')
    print(f'Weights: {weights}')
    print(f'val_str: {dataset_args.val_str}')
    print()
    print('=== Starting Script ===')

    tokenizer = LlamaTokenizer.from_pretrained(env.model_paths.llama_hf_7b)
    tokenizer.pad_id = 0 # TODO check if correct

    config = LlamaConfig.from_pretrained(env.model_paths.llama_hf_7b)
    model = modeling_llama_hf.LlamaForCausalLM.from_pretrained(env.model_paths.llama_hf_7b, config=config)
    model.half()

    print("Hugging Face Llama Model successfully loaded")

    ###

    slora_size = len(models)

    slora_config = SloraConfig(
            #task_type=TaskType.SEQ_2_SEQ_LM, 
            inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, n=slora_size, negated_adapters=negated_models
        )

    slora = SloraBaseModel(model, slora_config, adapter_name = "slora", adapter_weights = weights)

    print("Slora Configuration loaded and initialized")

    ###

    for idx, model in enumerate(models):
        checkpoint = torch.load(model_root + f"{model}.pt", map_location="cpu")
        slora.load_adapter(idx, checkpoint['parameters'])

    print("Adapter checkpoints loaded and linked")

    ###

    tokenizer = LlamaTokenizer.from_pretrained(env.model_paths.llama_hf_7b)
    tokenizer.pad_id = 0
    tokenizer._original_encode = tokenizer.encode
    tokenizer.encode = lambda x: tokenizer._original_encode(x, add_special_tokens=False)  # skip <s> added to the beginning decoded text

    dataset_mc_val = datasets.get_dataset(dataset_args, tokenizer, is_train=False)

    slora.to(f'cuda:{device}')

    print("Starting validation test")
    memories = train.validate(slora, tokenizer, dataset_mc_val, verbose=True, gem_max_token_len=250
                            #   , use_kv=True
                              )

    # silo_names = ["S1", "S2", "S3"]
    # schema_memories = zip(silo_names, memories.values())
    schema_memories = dict(memories.items())
    del schema_memories['all_samples']  # remove all_samples from memories
    del schema_memories['unk']
    del schema_memories['no_acc']

    print()
    print("=== Inputs ===")
    print(f'GPU: {device}')
    print(f'Sample Size: {sample_size}')
    print(f'val_str: {dataset_args.val_str}')
    print(f'Model root: {model_root}')
    print(f'Models: {models}')
    print(f'Negated Models: {negated_models}')
    print(f'Config Type: {SloraConfig}')
    print(f'Weights: {weights}')
    print()

    print("=== Results ===")
    perf = []
    for silo_name, val_data in schema_memories.items():
        metric = val_data['v']
        if len(metric) >= 2 and metric[1] != 0:
            perf.append((silo_name, (100 * metric[0] / metric[1])))
            # print(f'{silo_name}: {(100 * metric[0] / metric[1]):.2f}%')
    #print(f'latex:', ' & '.join([f'{100 * val_data["v"][0] / val_data["v"][1]:.2f}\%' for silo_name, val_data in schema_memories.items()]))
    
    tree = get_tree_results(memories)

    for idx in range(len(perf)):
        label, acc = perf[idx]
        dist = tree[0][idx]
        dist_norm = tree[1][idx]
        if label == dist[0]:
            print(f'{label},{acc},{dist[1]},{dist_norm[1]}')
        else:
            print("ERROR: Label mismatch when parsing cvs data")

def get_tree_results(val_results, debug=False):
    dists = {gsample['sample']._metadata['dataset']: [] for gsample in val_results['all_samples']}
    dists2 = {gsample['sample']._metadata['dataset']: [] for gsample in val_results['all_samples']}
    for gsample in val_results['all_samples']:
        dname = gsample['sample']._metadata['dataset']
        # GET CONDITIONS
        if '<pseudo>' in gsample['gen_text']:
            gen_text = gsample['gen_text']
            gen_text = gen_text[gen_text.find('conditions:'):gen_text.find('</pseudo>')].strip().split('\n')[1:]
            gen_text = ' AND '.join('(' + x + ')' for x in gen_text)  # wrap in brackets and join with AND 
            ground_query = gsample['sample'].pseudocode
            ground_query = ground_query[ground_query.find('conditions:'):].strip().split('\n')[1:]
            ground_query = ' AND '.join('(' + x + ')' for x in ground_query)  # wrap in brackets and join with AND

        elif '<sql>' in gsample['gen_text']:
            gen_text = gsample['gen_text']
            gen_text = gen_text[gen_text.find('WHERE') + 5:gen_text.find('</sql>')].strip().split('\n')
            gen_text = ' AND '.join('(' + x + ')' for x in gen_text)  # wrap in brackets and join with AND
            ground_query = gsample['sample'].sql_query
            ground_query = ground_query[ground_query.find('WHERE') + 5:].strip().split('\n')
            ground_query = ' AND '.join('(' + x + ')' for x in ground_query)
        else:
            print('error cannot find pseudo or sql in gen_text')
            dists[dname].append(1000)
            dists2[dname].append(1000)
            continue
            
        dist, dist_norm = utils.conds_distance(ground_query, gen_text)
        dists[dname].append(dist)
        dists2[dname].append(dist_norm)

    results = []
    norm_results = []
    dists_keys = list(dists.keys())
    dists2_keys = list(dists2.keys())
    dists_keys.sort()
    dists2_keys.sort()

    for dname in dists_keys:
        result = (dname, sum(dists[dname]) / len(dists[dname]))
        if debug:
            print(dname, result)
        results.append(result)
    for dname in dists2_keys:
        result = (f'normalized {dname}', sum(dists2[dname]) / len(dists2[dname]))
        if debug:
            print('normalized', dname, result)
        norm_results.append(result)

    return (results, norm_results)

def slora_config_type(config_name):
    config_classes = {
        'SloraSum': SloraSumConfig,
        'SloraMax': SloraMaxDiffElemConfig,
        'SloraLogit': SloraLogitConfig,
        'SloraLoraHub': SloraLoraHubConfig,
    }
    return config_classes.get(config_name, None)

def convert_to_integers(input):
    try:
        return int(input)
    except ValueError:
        raise argparse.ArgumentTypeError("Each item in the list should be an integer")

def convert_to_float(input):
    try:
        return float(input)
    except ValueError:
        raise argparse.ArgumentTypeError("Each item in the list should be an float or integer")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=2, help='Device number')
    parser.add_argument('--sample_size', type=int, default=10, help='Sample size')
    parser.add_argument('--models', nargs='*', default=["M1", "M2", "M3"], help='List of models')
    parser.add_argument('--mroot', type=str, default="./trained_models/v2/", help='Model root')
    parser.add_argument('--negated', nargs='*', type=convert_to_integers, default=[], help='List of models')
    parser.add_argument('--config', type=slora_config_type, default=SloraLogitConfig, help='\nSecure LLM lora configuration class\nChoose from SloraSum, SloraMax, SloraLogit, SloraLoraHub')
    parser.add_argument('--weights', type=convert_to_float, nargs='*', default=[], help='Weights')
    parser.add_argument('--mapping', type=int, default=0, help='Select type of mapping (0 is no mapping)')
    parser.add_argument('--gpt', action='store_true', help='Use GPT questions')
    parser.add_argument('--pseudo', action='store_true', help='Use 6NF Pseudocode questions')
    parser.add_argument('--val_str', nargs='*', default=['1', '2', '3', 'union12', 'union13', 'union23', 'union123'], help='List of validation sets')
    args = parser.parse_args()

    if len(args.weights) < 1:
        args.weights = None
    else:
        args.weights = torch.tensor(args.weights, device=args.device)

    kwargs = vars(args)
    main(**kwargs)