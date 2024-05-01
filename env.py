from pathlib import Path
import os

home = Path(os.getcwd()).parent
local = Path('SecureLLM')
models = Path('models')


class dataset_paths:
    schema_databases = home / local / 'training_data/databases'
    schema_samples = home / local / 'training_data/samples'
    metadata = home / local / 'schema'

class model_paths:
    save_model_path = home / local / 'trained_models/trained_model_{0}.pt'
    vanilla_llama_dir = home / models / 'llama2/llama'
    vanilla_tokenizer_path = home / models / 'llama2/llama/tokenizer.model'
    llama_hf_7b = home / models / 'llama2/llama_hf_converted/7b'
    llama_hf_70b = home / models / 'llama2/llama_hf_converted/70b'

class qgen_paths:
    schema_qgen = home / local / 'schema'

db_path = home / local / 'trained_models/a_model_list.db'
db_val_path = home / local / 'trained_models/b_val_list.db'
val_save_path = home / local / 'trained_models/validation/val_{0}.json'

def check_private_env(private_env):
    """Check that all paths exist in private_env (locals())"""
    assert 'dataset_paths' in private_env, f'dataset_paths does not exist in your private env. Please create it.'
    for i in dir(dataset_paths): 
        if not i.startswith('_'):
            assert hasattr(private_env['dataset_paths'], i), f'dataset_paths.{i} does not exist in your private env. Please create it.'
    assert 'model_paths' in private_env, f'model_paths does not exist in your private env. Please create it.'
    for i in dir(model_paths):
        if not i.startswith('_'):
            assert hasattr(private_env['model_paths'], i), f'model_paths.{i} does not exist in your private env. Please create it.'
    
    assert 'db_path' in private_env, f'db_path does not exist in your private env. Please create it.'
    assert 'db_val_path' in private_env, f'db_val_path does not exist in your private env. Please create it.'
    assert 'val_save_path' in private_env, f'val_save_path does not exist in your private env. Please create it.'

check_private_env(locals())  # check that all paths exist

