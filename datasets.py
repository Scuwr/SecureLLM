import random
import logging
import json
import itertools
import time
import numpy as np
from pathlib import Path

import torch

import env
import utils

# setup logger
logger = logging.getLogger(__name__)

class Datasets:    
    open_db_managers = {}
    @staticmethod
    def get_schema(schema_name, DATA_LIMIT=None):
        return Datasets._get_schema(schema_name, SQLSample, DATA_LIMIT=DATA_LIMIT)

    @staticmethod
    def get_schema_pseudo(schema_name, DATA_LIMIT=None):
        return Datasets._get_schema(schema_name, SQLPseudoSample, DATA_LIMIT=DATA_LIMIT)

    @staticmethod
    def _get_schema(schema_name, sample_class, DATA_LIMIT=None):
        import schema.schema_llm_data as schema_llm_data
        if '~' in schema_name:
            schema_name, settings = schema_name.split('~')
            settings = {i.split('=')[0]: i.split('=')[1] for i in settings.split('_')}
        else:
            settings = {}
        mapping = settings.get('mapping', None)
        context = settings.get('context', None)

        result = []
        fn = env.dataset_paths.schema_samples / f'samples_{schema_name}.json'
        if not fn.exists():
            raise ValueError('Schema file not found ' + str(fn))
        with open(fn) as f:
            _schema_json_list = json.load(f)
        if DATA_LIMIT is not None:
            _schema_json_list = _schema_json_list[:DATA_LIMIT]
        schema_id = schema_name.split('_')[0]
        db_fn = str(env.dataset_paths.schema_databases) + '/schema_' + str(schema_id) + '_{0}.db'
        if db_fn not in Datasets.open_db_managers:
            my_db_manager = schema_llm_data.MyDBManager.get_manager(db_fn, mapping=mapping, ERROR_ON_NO_DB=True)
            Datasets.open_db_managers[db_fn] = my_db_manager
        else:
            my_db_manager = Datasets.open_db_managers[db_fn]

        if context:
            print("Using SQL Metadata Context")
            metadata_filepath = str(env.dataset_paths.metadata) + '/schema_' + str(schema_id) + '.txt'
            with open(metadata_filepath) as f:
                context = f.read()
        else:
            context = ""

        for i in _schema_json_list:
            if sample_class == SQLPseudoSample:
                result.append(SQLPseudoSample(
                                    context=context,
                                    question=i['question'], 
                                    pseudocode=my_db_manager.apply_mapping(i['pseudocode']),
                                    sql_query=my_db_manager.apply_mapping(i['sql']), 
                                    sql_oracle=my_db_manager.is_query_equal,
                                    hardcode_ground_tables=True,
                                    metadata={
                                        'dataset': 'schemapseudo_' + schema_name, 
                                        'seed': i['seed']
                                    }))
            elif sample_class == SQLSample:
                result.append(SQLSample(question=i['question'], 
                                        sql_query=my_db_manager.apply_mapping(i['sql']), 
                                        sql_oracle=my_db_manager.is_query_equal,
                                        metadata={
                                            'dataset': 'schema_' + schema_name, 
                                            'seed': i['seed']
                                        }))
            else:
                raise ValueError('Unknown sample_class ' + str(sample_class))
        return result

    @staticmethod
    def _load_jsonl(f):
        with open(f, 'r') as fj:
            fj = '[' + ','.join(fj.readlines()) + ']'
            f = json.loads(fj)
        return f

    @staticmethod
    def _batchify(text, tokenizer, max_token_len):
        l = tokenizer.encode(text)
        result = []
        for i in range(0, len(l), max_token_len):
            if i+max_token_len > len(l):  # last batch
                i = max(0, len(l) - max_token_len)  # replace i with the start of the last batch
            result.append(l[i:i+max_token_len])
        result = [tokenizer.decode(x) for x in result]
        return result

# def get_dataset(string_representations, tokenizer, max_inp_matrix_size, batchify_len=None, shuffle=True, DEBUG_DATA_LIMIT=None):
def get_dataset(dataset_args, tokenizer, is_train, verbose=True) -> 'DynamicBatchLoader':
    string_representations: str = dataset_args.train_str if is_train else dataset_args.val_str
    max_inp_matrix_size = int(dataset_args.max_inp_matrix_size)
    shuffle = True if is_train else False

    if string_representations is None:
        return None
    # remove quotes if present
    if string_representations.startswith('"') and string_representations.endswith('"'):
        string_representations = string_representations[1:-1]

    if hasattr(dataset_args, 'batchify_len') and dataset_args.batchify_len is not None:
        batchify_len = int(dataset_args.batchify_len)
    else:
        batchify_len = max_inp_matrix_size
    
    if hasattr(dataset_args, 'max_batch_size') and dataset_args.max_batch_size is not None:
        max_batch_size = int(dataset_args.max_batch_size)
    else:
        max_batch_size = 32

    if hasattr(dataset_args, 'suppressed_val_tokens') and dataset_args.suppressed_val_tokens is not None:
        tokens_used = tokenizer.encode(dataset_args.suppressed_val_tokens)
        token_mask = np.zeros(32000, dtype=int)
        token_mask[tokens_used] = 1

        suppressed_val_tokens = token_mask
    else:
        suppressed_val_tokens = None

    lst = []
    for key in string_representations.split(','):
        key = key.strip()
        DATA_LIMIT = None
        if ':' in key:
            key, DATA_LIMIT = key.split(':')
            DATA_LIMIT = int(DATA_LIMIT)
        # assert string_representations.count(key) == 1, f'Duplicate dataset found {key} in {string_representations}'  # TODO this is good to have but crashes for example val_1 and val_1_v2
        if key.startswith('schema_'):
            schema_id = key.split('_', 1)[1]
            lst.extend(Datasets.get_schema(schema_id, DATA_LIMIT=DATA_LIMIT))
        elif key.startswith('schemapseudo_'):
            schema_id = key.split('_', 1)[1]
            lst.extend(Datasets.get_schema_pseudo(schema_id, DATA_LIMIT=DATA_LIMIT))
        else:
            raise ValueError('Unknown dataset ' + key)

    # print to console only if it takes more than 5 seconds
    start_time = time.time()
    for i, s in enumerate(lst):
        s.set_tokenized_len(tokenizer)
        if start_time is not None and time.time() - start_time > 1:  # calculate estimated time after 1 second
            estimated = len(lst) / (i + 1) * (time.time() - start_time)
            if verbose and estimated > 5:
                print(f'Estimated time to tokenize dataset: {estimated:.1f} seconds. Let dataset: {len(lst)}')
            start_time = None

    filtered_lst = []
    _warnings = {}
    for i, x in enumerate(lst):
        if x.tokenized_len > max_inp_matrix_size:
            _warnings.setdefault(x._metadata['dataset'], 0)
            _warnings[x._metadata['dataset']] += 1
        else:
            filtered_lst.append(x)
    
    if verbose and len(_warnings) > 0:
        logger.warning(f'Found {len(filtered_lst)} samples with tokenized length <= {max_inp_matrix_size}')
        for k,v in _warnings.items():
            logger.warning(f'{k}: {v} samples with tokenized length > {max_inp_matrix_size}')

    dataset = DynamicBatchLoader(filtered_lst, 
                            max_inp_matrix_size=max_inp_matrix_size, 
                            max_batch_size=max_batch_size, 
                            shuffle=shuffle, 
                            size_getter=lambda x: x.tokenized_len,
                            suppressed_val_tokens=suppressed_val_tokens
                        )
    # check for assertion errors if max_inp_matrix_size is too small
    for i in dataset:
        pass
    return dataset


class SingleSample:
    def __init__(self, *args, **kwargs):
        self.text = kwargs.get('text', None)
        self._metadata = kwargs.get('metadata', None)
    def get_text(self):
        return self.text
    def get_labels(self, input_ids, attention_mask, tokenizer=None):
        # attention_mask is 1 for all tokens, 0 for padding
        # -100 in target is ignored by CrossEntropyLoss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return labels
    def get_generation(self, tokenizer=None):
        raise NotImplementedError
    def is_generation_done(self, text, tokens=None):
        raise NotImplementedError
    # @staticmethod
    # def batch_get_inputids_and_attnmask(batch, tokenizer, device='cpu', for_generation=False):
    #     if for_generation:
    #         text = [n.get_generation() for n in batch]
    #     else:
    #         text = [n.get_text() for n in batch]
    #     t = tokenizer(text, padding=True, return_tensors='pt', truncation=True, max_length=1024)
    #     t['input_ids'] = t['input_ids'].to(device)
    #     t['attention_mask'] = t['attention_mask'].to(device)
    #     return t
    @staticmethod
    def batch_get_labels(batch, input_ids, attention_mask, tokenizer=None):
        result = [n.get_labels(input_ids[i], attention_mask[i], tokenizer=tokenizer) for i , n in enumerate(batch)]
        return torch.stack(result)
    def set_tokenized_len(self, tokenizer):
        text = self.get_text()
        self.tokenized_len = len(tokenizer.encode(text)) if isinstance(text, str) else len(text)

class CocoSample(SingleSample):
    def __init__(self, text):
        text = 'Caption: ' + text
        super().__init__(text=text)
    def get_generation(self, tokenizer=None):
        return 'Caption:'

class SQLSample(SingleSample):
    START_SQL = ' <sql> '
    END_SQL = ' </sql>'
    def __init__(self, question, sql_query, sql_oracle, *args, **kwargs):
        self.question = question
        self.sql_query = sql_query
        self.sql_oracle = sql_oracle
        super().__init__(text='', *args, **kwargs)
    def get_generation(self, tokenizer=None):
        return self._setup_text(add_answer=False)
    def is_generation_done(self, text, tokens=None):
        return self.END_SQL in text
    def get_text(self):
        return self._setup_text(add_answer=True)
    def _setup_text(self, add_answer):
        text = '<s>Question: ' + self.question + '\n'
        if add_answer:
            text += self.START_SQL + self.sql_query + self.END_SQL
        else:
            text += self.START_SQL
        return text
    def get_labels(self, input_ids, attention_mask, tokenizer):
        assert tokenizer is not None, 'Tokenizer must be provided to get signature for labels'
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        seq_to_find = _get_token_signature(self.START_SQL, tokenizer)

        labels_str = ','.join([str(i) for i in labels.tolist()])
        str_to_find = ','.join([str(i) for i in seq_to_find])
        assert labels_str.count(str_to_find) == 1, f'Expected 1 occurence of {str_to_find} got {labels_str.count(str_to_find)}, text: {self.get_text()}'
        str_pos = labels_str.find(str_to_find)
        start_pos = labels_str[:str_pos].count(',')
        end_pos = start_pos + len(seq_to_find)
        labels[:end_pos] = -100
        return labels

    def get_accuracy(self, gen_text, memory=None):
        if memory is None or memory == {}:
            memory = {
                'v': [0, 0, 0],  # correct, total, invalid sql
                'to_print': '',
            }
        v = memory['v']
        ans_pos = gen_text.find(self.START_SQL)
        assert ans_pos != -1, f'Expected "{self.START_SQL}" in {gen_text}'
        end_pos = gen_text.find(self.END_SQL)
        v[1] += 1  # total
        if end_pos == -1:
            v[2] += 1  # invalid answer
        else:
            gen_sql = gen_text[ans_pos+len(self.START_SQL):end_pos].strip()
            oracle_ret = self.sql_oracle(self.sql_query, gen_sql)  # 1 if correct, -1 if invalid sql, 0 if incorrect
            if oracle_ret == 1:
                v[0] += 1
            elif oracle_ret == -1:
                v[2] += 1
        memory['to_print'] = f'{100 * v[0] / v[1] if v[1] > 0 else -1:.1f}% ({v[2]})'
        return memory

class SQLPseudoSample(SingleSample):
    START_SQL = ' <pseudo> '
    END_SQL = ' </pseudo>'
    def __init__(self, context, question, pseudocode, sql_query, sql_oracle, hardcode_ground_tables=False, *args, **kwargs):
        self.context = context
        self.question = question
        self.pseudocode = pseudocode
        self.sql_query = sql_query
        self.sql_oracle = sql_oracle
        self.hardcode_ground_tables = hardcode_ground_tables
        super().__init__(text='', *args, **kwargs)
    def get_generation(self, tokenizer=None):
        return self._setup_text(add_answer=False)
    def is_generation_done(self, text, tokens=None):
        return self.END_SQL in text
    def get_text(self):
        return self._setup_text(add_answer=True)
    def _setup_text(self, add_answer):
        text = '<s>' + self.context + '\nQuestion: ' + self.question + '\n'
        if add_answer:
            text += self.START_SQL + self.pseudocode + self.END_SQL
        else:
            text += self.START_SQL
            # testing only do not push
            # add first two lines
            # text += '\n'.join(self.pseudocode.split('\n')[:2]) + '\n'
            # do not add tables
            # text += '\n'.join(self.pseudocode.split('\n')[:1]) + '\nconditions: \n' 
        return text
    def get_labels(self, input_ids, attention_mask, tokenizer):
        assert tokenizer is not None, 'Tokenizer must be provided to get signature for labels'
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        seq_to_find = _get_token_signature(self.START_SQL, tokenizer)

        labels_str = ','.join([str(i) for i in labels.tolist()])
        str_to_find = ','.join([str(i) for i in seq_to_find])
        assert labels_str.count(str_to_find) == 1, f'Expected 1 occurence of {str_to_find} got {labels_str.count(str_to_find)}, text: {self.get_text()}'
        str_pos = labels_str.find(str_to_find)
        start_pos = labels_str[:str_pos].count(',')
        end_pos = start_pos + len(seq_to_find)
        labels[:end_pos] = -100
        return labels

    def get_accuracy(self, gen_text, memory=None):
        if memory is None or memory == {}:
            memory = {
                'v': [0, 0, 0],  # correct, total, invalid sql
                'to_print': '',
            }
        v = memory['v']
        ans_pos = gen_text.find(self.START_SQL)
        assert ans_pos != -1, f'Expected "{self.START_SQL}" in {gen_text}'
        end_pos = gen_text.find(self.END_SQL)
        v[1] += 1  # total
        if end_pos == -1:
            v[2] += 1  # invalid answer
        else:
            gen_pseudo = gen_text[ans_pos+len(self.START_SQL):end_pos].strip()
            try:
                gen_sql = self.pseudocode_to_sql(gen_pseudo)
            except:
                gen_sql = None
            oracle_ret = self.sql_oracle(self.sql_query, gen_sql)  # 1 if correct, -1 if invalid sql, 0 if incorrect
            if oracle_ret == 1:
                v[0] += 1
            elif oracle_ret == -1:
                v[2] += 1
        memory['to_print'] = f'{100 * v[0] / v[1] if v[1] > 0 else -1:.1f}% ({v[2]})'
        return memory

    def pseudocode_to_sql(self, pseudocode):
        lines = pseudocode.split('\n')
        want = lines[0].split(':')[1].strip()
        # check tables
        if self.hardcode_ground_tables:
            # add tables automatically, GROUND TRUTH AUTOMATICALLY ADDED
            ground_tables = self.pseudocode.split('\n')[1]
            if 'tables' not in lines[1]:
                lines.insert(1, ground_tables)
            else:
                lines[1] = ground_tables
        elif 'tables' not in lines[1]:
            print('WARNING: tables not found in pseudocode, and hardcode_ground_tables is False')
            lines.insert(1, 'tables: ')
        tables = lines[1].split(':')[1].split(', ')
        tables = [n.strip() for n in tables]
        conditions = lines[3:]
        conditions = [n.strip() for n in conditions if len(n.strip()) > 0]
        sql = f'SELECT {want}\nFROM ' + tables[0]
        if len(tables) > 1:
            sql += '\nNATURAL JOIN ' + '\nNATURAL JOIN '.join(tables[1:])
        if len(conditions) > 0:
            sql += '\nWHERE ' + '\nAND '.join(f'({x})' for x in conditions)
        return sql

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        for d in data:
            assert isinstance(d, SingleSample), f'Expected SingleSample, got {type(d)}'
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index) -> SingleSample:
        return self.data[index]

class DynamicBatchLoader(object):
    '''Custom loader that supports variable batch size depending on the number of tokens in the samples
        This prevents CUDA Out of Memory errors when a batch gets samples that are too long, vice-versa this provides large batch size when samples are short
    '''
    def __init__(self, dataset: list[SingleSample], max_inp_matrix_size, size_getter, shuffle=False, max_batch_size=None, suppressed_val_tokens=None):
        self.dataset = dataset
        self.max_inp_matrix_size = max_inp_matrix_size
        self.size_getter = size_getter
        self.shuffle = shuffle
        self.max_batch_size = max_batch_size if max_batch_size else torch.inf
        self.suppressed_tokens = suppressed_val_tokens
        assert self.max_batch_size > 0

    def __can_accept_new(self, cur_batch_len, cur_max_size, next_size):
        # check if the next sample can be added to the current batch: 2 conditions
        # batch is not full
        # AND
        # new_max_size * new_batch_len <= max_inp_matrix_size
        return (cur_batch_len + 1 <= self.max_batch_size) and \
            max(cur_max_size, next_size)*(cur_batch_len+1) <= self.max_inp_matrix_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        ds_ind = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(ds_ind)

        batch = []
        cur_max_size = 0
        i = 0
        for i in ds_ind:
            size_i = self.size_getter(self.dataset[i])
            assert self.__can_accept_new(0, 0, size_i), f'Input size {size_i} for sample {i} is too big for max_inp_matrix_size {self.max_inp_matrix_size}'
            if self.__can_accept_new(len(batch), cur_max_size, size_i):
                batch.append(self.dataset[i])
                cur_max_size = max(cur_max_size, size_i)
                continue
            else:
                yield batch
                batch = [self.dataset[i]]
                cur_max_size = size_i
        if len(batch) > 0:
            yield batch

    def for_generation(self, tokenizer):
        to_do = [{'i': i, 'sample': x,
                            'gen_len': 0, 
                            'gen_text': x.get_generation(tokenizer=tokenizer),
                            'done': False,
                            'limit_hit': None,
                            'tokens': None,
                            'suppressed_tokens': self.suppressed_tokens
                         } for i, x in enumerate(self.dataset)]
        for d in to_do:
            if isinstance(d['gen_text'], str):
                d['tokens'] = tokenizer.encode(d['gen_text'])  
            elif isinstance(d['gen_text'], torch.Tensor):
                d['tokens'] = d['gen_text'].tolist()
            else:
                raise ValueError('Unknown type for gen_text')
        if self.shuffle:
            random.shuffle(to_do)
        while len(to_do) > 0:
            # to_do.sort(key=lambda x: x['gen_len'], reverse=True)  # reverse sort to prioritize longer samples
            batch = []
            cur_max_size = 0
            for i, d in enumerate(to_do):
                size_i = len(d['tokens'])
                if self.__can_accept_new(len(batch), cur_max_size, size_i):
                    batch.append(d)
                    cur_max_size = max(cur_max_size, size_i)
            if len(batch) == 0:
                print('WARNING:\n'*10, '  Batch size is 0, dataset contains samples that are too long. Exiting.')
                break
            yield batch
            for d in batch:  # remove done samples
                if d['done']:
                    to_do.remove(d)


class DynamicRawTextBatchLoader(DynamicBatchLoader):
    def __init__(self, tokenizer, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def __iter__(self):
        raise NotImplementedError


def _longest_common_sub(lst1: list, lst2: list) -> list:
    lst1_length = len(lst1)
    lst2_length = len(lst2)

    dp = [[0] * (lst2_length + 1) for _ in range(lst1_length + 1)]
    ans_index = 0
    ans_length = 0

    for i in range(1, lst1_length + 1):
        for j in range(1, lst2_length + 1):
            if lst1[i - 1] == lst2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
                if dp[i][j] > ans_length:
                    ans_index = i
                    ans_length = dp[i][j]

    res = lst1[ans_index - ans_length : ans_index]
    return res

_get_token_signature_cache = {}
def _get_token_signature(string, tokenizer, verbose=False):
    """
    A function that finds the longest common subsequence in the encoded tokens using a tokenizer.
    This is useful if we want to find a specific token in the encoded tokens, for example a token like </sql> or </answer> that marks the end of a sequence.
    """
    # memoized
    if string in _get_token_signature_cache.get(tokenizer, {}):
        return _get_token_signature_cache[tokenizer][string]
    pres = ['', ' ', 'A', ' A ', '\n', '\t']  # variety of prefixes to make sure signature is consistent
    tokens = [tokenizer.encode(i1+string+i2) for i1, i2 in itertools.product(pres, pres)]
    # add encoded token without begining and ending tokens
    tokens.append(tokenizer.encode('dummy ' + string + ' dummy')[1:-1]) 

    # find longest common subsequence in all tokens
    lcs = tokens[0]
    for i in tokens[1:]:
        lcs = _longest_common_sub(lcs, i)
    if verbose:
        print(f'Finding token signature for "{string}"')
        print('Raw encoded tokens:', tokenizer.encode(string))
        print(f'Final result =', lcs)
        print(f'Decoded:"{tokenizer.decode(lcs)}"')
        print('Allparts:')
        print(*[f'{t}: "{i1+string+i2}"'.replace('\n', '\\n') for t, (i1, i2) in zip(tokens, itertools.product(pres, pres))], sep='\n')
        print(f'Finding token signature for "{string}", Raw encoded tokens: {tokenizer.encode(string)} Final result: {lcs} Decoded:"{tokenizer.decode(lcs)}"')
    _get_token_signature_cache.setdefault(tokenizer, {})[string] = lcs
    return lcs


def __main_get_signature():
    import argparse
    import env
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(env.model_paths.llama_hf_7b, legacy=True)
    tokenizer.pad_id = 0
    tokenizer._original_encode = tokenizer.encode
    tokenizer.encode = lambda x: tokenizer._original_encode(x, add_special_tokens=False)  # skip <s> added to the beginning decoded text

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()
    _get_token_signature(args.text, tokenizer, verbose=True)
    # _get_token_signature('\nOutput:', tokenizer, verbose=True)

if __name__ == '__main__':
    __main_get_signature()
