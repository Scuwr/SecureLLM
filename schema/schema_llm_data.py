import os
import sqlite3
import json
import re

import sys
sys.path.append('..')

import env
from tqdm import tqdm

import schema

def generate_dataset(schema_filename, num_samples, save_to=None):
    all_rules, schema_specs = schema.parse_schema_get(schema_filename)
    results = []
    visited_seeds = set()
    pbar = tqdm(total=num_samples, dynamic_ncols=True)
    __total_iterations = 0
    while len(results) < num_samples:
        __total_iterations += 1
        if __total_iterations > 100*num_samples:
            raise Exception('Could not generate enough samples, tried {} times'.format(__total_iterations))

        resolved, gen_seed = schema.generator(all_rules, 'root', seeder=[])
        ques, args = resolved.walk_and_resolve()
        sql_query = schema.args_to_sql(args, schema_specs)
        if tuple(gen_seed) in visited_seeds:
            continue
        visited_seeds.add(tuple(gen_seed))
        results.append({'seed': gen_seed, 'question': ques, 'sql': sql_query})
        pbar.update(1)
        pbar.refresh()
    pbar.close()
    if save_to is not None:
        result_str = format_list_to_pretty_json(results)
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, 'w') as f:
            f.write(result_str)
    return results


def remove_zero_records(samples_path, dbpath, num_queries, debug_mode=False):
    print('input file', samples_path)
    with open(samples_path) as f:
        samples = json.load(f)
    new_samples = []

    db_manager = MyDBManager.get_manager(dbpath, ERROR_ON_NO_DB=True)
    zero_count = 0
    pbar = tqdm(samples)
    for i, s in enumerate(pbar):
        non_zero_count = i - zero_count
        if non_zero_count > num_queries:
            print(f'STOPPING after finding {num_queries} non_zero_count')
            pbar.set_description(f'zero_count: {zero_count}, non_zero_count: {non_zero_count}')
            pbar.close()
            break
        result = db_manager.exec_query(s['sql'])
        # print(result)
        is_zero_record = True
        for r in result:
            if len(r) > 1:
                is_zero_record = False
                break
            if len(r) == 1 and any(x is not None for x in r[0]):
                is_zero_record = False
                break
        else:
            is_zero_record = True
            zero_count += 1
        samples[i]['is_zero_record'] = is_zero_record
        if not is_zero_record:
            new_samples.append(samples[i])
        if i % 50 == 0:
            pbar.set_description(f'zero_count: {zero_count}, non_zero_count: {non_zero_count}')
            pbar.refresh()
    pbar.set_description(f'zero_count: {zero_count}, non_zero_count: {non_zero_count}')
    print(f'zero_count: {zero_count}, non_zero_count: {non_zero_count}')
    pbar.refresh()
    result_str = format_list_to_pretty_json(new_samples)
    if debug_mode:
        with open(str(env.home) + '/silos/TEMP.json', 'w') as f:
            f.write(result_str)
    else:
        with open(samples_path, 'w') as f:
            f.write(result_str)

def remove_no_condition_records(samples_path, dbpath, debug_mode=False):
    print('input file', samples_path)
    with open(samples_path) as f:
        samples = json.load(f)
    new_samples = []

    db_manager = MyDBManager.get_manager(dbpath, ERROR_ON_NO_DB=True)
    no_condition_count = 0
    pbar = tqdm(samples)
    for i, s in enumerate(pbar):
        pseudo = s['pseudocode']
        pseudo_no_conditions = pseudo.split('conditions:')[0]
        query = _pseudoscode_to_sql(pseudo_no_conditions)
        if db_manager.is_query_equal(s['sql'], query, memoize=True):
            no_condition_count += 1
        else:
            new_samples.append(samples[i])
        if i % 50 == 0:
            pbar.set_description(f'no condition: {no_condition_count}')
            pbar.refresh()
    pbar.set_description(f'no condition: {no_condition_count}')
    print(f'no condition: {no_condition_count}')
    print('final count:', len(new_samples))
    pbar.refresh()
    result_str = format_list_to_pretty_json(new_samples)
    if debug_mode:
        with open(str(env.home) + '/silos/TEMP.json', 'w') as f:
            f.write(result_str)
    else:
        with open(samples_path, 'w') as f:
            f.write(result_str)




def _remove_prefix_before_period(text):
    """Remove table names aaa XXX.YYY bbb -> aaa YYY bbb"""
    return re.sub(r'(?<!\w)(\w*?)\.', '', text)
def _args_to_pseudocode(args, schema_specs):
    result = []
    # copied from schema.py schema.args_to_sql
    if schema.CONST.TABLES in args and isinstance(args[schema.CONST.TABLES], str):
        args[schema.CONST.TABLES] = [args[schema.CONST.TABLES]]
    elif schema.CONST.TABLES not in args:
        args[schema.CONST.TABLES] = []
    if schema.CONST.CONDITIONS in args and isinstance(args[schema.CONST.CONDITIONS], str):
        args[schema.CONST.CONDITIONS] = [args[schema.CONST.CONDITIONS]]
    elif schema.CONST.CONDITIONS not in args:
        args[schema.CONST.CONDITIONS] = []
    _all_refs = ' '.join(args[schema.CONST.CONDITIONS]) + ' ' + (args[schema.CONST.WANT] if schema.CONST.WANT in args else '')  # all references to tables (i.e. "table.column")
    args[schema.CONST.TABLES].extend(re.findall("(?<!\w)(\w*?)\.", _all_refs))  # add everything before "." which are tables from conditions
    # --- take care of SELECT
    if schema.CONST.WANT not in args:  # no SELECT specified, select PK of FROM table
        select = schema_specs['PKs'][args['TABLES'][0]]
    else:
        select = args[schema.CONST.WANT]
    if schema.CONST.TRANSFORM in args:  # apply transformation if needed
        if args[schema.CONST.TRANSFORM].lower() == 'none':
            pass
        elif args[schema.CONST.TRANSFORM] == 'COUNT':
            select = 'COUNT(DISTINCT {})'.format(select)
        elif args[schema.CONST.TRANSFORM] in ['AVG', 'SUM', 'MAX', 'MIN']:
            select = '{}({})'.format(args[schema.CONST.TRANSFORM], select)
        else:
            raise Exception('Invalid T: {}'.format(args[schema.CONST.TRANSFORM]))
    # --- SELECT is done

    _prepared = schema._prepare_args_to_sql(args, schema_specs)  # need just for tables

    want = _remove_prefix_before_period(select)
    tables = ', '.join(sorted(list(set(_prepared['ALL_TABLES']))))
    conditions = [_remove_prefix_before_period(x) for x in args['CONDITIONS']]
    conditions = '\n'.join(conditions)
    result = f'want: {want}\ntables: {tables}\nconditions: \n{conditions}'
    return result
def _pseudoscode_to_sql(pseudocode):
    lines = pseudocode.split('\n')
    want = lines[0].split(':')[1].strip()
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

def add_pseudocode_to_samples(samples_path, qgen_path, dbpath, debug_mode=False):
    with open(samples_path) as f:
        samples = json.load(f)

    all_rules, schema_specs = schema.parse_schema_get(qgen_path)
    db_manager = MyDBManager.get_manager(dbpath, ERROR_ON_NO_DB=True)
    new_count = 0
    pbar = tqdm(samples)
    for i, s in enumerate(pbar):
        # if i > 1000:
        #     print('STOPPING at 1000')
        #     break
        resolved, gen_seed = schema.generator(all_rules, 'root', seeder=list(s['seed']))
        ques, args = resolved.walk_and_resolve()
        pseudo = _args_to_pseudocode(args, schema_specs)
        pseudo_sql_query = _pseudoscode_to_sql(pseudo)
        # print('---')
        # print(i, s['question'])
        # print(pseudo)
        # print('|')
        # print(pseudo_sql_query)
        # print('|')
        # print(s['sql'])
        if not db_manager.is_query_equal(s['sql'], pseudo_sql_query):
            print('ERROR: query not equal ------------', i)
            print(s['question'])
            print(s['sql'])
            print('-------------------')
            print(pseudo_sql_query)
            print()
            assert False
        if 'pseudocode' not in s or s['pseudocode'] != pseudo:
            new_count += 1
        s['pseudocode'] = pseudo
        if i % 100 == 0:
            pbar.set_description(f'new_count: {new_count}')
            pbar.refresh()

    pbar.set_description(f'new_count: {new_count}')
    pbar.refresh()
    result_str = format_list_to_pretty_json(samples)
    if debug_mode:
        with open(str(env.home), 'w') as f:
            f.write(result_str)
    else:
        with open(samples_path, 'w') as f:
            f.write(result_str)








def format_list_to_pretty_json(l):
    result = ['[\n']
    for item in l:
        result.append('{')
        for i, key in enumerate(sorted(item.keys())):
            spaces = '   ' if i > 0 else ''
            line = '{}"{}": {},\n'.format(spaces, key, json.dumps(item[key]))
            result.append(line)
        if len(item) > 0:
            result[-1] = result[-1][:-2] + '\n'
        # result += '},'
        result.append('},')
    result[-1] = result[-1][:-1]
    result += ['\n]']
    result = ''.join(result)
    json.loads(result)  # make sure valid json
    return result

class MyDBManager:
    def __init__(self, cursors, mapping=None):
        self.cursors = cursors
        self._memoize = {}
        if mapping is not None:
            self.use_mapping = True
            self._setup_mapping(mapping)
        else:
            self.use_mapping = False

    def _setup_mapping(self, mapping_number):
        column_mapping = {
            '3-1': [('school_id', 'school_id'), ('territory', 'territory'), ('funding', 'funding'), ('capacity', 'capacity'), ],
            '3-2': [('pupil_id', 'pupil_id'), ('school_id', 'school_id'), ('study', 'study'), ('housing', 'housing'), ('cafeteria', 'cafeteria'), ('end_date', 'end_date'), ('performance', 'performance'), ('classes', 'classes'), ],  # , ('name', 'name')


            '2-1': [('appliance_id', 'appliance_id'), ('company', 'company'), ('type', 'type'), ('appliance_rating', 'appliance_rating'), ],
            '2-2': [('store_id', 'store_id'), ('location', 'location'), ('star_rating', 'star_rating') ],  # , ('name', 'name'),
            '2-3': [('inventory_id', 'inventory_id'), ('appliance_id', 'appliance_id'), ('store_id', 'store_id'), ('width', 'width'), ('height', 'height'), ('value', 'value'), ('available', 'available'), ],


            '1-1': [('teacher_id', 'teacher_id'), ('teacher_age', 'teacher_age'), ],  # ('name', 'name'), 
            '1-2': [('class_id', 'class_id'), ('teacher_id', 'teacher_id'), ('level', 'level'), ('year', 'year'), ('grade', 'grade'), ('class_subject', 'class_subject'), ],
        }

        animals = ['gorilla', 'leopard', 'bull', 'mule', 'cat', 'bat', 'ocelot', 'goat', 'skunk', 'mare', 'lamb', 'mongoose', 'chicken', 'giraffe', 'guanaco', 'dog', 'gazelle', 'alpaca', 'raccoon', 'squirrel', 'parrot', 'bear', 'turtle', 'sloth', 'weasel', 'pony', 'jaguar', 'kangaroo', 'hare', 'crow', 'moose', 'opossum', 'frog', 'baboon', 'octopus', 'lizard', 'bunny', 'ape', 'camel', 'donkey', 'wildcat', 'deer']

        assert mapping_number == '1', 'Only mapping==1 is currently supported'

        if mapping_number == '1':
            to_replace = animals
        else:
            raise Exception('Unknown mapping')

        ind = 0
        for table in column_mapping.values():
            for i, pair in enumerate(table):
                table[i] = (table[i][0], to_replace[ind])
                ind += 1

        replacement_dict = {}
        for table in column_mapping.values():
            for i, pair in enumerate(table):
                replacement_dict[table[i][0]] = table[i][1]
                replacement_dict[table[i][1]] = table[i][0]
                
        self._mapping_rep = dict((re.escape(k), v) for k, v in replacement_dict.items())
        self._mapping_pattetn = re.compile("|".join(self._mapping_rep.keys()))

    def apply_mapping(self, text):
        """applying the map twice should return the original string"""
        if not self.use_mapping:
            return text
        # from https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
        return self._mapping_pattetn.sub(lambda m: self._mapping_rep[re.escape(m.group(0))], text)

    def exec_query(self, query, memoize=False):
        if "drop" in query.lower(): return []
        if self.use_mapping:
            query = self.apply_mapping(query)
        if memoize:
            if query in self._memoize:
                return self._memoize[query]
        result = [cursor.execute(query).fetchall() for cursor in self.cursors]
        if memoize:
            self._memoize[query] = result
        return result

    def is_query_equal(self, ground, newq, memoize=False):
        """Returns 1 if the query is equal, 0 if not equal, -1 if error"""
        a = self.exec_query(ground, memoize=memoize)
        try:
            b = self.exec_query(newq, memoize=memoize)
        except Exception as e:
            return -1
        if len(a) != len(b):
            return 0
        for i1, i2 in zip(a, b):
            if len(i1) != len(i2):
                return 0
            # sort
            i1 = sorted(i1)
            i2 = sorted(i2)
            for j1, j2 in zip(i1, i2):
                if j1 != j2:
                    return 0
        return 1
    def close(self):
        for cursor in self.cursors:
            cursor.close()    
    @classmethod
    def get_manager(cls, filename, mapping=None, ERROR_ON_NO_DB=True):
        """Returns a MyDBManager object with cursors to all databases in the filename format with incrementing numbers."""
        cursors = []
        j = 0
        _opened = set()
        while True:
            f = filename.format(j)
            if not os.path.exists(f):
                break
            assert f not in _opened
            _opened.add(f)
            db = sqlite3.connect(f)
            cursors.append(db.cursor())
            j += 1

        my_db_manager = MyDBManager(cursors, mapping=mapping)
        if len(cursors) == 0:
            if ERROR_ON_NO_DB:
                raise Exception(f'No database found: {filename.format(j)}')
            print('No database found')
        return my_db_manager

def check_samples(query_file, database):
    with open(query_file, 'r') as f:
        queries = json.load(f)
    my_db_manager = MyDBManager.get_manager(database)

    error_count = 0
    total_count = 0
    for i in range(len(queries)):
        try:
            my_db_manager.exec_query(queries[i]['sql'])
        except Exception as e:
            print('Error on query {}: {}'.format(i, queries[i]['sql'].replace('\n', ' ')))
            print('Error:', e)
            error_count += 1
        total_count += 1
    my_db_manager.close()
    print()
    if error_count > 0:
        print('Error on {} out of {} queries'.format(error_count, total_count))
    else:
        print(f'All {total_count} queries are valid')

def rephrase_questions_with_llama(input_fn, output_fn, device, mode_from, samples=None):
    # device = 'cuda:6'
    # input_fn = '../schema/schema_data/samples_1_val.json'
    # output_fn = '../schema/schema_data/samples_1_val_r.json'
    # TODO Fix hack that adds parent dir to path
    import sys
    sys.path.append('../')
    import datasets, model_selector, train

    assert mode_from in ['question', 'sql']

    class tempSample(datasets.SingleSample):
        def __init__(self, text, *args, **kwargs):
            super().__init__(text=text, *args, **kwargs)
        def get_new_text(self, generated_text):
            if not generated_text.startswith(self.text):  # something went wrong
                print('WARNING: generated text does not start with original text', generated_text.replace('\n', '\\n'), self.text.replace('\n', '\\n'))
                return None
            return generated_text[len(self.text):]
        def get_generation(self):
            return self.text
        def is_generation_done(self, text):
            new_text = self.get_new_text(text)
            if new_text is None:  # something went wrong, cancel generation
                return True
            return '\n' in new_text

    def get_formatted_output(gen_output):
        if gen_output is None or not gen_output: # empty
            return ''
        if not gen_output.endswith('"\n'):  # not properly terminated
            return ''
        gen_output = gen_output[:-2].strip()
        bad_chars = ['\\', '\n', '\r', '\t']
        if any(i in gen_output for i in bad_chars):  # contains bad/unexpected characters
            return ''
        return gen_output

    prompt_question = """Statement: "what is the minimum grade of all classes for fifth graders that achieved a grade lower than 56 and were conducted before 2007 in our records"
    Rephrase: "What's the lowest grade among fifth grade classes whose students scored below 56 and the courses were held prior to 2007?"

    Statement: "provide the average age of all teachers that are older than 78 and that taught classes for 12th graders in our records"
    Rephrase: "Give me the mean age of teachers who are above the age of 78 and have taught 12th-grade classes in the database."

    Statement: "what's the maximum grade of all chemistry or literature classes in our records"
    Rephrase: "Show me the highest grade of any class that taught chemistry or literature."

    Statement: "{0}"
    Rephrase: "
    """
    prompt_sql = """SQL: " SELECT COUNT(DISTINCT appliance_id)\nFROM appliance\nWHERE appliance.manufacturer = 'GE' OR appliance.manufacturer = 'Sony' "
    Explination of SQL: " Count the number of appliances that are manufactured by GE or Sony "
------------------------
    SQL: " SELECT MIN(inventory.height)\nFROM inventory\nINNER JOIN store ON store.store_id = inventory.store_id\nWHERE inventory.available = 0 AND store.rating = 3 "
    Explination of SQL: " what is the minimum height of all appliances in the inventory that are currently unavailable in stores with a rating of 3 stars "
------------------------
    SQL: " SELECT MAX(inventory.value)\nFROM inventory\nINNER JOIN store ON store.store_id = inventory.store_id\nWHERE inventory.available = 1 AND store.rating <= 2 "
    Explination of SQL: " what's the maximum value of all appliances in the inventory that are currently available in stores with a rating lower than or equal to 2 stars "
------------------------
    SQL: " {0} "
    Explination of SQL: "
    """

    inp_data = json.load(open(input_fn))
    if samples is not None:
        samples = min(samples, len(inp_data))
    if samples is not None and samples < len(inp_data):
        print(f'ONLY {samples} SAMPLES')
        inp_data = inp_data[:samples]

    if os.path.exists(output_fn):
        print('reading file that already exists')
        out_data = json.load(open(output_fn))
        assert len(inp_data) == len(out_data)
        assert all(inp_data[i]['seed'] == out_data[i]['seed'] for i in range(len(inp_data)))
        assert all(inp_data[i]['sql'] == out_data[i]['sql'] for i in range(len(inp_data)))
    else:
        print('output json doesnt exist. will create it')

    class model_args:
        name = 'llama-2-7b'
        max_seq_len = '1024'
        max_batch_size = '32'

        # type = 'adapter'
        # adapter_len = 0
        # adapter_layer = 0

        type = 'lora'
        lora_r = 0
        lora_alpha = 0
        lora_dropout = 0.1
    mymodel1 = model_selector.main_get_model(model_args=model_args, world_size=1, device=device, verbose=True)

    tokenizer = mymodel1.tokenizer
    lst = []
    if mode_from == 'question':
        lst = [tempSample(prompt_question.format(v['question'])) for v in inp_data]
    elif mode_from == 'sql':
        lst = [tempSample(prompt_sql.format(v['sql'])) for v in inp_data]

    for s in lst: s.set_tokenized_len(tokenizer)

    bad_output_count = 0
    gens = []
    pbar = tqdm(total=len(lst), dynamic_ncols=True)
    while len(lst) > 0:
        dataset = datasets.DynamicBatchLoader(lst, max_inp_matrix_size=8000, max_batch_size=32, shuffle=False, size_getter=lambda x: x.tokenized_len)
        for gen in train.llm_generate(mymodel1.model, mymodel1.tokenizer, dataset, temprature=0.6, max_gen_len=75):
            gen['output_before_cleaning'] = gen['sample'].get_new_text(gen['gen_text'])
            gen['output'] = get_formatted_output(gen['output_before_cleaning'])
            if len(gen['output']) == 0:  # bad output
                bad_output_count += 1
                pbar.set_description(f'Bad output count: {bad_output_count}')
                pbar.refresh()
                continue
            gens.append(gen)
            lst.remove(gen['sample'])
            pbar.update(1)
            pbar.refresh()
            # save every x sample
            if len(gens) % 100 == 0:
                gens = sorted(gens, key=lambda x: x['i'])
                out_data = [{**v, 'question': gen['output'], 'i': gen['i']}  for v, gen in zip(inp_data, gens)]
                res = format_list_to_pretty_json(out_data)
                with open(output_fn, 'w') as f:
                    f.write(res)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help='sub-command help', dest='command')
    # subparser generate
    parser_generate = subparsers.add_parser('generate', help='Generate .json dataset (question/sql pairs) from CFG')
    parser_generate.add_argument('--schema', type=str, required=True, help='path to schema file. e.g. ./schema_qgen1.txt')
    parser_generate.add_argument('--num_samples', type=int, required=True, help='number of samples to generate')
    parser_generate.add_argument('--save_to', type=str, required=True, help='path to save the generated dataset. e.g. ./schema_data/samples_1.json')

    # subparser check
    parser_check = subparsers.add_parser('check', help='Check if the queries in a generated .json file are valid (don\'t crash) against a database')
    parser_check.add_argument('--query_file', type=str, required=True, help='path to query file. e.g. ./schema_data/samples_2.json')
    parser_check.add_argument('--database', type=str, required=True, help='path to database file. e.g. ./schema_data/schema_2_{0}.db')

    # subparser postprocess
    parser_check = subparsers.add_parser('postprocess', help='Postprocess a generated .json file to remove zero records and add pseudocode')
    parser_check.add_argument('--query_file', type=str, required=True, help='path to query file. e.g. ./schema_data/samples_2.json')
    parser_check.add_argument('--database', type=str, required=True, help='path to database file. e.g. ./schema_data/schema_2_{0}.db')
    parser_check.add_argument('--qgen', type=str, required=True, help='path to qgen file. e.g. ./schema_qgen1.txt')
    parser_check.add_argument('--num_queries', type=int, required=True, help='number of queries to keep')

    # subparser rephrase
    parser_rephrase = subparsers.add_parser('rephrase', help='Rephrase questions using Llama')
    parser_rephrase.add_argument('--input_fn', type=str, required=True, help='path to input json file. e.g. ./schema_data/samples_2.json')
    parser_rephrase.add_argument('--output_fn', type=str, required=True, help='path to output json file. e.g. ./schema_data/samples_2_r.json')
    parser_rephrase.add_argument('--device', type=str, required=True, help='device to use. e.g. cuda:6')
    parser_rephrase.add_argument('--samples', type=int, default=None, help='number of samples to rephrase. default is all samples')
    parser_rephrase.add_argument('--mode_from', type=str, default='question', help='whether to rephrase using the question or sql. default is question')

    args = parser.parse_args()
    if args.command == 'generate':
        generate_dataset(args.schema, args.num_samples, args.save_to)
    elif args.command == 'postprocess':
        remove_zero_records(args.query_file, args.database, args.num_queries)
        add_pseudocode_to_samples(args.query_file, args.qgen, args.database, debug_mode=False)
        remove_no_condition_records(args.query_file, args.database, debug_mode=False)
    elif args.command == 'check':
        check_samples(args.query_file, args.database)
    elif args.command == 'rephrase':
        rephrase_questions_with_llama(input_fn=args.input_fn, output_fn=args.output_fn, device=args.device, mode_from=args.mode_from, samples=args.samples)
    elif args.command is None:
        parser.print_help()
    else:
        raise Exception('Unknown command: {}'.format(args.command))
    
