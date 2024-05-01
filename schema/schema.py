import random
from typing import Union
import json
import re
import itertools
from collections import defaultdict


DEFINED_TERMINAL_RULES = ['number']

def _has_unwrapped_sub(str, substr):
    '''Return True if there is an OR/AND that is not wrapped by parenthesis'''
    res = re.search("(?<!\()" + substr + "(?![\w\s]*[\)])", str) is not None
    # if res: print('DETECTED', str)
    return res

def _remove_double_space(s):
    return ' '.join(x for x in s.split(' ') if x)

class CONST:
    TRANSFORM = 'T'
    WANT = 'WANT'
    TABLES = 'TABLES'
    CONDITIONS = 'CONDITIONS'
    ADDER_PROPS = 'ADDER_PROPS'
    REFS = 'REFS'


class Rule:
    """One instance of this class represents one rule (one line) in the schema."""
    _id_iter = itertools.count()
    def __init__(self, name, rule_lst, args, nickname, weight):
        self.name = name
        self.rule_lst = rule_lst
        self.args = args
        self.nickname = nickname
        self.weight = weight
        self._rule_unique_id = next(Rule._id_iter)

    def __repr__(self):
        return 'STR={} | ARGS={}'.format(self.rule_lst, self.args)
    
    def needed_children(self):
        return [{'id': x[1], 'pos': i} for i, x in enumerate(self.rule_lst) if x[0] == 'rule']
    
    @staticmethod
    def add_args(args1, args2, props: dict):
        res = {}
        for k, v in (*args1.items(), *args2.items()):
            if k == CONST.REFS:  # simple dict merge
                res[CONST.REFS] = {**res.get(CONST.REFS, {}), **v}
                continue
            if k not in res:
                res[k] = v
            else:  # key already exists
                if isinstance(res[k], str):
                    res[k] = [res[k]]
                if isinstance(v, str):
                    v = [v]
                # now both are list
                if k != CONST.CONDITIONS:
                    res[k] += v
                else:  # special case for CONDITIONS
                    comb_type = props.get('comb', 'AND')
                    if comb_type == 'AND':  # ADD is default and is done automatically by args_to_sql
                        res[k] += v
                    elif comb_type == 'OR': # OR needs to be done here
                        a = _concat_conditions(res[k], comb_type='AND')  # AND is default
                        b = _concat_conditions(v, comb_type='AND')  # AND is default
                        res[k] = _concat_conditions([a, b], comb_type='OR')
                    else:
                        raise Exception('Invalid comb type: {}'.format(comb_type))
        return res

class ResolvedRule:
    """Instances of this class represents a resolved rule, most rules can resolve in many different ways depending on the choice of which children to use."""
    def __init__(self, node: Rule, children: list['ResolvedRule']):
        self.node = node
        self.children = children
    
    def __repr__(self, level=0):
        if len(self.children) == 0:
            return str(self.node)
        return str(self.node) + '\n' + '\n'.join('|'*(level) + '+-' + x.__repr__(level=level+2) for x in self.children)

    def walk_and_resolve(self):
        result = []
        child_index = 0
        children_args = {}
        resolved_children = {}
        arg_adder_props = json.loads(self.node.args.get(CONST.ADDER_PROPS, '{}'))
        for part_type, part_content in self.node.rule_lst:
            if part_type == 'text':
                result.append(part_content)
            elif part_type == 'rule':
                child = self.children[child_index]
                child, resolved_child_args = child.walk_and_resolve()
                children_args = Rule.add_args(children_args, resolved_child_args, props=arg_adder_props)  # merge args
                resolved_children[str(child_index)] = child
                child_index += 1
                if len(child) == 0:  # child resolved to empty string
                    continue
                result.append(child)
            else:
                raise Exception('Invalid part: {} {}'.format(part_type, part_content))
        result = _remove_double_space(''.join(result))  # hack: remove double space caused by "text {var}" and {var} is empty
        resolved_children = {**resolved_children, **children_args.get(CONST.REFS, {})}  # add REFS so current args can reference them
        cur_args = self.resolve_args(resolved_children)
        result_args = Rule.add_args(cur_args, children_args, props=arg_adder_props)
        return result, result_args

    def resolve_args(self, resolved_children):
        result = {}
        for k,v_part in self.node.args.items():
            if k in [CONST.ADDER_PROPS, CONST.REFS]:  # special prop thats json encoded
                result[k] = json.loads(v_part) if isinstance(v_part, str) else v_part
                continue
            v_reconstructed = []
            while '{' in v_part:  # covert {0} or {1} etc... to resolved child
                start = v_part.index('{')
                end = v_part.index('}')
                assert end >= 0, 'Invalid part: {}'.format(v_part)
                if start > 0:
                    v_reconstructed.append(v_part[:start])
                child_ref = v_part[start+1:end]
                v_reconstructed.append(resolved_children[child_ref])
                v_part = v_part[end+1:]
            if v_part:
                v_reconstructed.append(v_part)

            result[k] = ''.join(v_reconstructed)
        return result

def _resolve_terminal(rule_name, seeder=None):
    rule_name, *params = rule_name.split(',')
    assert rule_name in DEFINED_TERMINAL_RULES, 'Invalid terminal: {}'.format(rule_name)
    if rule_name == 'number':
        assert len(params) == 2, 'Invalid number: {}'.format(params)
        random_int = seeder if seeder is not None else random.randint(int(params[0]), int(params[1]))
        node = Rule('terminal_number', [['text', str(random_int)]], {}, 'terminal_number', 1)
        res = ResolvedRule(node, [])
        res_seed = random_int
        return res, res_seed
    else:
        raise Exception('Unhandled terminal: {} SHOULD NEVER HAPPEN'.format(rule_name))

def generator(all_rules, rule_name, seeder=None, parent_duplicate_preventor=None) -> tuple[ResolvedRule, list]:
    if seeder is None:
        seeder = []
    if rule_name not in all_rules:  # terminal
        i = None if len(seeder)==0 else seeder.pop(0)
        res, res_seed = _resolve_terminal(rule_name, i)
        return res, [res_seed]

    choices = all_rules[rule_name]
    if len(seeder) > 0:
        i = seeder.pop(0)
    else:
        weights = [x.weight for x in choices]
        if parent_duplicate_preventor is not None and rule_name in parent_duplicate_preventor:  # prevent duplicates with siblings
            for i in parent_duplicate_preventor[rule_name]:
                weights[i] = 0
        i = random.choices(range(len(choices)), weights=weights)[0]

    result_node = choices[i]
    result_children = []
    result_seed = [i]
    children_duplicate_preventor = defaultdict(list)  # prevent duplicates with children

    needed_children = result_node.needed_children()
    for child in needed_children:
        new_child, new_seed = generator(all_rules, child['id'], seeder=seeder, parent_duplicate_preventor=children_duplicate_preventor)
        result_children.append(new_child)
        result_seed += new_seed
        children_duplicate_preventor[child['id']].append(new_seed[0])  # first int is the seed my direct child used
    result = ResolvedRule(node=result_node, children=result_children)
    return result, result_seed

def _concat_conditions(conds, comb_type='AND'):
    if len(conds) == 1:  # only one condition no need to do anything
        return conds[0]
    if comb_type == 'AND':
        return ' AND '.join(x if not _has_unwrapped_sub(x, substr='OR') else '(' + x + ')' for x in conds)
    elif comb_type == 'OR':
        return ' OR '.join(conds)
    else:
        raise Exception('Invalid comb type: {}'.format(comb_type))

def _prepare_args_to_sql(args, schema_specs):
    result = {'SELECT': '', 'FROM': '', 'JOIN': [], 'WHERE': '', 'ALL_TABLES': []}
    # take care of single element being string instead of list
    if CONST.TABLES in args and isinstance(args[CONST.TABLES], str):
        args[CONST.TABLES] = [args[CONST.TABLES]]
    elif CONST.TABLES not in args:
        args[CONST.TABLES] = []
    if CONST.CONDITIONS in args and isinstance(args[CONST.CONDITIONS], str):
        args[CONST.CONDITIONS] = [args[CONST.CONDITIONS]]
    elif CONST.CONDITIONS not in args:
        args[CONST.CONDITIONS] = []
    
    # --- take care of tables, FROM and JOIN
    _all_refs = ' '.join(args[CONST.CONDITIONS]) + ' ' + (args[CONST.WANT] if CONST.WANT in args else '')  # all references to tables (i.e. "table.column")
    args[CONST.TABLES].extend(re.findall("(?<!\w)(\w*?)\.", _all_refs))  # add everything before "." which are tables from conditions
    # print(CONST.TABLES, args[CONST.TABLES], CONST.CONDITIONS, args[CONST.CONDITIONS])
    assert len(args[CONST.TABLES]) > 0, 'No tables specified'
    result['ALL_TABLES'] = [args[CONST.TABLES][0]]
    if len(set(args[CONST.TABLES])) == 1:
        result['FROM'] = args[CONST.TABLES][0]
        result['JOIN'] = ''
    else:  # multiple tables
        result['FROM'] = args[CONST.TABLES][0]
        _done_tables = set([args[CONST.TABLES][0]])  # keep track of tables that are already joined
        _todo_tables = set(args[CONST.TABLES]) - _done_tables  # keep track of tables that are not joined yet
        # I HAVE NO IDEA HOW TO MAKE THIS SIMPLER
        # The point is I have a schema with many tables, the tables are connected by JOINs, I need to find a path to join all tables needed
        while _todo_tables:
            for _todo_table, _done_table, schema_spec_join in itertools.product(_todo_tables, _done_tables, schema_specs['JOINs']):
                if set((_todo_table, _done_table)) == set(schema_spec_join[CONST.TABLES]):
                    # found a join
                    if 'COLUMNS' in schema_spec_join:  # able to join
                        join_col = schema_spec_join['COLUMNS']
                        if len(join_col) == 1:
                            join_col1, join_col2 = join_col[0][0], join_col[0][1]
                            on_str = '{}.{} = {}.{}'.format(schema_spec_join[CONST.TABLES][0], join_col1, schema_spec_join[CONST.TABLES][1], join_col2)
                        else:
                            assert False, 'NOT IMPLEMENTED YET...'
                            # on_str = ' AND '.join(['{}.{} = {}.{}'.format(schema_spec_join[CONST.TABLES][0], x, schema_spec_join[CONST.TABLES][1], x) for x in join_col])
                        result['JOIN'] += ['INNER JOIN ' + _todo_table + ' ON ' + on_str]
                        result['ALL_TABLES'].append(_todo_table)
                        _done_tables.add(_todo_table)
                        _todo_tables.remove(_todo_table)
                        break
                    else:  # cannot join
                        new_tables = set(schema_spec_join['LINKING_TABLES']) - _done_tables - _todo_tables
                        if not new_tables:  # nothing gained from this pair
                            continue
                        _todo_tables.add(*new_tables)
                        break
            else:
                raise Exception('Cannot find join for {} and {}'.format(_todo_tables, _done_tables))
        

    # --- FROM and JOIN are done
    # --- take care of WHERE
    if CONST.CONDITIONS not in args or not args[CONST.CONDITIONS]:
        result['WHERE'] = ''
    else:
        result['WHERE'] = 'WHERE ' + _concat_conditions(args[CONST.CONDITIONS], comb_type='AND')
    # --- WHERE is done
    # --- take care of SELECT
    if CONST.WANT not in args:  # no SELECT specified, select PK of FROM table
        result['SELECT'] = result['FROM'] + '.' + schema_specs['PKs'][result['FROM']]
    else:
        assert isinstance(args[CONST.WANT], str), 'Invalid SELECT: {}'.format(args[CONST.WANT])
        result['SELECT'] = args[CONST.WANT]
    if CONST.TRANSFORM in args:  # apply transformation if needed
        if args[CONST.TRANSFORM].lower() == 'none':
            pass
        elif args[CONST.TRANSFORM] == 'COUNT':
            result['SELECT'] = 'COUNT(DISTINCT {})'.format(result['SELECT'])
        elif args[CONST.TRANSFORM] in ['AVG', 'SUM', 'MAX', 'MIN']:
            result['SELECT'] = '{}({})'.format(args[CONST.TRANSFORM], result['SELECT'])
        else:
            raise Exception('Invalid T: {}'.format(args[CONST.TRANSFORM]))
    # --- SELECT is done
    return result

def args_to_sql(args, schema_specs):
    result = _prepare_args_to_sql(args, schema_specs)
    result = 'SELECT ' + result['SELECT'] + '\n' \
            + 'FROM ' + result['FROM'] + '\n' \
            + ('\n'.join(result['JOIN']) + '\n' if result['JOIN'] else '') \
            + (result['WHERE'] + '\n' if result['WHERE'] else '')
    result = result.strip()
    return result


def parse_schema_get(filename):
    with open(filename, 'r') as f:
        all_txt = f.read().split('-----------------------------------SCHEMA-SECTION-SEPERATOR-----------------------------------')
    # ------- first section of schema ------- #
    rules = {}
    for line_num, line in enumerate(all_txt[0].split('\n')):
        rule = line.split('#')[0]  # remove comments
        if not rule:
            continue
        rule = rule.split(':', 4)  # split upto 5 parts
        assert 4 <= len(rule) <= 5, 'Invalid rule: {} (line {}) len={}'.format(line, line_num, len(rule))
        rule_nickname, rule_weight, rule_name, rule_str = [x.strip() for x in rule[:4]]
        rule_args_str = rule[4].strip() if len(rule) > 4 else ''

        rule_lst = []
        while '{' in rule_str:
            start = rule_str.index('{')
            end = rule_str.index('}')
            assert end >= 0, 'Invalid rule: {}'.format(rule_str)
            if start > 0:
                rule_lst.append(['text', rule_str[:start]])
            rule_lst.append(['rule', rule_str[start+1:end]])  # don't include the curly braces
            rule_str = rule_str[end+1:]
        if rule_str:
            rule_lst.append(['text', rule_str])

        args = {}
        if rule_args_str:
            for arg in rule_args_str.split(';'):
                if arg.strip().lower() in ('none', ''):
                    continue
                arg = [x.strip() for x in arg.split('=', 1)]
                assert arg[0] in CONST.__dict__.values(), 'Invalid arg: {}'.format(arg)
                assert len(arg) == 2, 'Invalid arg: {}'.format(arg)
                args[arg[0]] = arg[1]

        new_rule = Rule(
            rule_name,
            rule_lst, 
            args,
            rule_nickname,
            float(rule_weight),
            )
        if rule_name not in rules:
            rules[rule_name] = []
        rules[rule_name].append(new_rule)
    # ------- second section of schema ------- #
    schema_specs = json.loads(all_txt[1])
    return rules, schema_specs

def cycle_check_and_count_combinations(rules, current_rule_name='root', parents=None):
    if parents is None:
        parents = set()
    parents = set(parents) # copy
    parents.add(current_rule_name)
    rules_counts = [1 for _ in rules[current_rule_name]]
    for rule_i, rule in enumerate(rules[current_rule_name]):
        for part in rule.rule_lst:
            if part[0] == 'rule':
                name, *params = part[1].split(',')
                if name in DEFINED_TERMINAL_RULES:
                    rules_counts[rule_i] *= int(params[1]) - int(params[0]) + 1  # comment if you want to ignore terminal node count
                    continue
                if part[1] in parents:
                    raise Exception('Cycle detected: {} in rule {}'.format(part[1], current_rule_name))
                rules_counts[rule_i] *= cycle_check_and_count_combinations(rules, part[1], parents)
    return sum(rules_counts)