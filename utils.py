import re
import edist.uted
import argparse

class ParseKwargs(argparse.Action):
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
        def __getstate__(self): return self.__dict__
        def __setstate__(self, d): self.__dict__.update(d)


    '''parse kwargs as key=value pairs from command line'''
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.dotdict())
        for value in values:
            assert '=' in value, f'invalid argument: {value}. Must be in the form key=value (no unescaped spaces)'
            key, value = value.split('=', 1)
            getattr(namespace, self.dest)[key] = value

def print_tree(tree, tree_inds):
    from treelib import Node, Tree
    result = Tree()
    todo = [(0, None)]
    while todo:
        # print(todo)
        i, parent = todo.pop()
        # print('adding', i, parent, 'name', tree[i])
        if parent is None:
            # print('root')
            result.create_node(tree[i], i)
        else:
            # print('child')
            # print('error', i)
            result.create_node(tree[i], i, parent=parent)
        for child_id in tree_inds[i]:
            todo.append((child_id, i))

    print(result.show(stdout=False))

def _split_andor_no_bracket(text):

    def _clean_result(lst):
        # remove empty strings
        lst = [x for x in lst if x]
        # if single element and list -> return element
        if len(lst) == 1 and not isinstance(lst[0], str):
            lst = lst[0]
            _clean_result(lst)
        return lst

    stack = []
    to_find = [(r'\sAND\s', 5), (r'\sOR\s', 4)]
    cur_i = 0
    prev_i = 0
    while cur_i < len(text):
        token = text[cur_i]
        # print(cur_i, 't', token)
        if token == '(':  # recursive
            stack.append(text[prev_i:cur_i])
            rest, new_i = _split_andor_no_bracket(text[cur_i+1:])
            # print('exit', new_i, rest)
            stack.append(rest)
            # print('after exit', stack)
            cur_i = cur_i + new_i + 2
            prev_i = cur_i
        elif token == ')':
            stack.append(text[prev_i:cur_i])
            stack = _clean_result(stack)
            return stack, cur_i
        else:
            for word, length in to_find:
                if re.match(word, text[cur_i:cur_i+length], re.IGNORECASE):
                    stack.append(text[prev_i:cur_i])
                    stack.append(text[cur_i:cur_i+length])
                    prev_i = cur_i+length
                    cur_i = prev_i
                    break
            else:
                cur_i += 1
    stack.append(text[prev_i:])
    stack = _clean_result(stack)
    return stack, cur_i

def _fix_both_conds_same_level(lst):
    """If AND and OR are on the same level, fix it by adding brackets around the AND because AND has higher precedence"""

    for i, ele in enumerate(lst):  # recursive
        if isinstance(ele, list):
            lst[i] = _fix_both_conds_same_level(ele)

    contains_and = [x.strip().upper() == 'AND' for x in lst if isinstance(x, str)]
    contains_or = [x.strip().upper() == 'OR' for x in lst if isinstance(x, str)]
    result = []
    if any(contains_and) and any(contains_or):
        # needs fixing
        last_or_i = 0
        and_seen = False
        for i, (a, o) in enumerate(zip(contains_and, contains_or)):
            if a:
                and_seen = True
            if o:
                if and_seen:
                    result.append(lst[last_or_i:i])  # currently switching from and to or, brackets to the and part
                    result.append(lst[i])  # add the OR
                else:
                    result.extend(lst[last_or_i:i])  # no and seen, keep conditions top level
                    result.append(lst[i])  # add the OR
                last_or_i = i + 1
                and_seen = False
                
        if last_or_i != i + 1:
            if and_seen:
                result.append(lst[last_or_i:])  # currently switching from and to or, brackets to the and part
            else:
                result.extend(lst[last_or_i:])  # no and seen, keep conditions top level

    else:
        result = lst

    return result

def _nested_lst_to_tree(lst, parent_ind, tree, tree_inds):
    for p in lst:
        if isinstance(p, list):
            tree.append('')
            tree_inds.append([])
            tree_inds[parent_ind].append(len(tree)-1)
            _nested_lst_to_tree(p, parent_ind=len(tree)-1, tree=tree, tree_inds=tree_inds)
        elif isinstance(p, str):
            tree.append(p.strip().upper())
            tree_inds.append([])
            tree_inds[parent_ind].append(len(tree)-1)
        else:
            raise ValueError('p is neither list nor str')
    return tree, tree_inds

def conds_to_tree(conds):
    matches, _ = _split_andor_no_bracket(conds)
    matches = _fix_both_conds_same_level(matches)
    tree = ['root']
    tree_inds = [[]]
    _nested_lst_to_tree(matches, parent_ind=0, tree=tree, tree_inds=tree_inds)
    return tree, tree_inds

def conds_distance(cond1, cond2, debug=False):
    x_nodes, x_adj = conds_to_tree(cond1)
    y_nodes, y_adj = conds_to_tree(cond2)
    if debug:
        print_tree(x_nodes, x_adj)
        print_tree(y_nodes, y_adj)
    dist = edist.uted.uted(x_nodes, x_adj, y_nodes, y_adj)
    dist_norm = dist / (len(x_nodes))
    return dist, dist_norm
