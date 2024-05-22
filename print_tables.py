import pandas as pd
from pathlib import Path
import re

def parse_log(log_path):
    with open(log_path) as f:
        lines = f.readlines()
    # ingore everything before === Results ===
    results = []
    results_started = False
    for line in lines:
        if line.strip() == '=== Results ===':
            results_started = True
            continue
        if results_started:
            results.append(line.strip().split(','))  # dataset, acc, tree, tree-norm
    # convert to dataframe
    results = pd.DataFrame(results, columns=['dataset', 'acc', 'tree', 'tree-norm'])
    results['name'] = log_path.stem
    return results


def generate_table(table_num, round_digits=1, maincol='tree-norm', acc_cols=[], print_sql=True):
    # table table_num
    print('------------- GENERATING TABLE', table_num, '-------------')
    allpaths = list(Path('.').glob(f'logs/T{table_num}.*.log'))
    if not allpaths:
        print(f'WARNING: No logs found for table {table_num} Skipping table {table_num}...\n'*30)
        return
    print('Found', len(allpaths), 'logs')
    df = pd.concat([parse_log(log) for log in allpaths])
    df = df.reset_index(drop=True)

    # keep only maincol
    new_df = df[['name', 'dataset', maincol]].copy()
    new_df.loc[:, maincol] = new_df[maincol].astype(float)

    # mean and std for each model
    mean_std = new_df.groupby('name').agg({maincol: ['mean', 'std']})
    mean_std.columns = mean_std.columns.droplevel(0)
    mean_std.loc[:, 'mean'] = mean_std['mean'].astype(float).round(2)
    mean_std.loc[:, 'std'] = mean_std['std'].astype(float).round(2)
    mean_std['mean_std'] = mean_std['mean'].astype(str) + ' +- ' + mean_std['std'].astype(str)

    new_df.loc[:, maincol] = new_df[maincol].astype(float).round(round_digits).astype(str)
    if acc_cols:
        # show_acc_for = df['acc'].astype(float) > 5.0  # for accuracy >5%
        show_acc_for = df['name'].str.contains('|'.join(acc_cols))

        accs = df.loc[show_acc_for, 'acc'].astype(float).round(1).astype(str)
        new_df.loc[show_acc_for, maincol] = new_df.loc[show_acc_for, maincol] + ' (' + accs + '%)'
    new_df.loc[:, 'dataset'] = new_df['dataset'].str.replace('schemapseudo', 'schema').str.replace('_val', '').str.replace('schema_', 'S').str.replace('union', '').str.replace('_gpt', '') # rename datasets to just S#
    new_df = new_df.pivot(index='dataset', columns='name', values=maincol)
    to_sort = ['S1', 'S2', 'S3', 'S12', 'S13', 'S23', 'S123']  # sort rows by this order
    new_df = new_df.reindex(to_sort)
    new_df.loc['mu +- std'] = mean_std['mean_std'].to_dict()  # new row for 'mu +- std'
    if COLS_TO_BOLD:  # find minimums for row and bold them
        min_values = new_df.copy()
        piped = '|'.join(COLS_TO_BOLD)
        min_values = min_values.loc[:, min_values.columns.str.contains(piped)]  # dont calculate minimums over c1 or c2
        min_values = min_values.apply(lambda x: x.str.extract(r'(.*?)( |$)')[0].astype(float), axis=1).min(axis=1)
        for dname in new_df.index:
            min_val = min_values[dname]
            # cols that contained piped
            cols = new_df.columns[new_df.columns.str.contains(piped)]
            new_df.loc[dname, cols] = new_df.loc[dname, cols].apply(lambda x: f'\\textbf{{{x}}}' if re.match(rf'{min_val}( |$)', x) else x)
    print(new_df)
    if print_sql:
        latex_df = new_df.copy()
        # if column has "\d+ .+" then add \phantom{0} to it
        def add_phantom(x):
            s = r'([\d\.]+?) (\(.+)'
            v12 = re.match(s, x)
            if v12:
                zero_count = 8 - len(v12.group(2))
                if zero_count > 0:
                    phantom_str = '{' + '0' * zero_count + '}'
                    return f'{v12.group(1)} \\phantom{phantom_str}{v12.group(2)}'
            return x
        latex_df = latex_df.applymap(add_phantom)  # add phantom 0 to columns with accuracy
        dataset_mapping = {'S1': '$\\text{Silos}_1$', 'S2': '$\\text{Silos}_2$', 'S3': '$\\text{Silos}_3$', 'S12': '$\\text{Silos}_{1 \cup 2}$', 'S13': '$\\text{Silos}_{1 \cup 3}$', 'S23': '$\\text{Silos}_{2 \cup 3}$', 'S123': '$\\text{Silos}_{1 \cup 2 \cup 3}$'}
        latex_df = latex_df.rename(index=dataset_mapping)
        latex = latex_df.to_latex()
        latex = latex[latex.find('\\midrule')+len('\\midrule'):latex.rfind('\\bottomrule')]  # remove everything before midrule and after bottomrule
        latex = latex.replace('%', '\\%')  # replace % with \%
        latex = latex.replace('mu +- std', '\midrule $\\mu\\pm\\sigma$')  # replace mu +- std with latex
        latex = re.sub(r'\\textbf\{(\d+.\d+) \+\- (\d+.\d+)\}', r'$\\boldsymbol{\1\\pm\2}$', latex)  # replace bolded +- values with latex
        latex = re.sub(r'(\d+.\d+) \+\- (\d+.\d+)', r'$\1\\pm\2$', latex)  # replace 1.2 +- 0.3 with latex
        print('----------- SQL -----------')
        print(latex)
        with open(f'logs/table{table_num}.tex', 'w') as f:
            f.write(latex)
    new_df.to_csv(f'logs/table{table_num}.csv', index=False)




# COLS_TO_BOLD = []
COLS_TO_BOLD = ['C3', 'C4', 'C5', 'C6', 'C7']

PRINT_SQL = True

COLUMNS_SHOW_ACCURACY = ['C1', 'C2']
# COLUMNS_SHOW_ACCURACY = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

generate_table(table_num=1, acc_cols=COLUMNS_SHOW_ACCURACY, round_digits=1, maincol='tree-norm', print_sql=PRINT_SQL)
generate_table(table_num=2, acc_cols=COLUMNS_SHOW_ACCURACY, round_digits=1, maincol='tree-norm', print_sql=PRINT_SQL)
generate_table(table_num=3, acc_cols=COLUMNS_SHOW_ACCURACY, round_digits=1, maincol='tree-norm', print_sql=PRINT_SQL)
