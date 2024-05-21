import pandas as pd
from pathlib import Path

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


def generate_table(table_num, round_digits=1, maincol='tree-norm', show_accuracy=True, show_sql=True):
    # table table_num
    print('------------- GENERATING TABLE', table_num, '-------------')
    allpaths = list(Path('.').glob(f'logs/T{table_num}.*.log'))
    if not allpaths:
        print(f'WARNING: No logs found for table {table_num} Skipping table {table_num}...\n'*30)
        return
    df = pd.concat([parse_log(log) for log in allpaths])

    # keep only maincol
    new_df = df[['name', 'dataset', maincol]]
    new_df.loc[:, maincol] = new_df[maincol].astype(float)

    # mean and std for each model
    mean_std = new_df.groupby('name').agg({maincol: ['mean', 'std']})
    mean_std.columns = mean_std.columns.droplevel(0)
    mean_std.loc[:, 'mean'] = mean_std['mean'].astype(float).round(2)
    mean_std.loc[:, 'std'] = mean_std['std'].astype(float).round(2)
    mean_std['mean_std'] = mean_std['mean'].astype(str) + ' \pm ' + mean_std['std'].astype(str)

    new_df.loc[:, maincol] = new_df[maincol].astype(float).round(round_digits)
    if show_accuracy:
        # for accuracy >5%
        # show_acc_for = df['acc'].astype(float) > 5.0
        # for all
        # show_acc_for = df['acc'].astype(float) > -1
        # for first two columns
        show_acc_for = df['name'].str.contains('C1|C2')
        new_df.loc[show_acc_for, maincol] = new_df.loc[show_acc_for, maincol].astype(str) + ' (' + df.loc[show_acc_for, 'acc'] + ')'
    new_df.loc[:, 'dataset'] = new_df['dataset'].str.replace('schemapseudo', 'schema').str.replace('_val', '').str.replace('schema_', 'S').str.replace('union', '').str.replace('_gpt', '') # rename datasets to just S#
    new_df = new_df.pivot(index='dataset', columns='name', values=maincol)
    to_sort = ['S1', 'S2', 'S3', 'S12', 'S13', 'S23', 'S123']  # sort rows by this order
    new_df = new_df.reindex(to_sort)
    new_df.loc['mu +- std'] = mean_std['mean_std'].to_dict()  # new row for 'mu +- std'
    print(new_df)
    if show_sql:
        print('----------- SQL -----------')
        print(new_df.to_latex())
    new_df.to_csv(f'logs/table{table_num}.csv', index=False)





SHOW_SQL = True
SHOW_ACCURACY = True

generate_table(table_num=1, show_accuracy=SHOW_ACCURACY, round_digits=1, maincol='tree-norm', show_sql=SHOW_SQL)
generate_table(table_num=2, show_accuracy=SHOW_ACCURACY, round_digits=1, maincol='tree-norm', show_sql=SHOW_SQL)
generate_table(table_num=3, show_accuracy=SHOW_ACCURACY, round_digits=1, maincol='tree-norm', show_sql=SHOW_SQL)
