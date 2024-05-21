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
    df = pd.concat([parse_log(log) for log in Path('.').glob(f'logs/T{table_num}.*.log')])

    # keep only maincol
    new_df = df[['name', 'dataset', maincol]]
    new_df.loc[:, maincol] = new_df[maincol].astype(float).round(round_digits)
    if show_accuracy:
        # for accuracy >5%
        # show_acc_for = df['acc'].astype(float) > 5.0
        # for all
        # show_acc_for = df['acc'].astype(float) > -1
        # for first two columns
        show_acc_for = df['name'].str.contains('C1|C2')
        new_df.loc[show_acc_for, maincol] = new_df.loc[show_acc_for, maincol].astype(str) + ' (' + df.loc[show_acc_for, 'acc'] + ')'
    new_df.loc[:, 'dataset'] = new_df['dataset'].str.replace('schemapseudo', 'schema') # rename schemapseudo to schema in dataset
    new_df = new_df.pivot(index='dataset', columns='name', values=maincol)
    print(new_df)
    if show_sql:
        print(new_df.to_latex())
    new_df.to_csv(f'logs/table{table_num}.csv', index=False)





SHOW_SQL = True
SHOW_ACCURACY = True

generate_table(table_num=1, show_accuracy=SHOW_ACCURACY, round_digits=1, maincol='tree-norm', show_sql=SHOW_SQL)
generate_table(table_num=2, show_accuracy=SHOW_ACCURACY, round_digits=1, maincol='tree-norm', show_sql=SHOW_SQL)
generate_table(table_num=3, show_accuracy=SHOW_ACCURACY, round_digits=1, maincol='tree-norm', show_sql=SHOW_SQL)
