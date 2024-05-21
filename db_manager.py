import os
import sqlite3
import pandas as pd
from pathlib import Path

# TODO much needed improvement in future database , 
# 1. add a column for args not to be dumped in info_json
# 2. move deleted models to "lost and found" folder for perma deletion
# 3. seperate dir to store models from db file to not cause mess in explorer view

class DBManager:
    def __init__(self, db_path, auto_update_txt=True):
        self.db_path = Path(db_path)
        self.auto_update_txt = auto_update_txt

        self.txt_out_path = self.db_path.parent / (self.db_path.stem + '.txt')
        assert self.txt_out_path != self.db_path, "txt_out_path cannot be the same as db_path"
        assert os.path.exists(Path(self.db_path).parent), f'Path does not exist for file:\n\t{Path(self.db_path)}'
        # print(f'Opening database: {self.db_path}')
        self.con = sqlite3.connect(db_path)
        self.cur = self.con.cursor()
        # check and create tables
        if not self.cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='models'").fetchone():
            print("Creating table models")
            self.cur.execute("CREATE TABLE models(id INTEGER PRIMARY KEY, filename, info_json)")
            self.con.commit()
        if not self.cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone():
            print("Creating table metadata")
            self.cur.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value)")
            self.con.commit()
        
        self.export_if_required()

    def insert_model(self, filename: str, info_json: str, model_id=None):
        if model_id:
            next_id = int(model_id)
        else:
            next_id = int(self.get_max_id()) + 1
        filename = filename.format(next_id)
        self.cur.execute("INSERT INTO models VALUES (?, ?, ?)", (next_id, filename, info_json))
        self.con.commit()
        self.export_if_required()
        return filename
    
    def insert_update_metadata(self, key, value):
        if self.cur.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone():
            self.cur.execute("UPDATE metadata SET value=? WHERE key=?", (value, key))
        else:
            self.cur.execute("INSERT INTO metadata VALUES (?, ?)", (key, value))
        self.con.commit()
        self.export_if_required()

    def delete_model(self, id, delete_file=True):
        filepath = self.cur.execute("SELECT filename FROM models WHERE id=?", (id,)).fetchone()[0]
        self.cur.execute("DELETE FROM models WHERE id=?", (id,))
        if delete_file:
            os.remove(filepath)
        self.con.commit()
        self.export_if_required()

    def export_if_required(self):
        '''Should be called everytime the database is updated to keep the txt file up to date'''
        if self.auto_update_txt:
            self.export_db_to_csv() 

    def export_db_to_csv(self):
        df_meta = pd.read_sql_query("SELECT * FROM metadata", self.con).to_json(orient='values')  # metadata
        df = pd.read_sql_query("SELECT * FROM models", self.con)  # models
        table_tostring = 'metadata:\n' + df_meta + '\n\nmodels:\n'
        for row in df.iterrows():
            table_tostring += 'id: {0}, filename: {1}, info_json:{2}\n'.format(row[1]['id'], row[1]['filename'], row[1]['info_json'])
        with open(self.txt_out_path, 'w') as f:
            f.write(table_tostring)
        return table_tostring

    def get_max_id(self):
        res = self.cur.execute("SELECT MAX(id) FROM models").fetchone()[0]
        return res if res else 0

    def close(self):
        self.con.close()

    def _update_all_models(self, update_fn):
        '''update_fn takes in a dict of info_json and returns a dict of info_json'''
        df = pd.read_sql_query("SELECT * FROM models", self.con)
        for row in df.iterrows():
            info_json = update_fn(row[1]['info_json'])
            self.cur.execute("UPDATE models SET info_json=? WHERE id=?", (info_json, row[1]['id']))
        self.con.commit()
        self.export_if_required()
