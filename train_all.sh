echo " -------- Pairs for ./trained_models/SQL/ --------"

export TRAIN_STR="schema_1:1000,schema_2:1000,schema_3:1000,schema_union12:1000,schema_union13:1000,schema_union23:1000,schema_union123:1000"
export SAVE_PATH="./trained_models/SQL/M123E.pt"
./train_one.sh


export TRAIN_STR="schema_1:1000,schema_2:1000,schema_3:1000"
export SAVE_PATH="./trained_models/SQL/M123.pt"
./train_one.sh

export TRAIN_STR="schema_1:1000"
export SAVE_PATH="./trained_models/SQL/M1.pt"
./train_one.sh

export TRAIN_STR="schema_2:1000"
export SAVE_PATH="./trained_models/SQL/M2.pt"
./train_one.sh

export TRAIN_STR="schema_3:1000"
export SAVE_PATH="./trained_models/SQL/M3.pt"
./train_one.sh


echo " -------- Pairs for ./trained_models/6NF/ --------"

export TRAIN_STR="schemapseudo_1:1000"
export SAVE_PATH="./trained_models/6NF/M1.pt"
./train_one.sh

export TRAIN_STR="schemapseudo_2:1000"
export SAVE_PATH="./trained_models/6NF/M2.pt"
./train_one.sh

export TRAIN_STR="schemapseudo_3:1000"
export SAVE_PATH="./trained_models/6NF/M3.pt"
./train_one.sh


echo " -------- Pairs for ./trained_models/SQL_obf/ --------"

export TRAIN_STR="schema_1~mapping=1:1000,schema_2~mapping=1:1000,schema_3~mapping=1:1000,schema_union12~mapping=1:1000,schema_union13~mapping=1:1000,schema_union23~mapping=1:1000,schema_union123~mapping=1:1000"
export SAVE_PATH="./trained_models/SQL_obf/M123E.pt"
./train_one.sh

export TRAIN_STR="schema_1~mapping=1:1000,schema_2~mapping=1:1000,schema_3~mapping=1:1000"
export SAVE_PATH="./trained_models/SQL_obf/M123.pt"
./train_one.sh

export TRAIN_STR="schema_1~mapping=1:1000"
export SAVE_PATH="./trained_models/SQL_obf/M1.pt"
./train_one.sh

export TRAIN_STR="schema_2~mapping=1:1000"
export SAVE_PATH="./trained_models/SQL_obf/M2.pt"
./train_one.sh

export TRAIN_STR="schema_3~mapping=1:1000"
export SAVE_PATH="./trained_models/SQL_obf/M3.pt"
./train_one.sh


echo " -------- Pairs for ./trained_models/6NF_obf/ --------"

export TRAIN_STR="schemapseudo_1~mapping=1:1000"
export SAVE_PATH="./trained_models/6NF_obf/M1.pt"
./train_one.sh

export TRAIN_STR="schemapseudo_2~mapping=1:1000"
export SAVE_PATH="./trained_models/6NF_obf/M2.pt"
./train_one.sh

export TRAIN_STR="schemapseudo_3~mapping=1:1000"
export SAVE_PATH="./trained_models/6NF_obf/M3.pt"
./train_one.sh
