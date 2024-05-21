# Installatioin

## prerequisite

First you need to obtain the hugging-face llama 2 model from (request access from `https://llama.meta.com/llama2/`) and place it in a folder `./models/llama2/llama_hf_converted/7b` along with the tokenizer `./models/llama2/llama/tokenizer.model`

Clone the repo into a folder `./SecureLLM`

Then create a conda environment `conda env create -f ./environment.yml`

Then activate the conda environment `conda activate securellm`

Finally make sure all `.sh` files are executable. Go to `./SecureLLM` and run `chmod +x *.sh`

# Results

To get the results for all columns of all 3 tables run the bash script `run_all.sh` inside `./SecureLLM`

To print all the results into tables (once `run_all.sh` finishes) run `python print_tables.py`

For individual tables run the following:

For Table 1 results:

first two columns:
`python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M123E  > T1.C1.log`
`python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M123  > T1.C2.log`
column 3-7:
`python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraLoraHub  >  T1.C3.log`
`python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraSum  >  T1.C4.log`
`python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraMax  >  T1.C5.log`
`python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraLogit  >  T1.C6.log`
`python experiments.py --device 0 --sample_size 120 --mroot "./trained_models/6NF/" --pseudo --models M1 M2 M3 --config SloraLogit  >  T1.C7.log`

For Table 2 results:

first two columns:
`python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M123E  > T2.C1.log`
`python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M123  > T2.C2.log`
column 3-7:
`python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraLoraHub  >  T2.C3.log`
`python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraSum  >  T2.C4.log`
`python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraMax  >  T2.C5.log`
`python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/SQL/" --models M1 M2 M3 --config SloraLogit  >  T2.C6.log`
`python experiments.py --device 0 --sample_size 120 --gpt --mroot "./trained_models/6NF/" --pseudo --models M1 M2 M3 --config SloraLogit  >  T2.C7.log`

For Table 3 results:

first two columns:
`python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M123E  > T3.C1.log`
`python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M123  > T3.C2.log`
column 3-7:
`python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M1 M2 M3 --config SloraLoraHub  >  T3.C3.log`
`python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M1 M2 M3 --config SloraSum  >  T3.C4.log`
`python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M1 M2 M3 --config SloraMax  >  T3.C5.log`
`python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/SQL_obf/" --models M1 M2 M3 --config SloraLogit  >  T3.C6.log`
`python experiments.py --device 0 --sample_size 120 --mapping 1 --mroot "./trained_models/6NF_obf/" --pseudo --models M1 M2 M3 --config SloraLogit  >  T3.C7.log`


# Train

## Train one model

To train a model, run the following command inside `./SecureLLM` (update the `TRAIN_STR` and `SAVE_PATH` depending on which model you want to train).

```
export TRAIN_STR="schema_1:1000,schema_2:1000,schema_3:1000"
export SAVE_PATH="./trained_models/SQL/M123.pt"
python train.py \
    --dataset_args \
        train_str=${TRAIN_STR} \
        val_str="schema_1_val:100,schema_2_val:100,schema_3_val:100" \
        max_inp_matrix_size=800 \
    --model_args \
        name=llama-2-7b \
        max_seq_len=1024 \
        max_batch_size=32 \
        type=lora \
        lora_r=8 \
        lora_alpha=32 \
        lora_dropout=0.1 \
    --train_args \
        epochs=1 \
        lr=2e-4 \
        weight_decay=0.002 \
    --save_path=${SAVE_PATH} \
    --experiment "0.0" \
    --world_size 1 \
    --seed 2 \
    --device cuda:0
```

Note: the `val_str` line can be deleted which will make the command finish faster with no affect on the saved model.

The general structure for the train_str is simple, schema_1~mapping=1 for column obfuscation, schemapseudo_1 for pseudocode, schemapseudo_1~mapping=1 for both. Look at `train_all.sh` for all combinations.
## Train all models

The file `train_all.sh` contains and will train all pairs that are needed to reproduce all models used.
