echo "Training model $SAVE_PATH with $TRAIN_STR"
python train.py \
    --dataset_args \
        train_str=${TRAIN_STR} \
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
