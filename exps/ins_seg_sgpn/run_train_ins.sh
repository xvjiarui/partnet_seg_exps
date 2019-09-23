CUDA_VISIBLE_DEVICES=2 python train_ins.py \
    --category Chair \
    --level_id 3 \
    --model model_ins \
    --log_dir log_finetune_model_ins_Chair_3 \
    --epoch 51 \
    --batch 1 \
    --point_num 10000 \
    --group_num 200 \
    --restore_dir log_pretrain_model_ins_Chair_3 \
    --margin_same 1 \
    --margin_diff 2

