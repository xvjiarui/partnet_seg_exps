CUDA_VISIBLE_DEVICES=2 python train_ins.py \
    --category Chair Lamp StorageFurniture \
    --level_id 1 \
    --model model_ins \
    --log_dir log_finetune_model_ins_Chair_Lamp_StorageFurniture_1 \
    --epoch 51 \
    --batch 1 \
    --point_num 10000 \
    --group_num 200 \
    --restore_dir log_pretrain_model_ins_Chair_Lamp_StorageFurniture_1 \
    --margin_same 1 \
    --margin_diff 2

