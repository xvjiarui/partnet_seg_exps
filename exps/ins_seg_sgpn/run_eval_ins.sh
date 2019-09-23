CUDA_VISIBLE_DEVICES=0 python eval_ins.py \
    --model model_ins \
    --category Lamp \
    --level_id 3 \
    --num_ins 200 \
    --log_dir \
    log_finetune_model_ins_Chair_Lamp_StorageFurniture_1 \
    log_finetune_model_ins_Chair_Lamp_StorageFurniture_2 \
    log_finetune_model_ins_Chair_Lamp_StorageFurniture_3 \
    --valid_dir valid_ins \
    --eval_dir eval_lamp \
    --pred_dir pred \
    --visu_dir visu \
    --visu_batch 16 \
    --num_point 10000 \
    --batch_size 1 \
    --margin_same 1.0
