CUDA_VISIBLE_DEVICES=0 python valid_ins.py \
    --model model_ins \
    --category Chair Lamp StorageFurniture \
    --level_id 2 \
    --num_ins 200 \
    --log_dir log_finetune_model_ins_Chair_Lamp_StorageFurniture_2 \
    --valid_dir valid_ins \
    --num_point 10000 \
    --batch_size 1 \
    --margin_same 1.0
