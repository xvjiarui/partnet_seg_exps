import os
from multiprocessing import Process
gt_dir = '../../data/ins_seg_h5_gt'

category_list = []
for gt_folder in os.listdir(gt_dir):
    category, _ = gt_folder.split('-')
    category_list.append(category)
category_list = list(set(category_list))

def eval(category, level_id):
    cmd = 'CUDA_VISIBLE_DEVICES=0 ' \
          'python eval_ins.py  ' \
          '--model model_ins  ' \
          '--category {}  ' \
          '--level_id 1  ' \
          '--num_ins 200  ' \
          '--log_dir  log_finetune_model_ins_Chair_Lamp_StorageFurniture_{}  ' \
          '--valid_dir valid_ins  ' \
          '--eval_dir eval_{}  ' \
          '--pred_dir pred  ' \
          '--visu_dir visu ' \
          '--visu_batch 16 ' \
          '--num_point 10000 ' \
          '--batch_size 1 ' \
          '--margin_same 1.0'
    cmd = cmd.format(category, level_id, category.lower())
    os.system(cmd)


for category in category_list:
    print category
    pool = []
    for level_id in [1, 2, 3]:
        pool.append(Process(target=eval, args=(category, level_id)))
    for p in pool:
        p.start()
    for p in pool:
        p.join()
