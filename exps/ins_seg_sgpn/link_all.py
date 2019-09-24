import os
import os.path as osp
gt_dir = '../../data/ins_seg_h5_gt'

category_list = []
for gt_folder in os.listdir(gt_dir):
    category, _ = gt_folder.split('-')
    category_list.append(category)
category_list = list(set(category_list))

log_dir_template = 'log_finetune_model_ins_Chair_Lamp_StorageFurniture_{}'

log_dir = 'fusion/pred'

for category in category_list:
    print category
    cur_folder = osp.join(log_dir, category)
    if not osp.exists(cur_folder):
        os.makedirs(cur_folder)
    for level in [1, 2, 3]:
        src_folder = osp.join(log_dir_template.format(level), 'eval_{}'.format(category.lower()), 'pred')
        dst_folder = osp.join(category, 'Level-{}'.format(level))
        print('linking {} to {}'.format(src_folder, dst_folder))
