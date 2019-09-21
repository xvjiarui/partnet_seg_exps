import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from eval_utils import eval_per_class_ap_ins, eval_per_shape_mean_ap_ins, eval_recall_iou_ins
import numpy as np
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, help='Category name [default: Chair]')
parser.add_argument('--level_id', type=int, help='Level ID [default: 3]')
parser.add_argument('--pred_dir', type=str, help='log prediction directory [default: log]')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU Threshold [default: 0.5]')
parser.add_argument('--plot_dir', type=str, default=None, help='PR Curve Plot Output Directory [default: None, meaning no output]')
FLAGS = parser.parse_args()

stat_in_fn = '../../stats/after_merging_label_ids/%s-level-%d.txt' % (FLAGS.category, FLAGS.level_id)

with open(stat_in_fn, 'r') as fin:
    part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
gt_in_dir = '../../data/ins_seg_h5_gt/%s-%d/' % (FLAGS.category, FLAGS.level_id)
pred_dir = FLAGS.pred_dir

recalls = eval_recall_iou_ins(stat_in_fn, gt_in_dir, pred_dir, iou_threshold=FLAGS.iou_threshold, plot_dir=FLAGS.plot_dir)
print(recalls)
print('mRecall %f'%np.mean(recalls))

# zip_path = '/'
# for i in FLAGS.plot_dir.split('/')[:-1]:
#     zip_path = os.path.join(zip_path, i)
# zip_path = os.path.join(zip_path,'visu.zip')
#
# if not os.path.exists(zip_path):
#     cmd = 'cd %s && cd .. && zip -r visu.zip visu'%(FLAGS.plot_dir)
#     call(cmd, shell=True)

