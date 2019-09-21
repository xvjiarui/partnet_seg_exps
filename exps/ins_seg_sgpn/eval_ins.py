import argparse
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
import tf_util
import json
from commons import check_mkdir, force_mkdir
from geometry_utils import *
from progressbar import ProgressBar
from subprocess import call
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', type=str, default='model', help='Model name [default: model]')
parser.add_argument('--category', type=str, default='Chair', help='Category name [default: Chair]')
parser.add_argument('--level_id', type=int, default='3', help='Level ID [default: 3]')
parser.add_argument('--num_ins', type=int, default='200', help='Max Number of Instance [default: 200]')
parser.add_argument('--log_dir', type=str, default='log', help='Log dir [default: log]')
parser.add_argument('--valid_dir', type=str, default='valid', help='Valid dir [default: valid]')
parser.add_argument('--eval_dir', type=str, default='eval', help='Eval dir [default: eval]')
parser.add_argument('--pred_dir', type=str, default='pred', help='Pred dir [default: pred]')
parser.add_argument('--visu_dir', type=str, default=None, help='Visu dir [default: None, meaning no visu]')
parser.add_argument('--visu_batch', type=int, default=16, help='visu batch [default: 16]')
parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 10000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--margin_same', type=float, default=1.0, help='Double hinge loss margin: same semantic [default: 1]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
CKPT_DIR = os.path.join(LOG_DIR, 'trained_models')
if not os.path.exists(LOG_DIR):
    print('ERROR: log_dir %s does not exist! Please Check!' % LOG_DIR)
    exit(1)
VALID_DIR = os.path.join(LOG_DIR, FLAGS.valid_dir)
if not os.path.exists(VALID_DIR):
    print('ERROR: valid_dir %s does not exist! Run valid.py first!' % VALID_DIR)
    exit(1)
LOG_DIR = os.path.join(LOG_DIR, FLAGS.eval_dir)
check_mkdir(LOG_DIR)
PRED_DIR = os.path.join(LOG_DIR, FLAGS.pred_dir)
force_mkdir(PRED_DIR)
if FLAGS.visu_dir is not None:
    VISU_DIR = os.path.join(LOG_DIR, FLAGS.visu_dir)
    force_mkdir(VISU_DIR)

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_eval.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# load meta data files
stat_in_fn = '../../stats/after_merging_label_ids/%s-level-%d.txt' % (FLAGS.category, FLAGS.level_id)
print('Reading from ', stat_in_fn)
with open(stat_in_fn, 'r') as fin:
    part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
print('Part Name List: ', part_name_list)
data_in_dir = '../../data/ins_seg_h5_for_sgpn/%s-%d/' % (FLAGS.category, FLAGS.level_id)
test_h5_fn_list = []
for item in os.listdir(data_in_dir):
    if item.endswith('.h5') and item.startswith('test-'):
        test_h5_fn_list.append(item)

NUM_CLASSES = len(part_name_list)
print('Semantic Labels: ', NUM_CLASSES)
NUM_CLASSES = 1
print('force Semantic Labels: ', NUM_CLASSES)
NUM_INS = FLAGS.num_ins
print('Number of Instances: ', NUM_INS)

# load validiation hyper-parameters
pw_sim_thres = np.loadtxt(os.path.join(VALID_DIR, 'per_category_pointwise_similarity_threshold.txt')).reshape(NUM_CLASSES)
avg_group_size = np.loadtxt(os.path.join(VALID_DIR, 'per_category_average_group_size.txt')).reshape(NUM_CLASSES)
min_group_size = 0.25 * avg_group_size

for i in range(NUM_CLASSES):
    print('%d %s %f %d' % (i, part_name_list[i], pw_sim_thres[i], min_group_size[i]))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def load_h5(fn):
    with h5py.File(fn, 'r') as fin:
        pts = fin['pts'][:, :NUM_POINT, :]
        return pts

def load_json(fn):
    with open(fn, 'r') as fin:
        return json.load(fin)

def load_data(fn):
    cur_json_fn = fn.replace('.h5', '.json')
    record = load_json(cur_json_fn)
    pts = load_h5(fn)
    return pts, record

def save_h5(fn, mask, valid, conf):
    fout = h5py.File(fn, 'w')
    fout.create_dataset('mask', data=mask, compression='gzip', compression_opts=4, dtype='bool')
    fout.create_dataset('valid', data=valid, compression='gzip', compression_opts=4, dtype='bool')
    fout.create_dataset('conf', data=conf, compression='gzip', compression_opts=4, dtype='float32')
    fout.close()

# Adapted from https://github.com/laughtervv/SGPN/blob/master/utils/test_utils.py#L95-L163
def GroupMerging(pts_corr, confidence, label_bin):
    seg = np.zeros_like(confidence).astype(np.uint)
    confvalidpts = (confidence>0.4)
    un_seg = np.unique(seg)
    refineseg = -1* np.ones(pts_corr.shape[0])
    groupid = -1* np.ones(pts_corr.shape[0])
    numgroups = 0
    groupseg = {};
    groupconf = {};
    for i_seg in un_seg:
        pts_in_seg = (seg==i_seg)
        valid_seg_group = np.where(pts_in_seg & confvalidpts)
        proposals = []
        if valid_seg_group[0].shape[0]==0:
            proposals += [pts_in_seg]
        else:
            for ip in valid_seg_group[0]:
                validpt = (pts_corr[ip] < label_bin[i_seg]) & pts_in_seg
                if np.sum(validpt)>5:
                    flag = False
                    for gp in range(len(proposals)):
                        iou = float(np.sum(validpt & proposals[gp])) / np.sum(validpt|proposals[gp])#uniou
                        validpt_in_gp = float(np.sum(validpt & proposals[gp])) / np.sum(validpt)#uniou
                        if iou > 0.6 or validpt_in_gp > 0.8:
                            flag = True
                            if np.sum(validpt)>np.sum(proposals[gp]):
                                proposals[gp] = validpt
                            continue

                    if not flag:
                        proposals += [validpt]

            if len(proposals) == 0:
                proposals += [pts_in_seg]
        for gp in range(len(proposals)):
            if np.sum(proposals[gp])>50:
                groupid[proposals[gp]] = numgroups
                groupseg[numgroups] = i_seg
                groupconf[numgroups] = np.sum(confidence[proposals[gp]]) / np.sum(proposals[gp])
                numgroups += 1
                refineseg[proposals[gp]] = stats.mode(seg[proposals[gp]])[0]

    un, cnt = np.unique(groupid, return_counts=True)
    for ig, g in enumerate(un):
        if cnt[ig] < 50:
            groupid[groupid==g] = -1

    un, cnt = np.unique(groupid, return_counts=True)
    groupidnew = groupid.copy()
    newgroupseg = {}
    newgroupconf = {}
    for ig, g in enumerate(un):
        if g == -1:
            continue
        groupidnew[groupid==g] = (ig-1)
        newgroupseg[(ig-1)] = groupseg.pop(g)
        newgroupconf[(ig-1)] = groupconf.pop(g)
    groupid = groupidnew
    groupseg = newgroupseg
    groupconf = newgroupconf

    for ip, gid in enumerate(groupid):
        if gid == -1:
            pts_in_gp_ind = (pts_corr[ip] < label_bin[seg[ip]])
            pts_in_gp = groupid[pts_in_gp_ind]
            pts_in_gp_valid = pts_in_gp[pts_in_gp!=-1]
            if len(pts_in_gp_valid) != 0:
                groupid[ip] = stats.mode(pts_in_gp_valid)[0][0]

    return groupid, refineseg, groupseg, groupconf

def render_pts_pptk(out, pts, delete_img=False, point_size=6, point_color='FF0000FF'):
    tmp_pts = out.replace('.png', '.pts')
    export_pts(tmp_pts, pts)

def render_pts_with_label_pptk(out, pts, label, delete_img=False, base=0, point_size=6):
    tmp_pts = out.replace('.png', '.pts')
    tmp_label = out.replace('.png', '.label')

    label += base

    export_pts(tmp_pts, pts)
    export_label(tmp_label, label)

def render_pts_with_mask_pptk(out, pts, mask, delete_img=False, base=0, point_size=6):
    tmp_pts = out.replace('.png', '.pts')
    tmp_label = out.replace('.png', '.mask')

    export_pts(tmp_pts, pts)
    export_label(tmp_label, mask)

def gen_visu(base_idx, pts, mask, valid, conf, record, num_pts_to_visu=1000):
    n_shape = pts.shape[0]
    n_ins = mask.shape[1]
    
    pts_dir = os.path.join(VISU_DIR, 'pts')
    info_dir = os.path.join(VISU_DIR, 'info')
    child_dir = os.path.join(VISU_DIR, 'child')

    if base_idx == 0:
        os.mkdir(pts_dir)
        os.mkdir(info_dir)
        os.mkdir(child_dir)

    for i in range(n_shape):
        cur_pts = pts[i, ...]
        cur_mask = mask[i, ...]
        cur_valid = valid[i, :]
        cur_conf = conf[i, :]
        cur_record = record[i]

        cur_idx_to_visu = np.arange(NUM_POINT)
        np.random.shuffle(cur_idx_to_visu)
        cur_idx_to_visu = cur_idx_to_visu[:num_pts_to_visu]

        cur_shape_prefix = 'shape-%03d' % (base_idx + i)
        out_fn = os.path.join(pts_dir, cur_shape_prefix+'.png')
        render_pts_with_mask_pptk(out_fn, cur_pts[cur_idx_to_visu], cur_mask[cur_valid, :].argmax(0)[cur_idx_to_visu])
        out_fn = os.path.join(info_dir, cur_shape_prefix+'.txt')
        with open(out_fn, 'w') as fout:
            fout.write('model_id: %s, anno_id: %s\n' % (cur_record['model_id'], cur_record['anno_id']))
        
        cur_child_dir = os.path.join(child_dir, cur_shape_prefix)
        os.mkdir(cur_child_dir)
        child_pred_dir = os.path.join(cur_child_dir, 'pred')
        os.mkdir(child_pred_dir)
        child_info_dir = os.path.join(cur_child_dir, 'info')
        os.mkdir(child_info_dir)

        cur_conf[~cur_valid] = 0.0
        idx = np.argsort(-cur_conf)
        for j in range(n_ins):
            cur_idx = idx[j]
            if cur_valid[cur_idx]:
                cur_part_prefix = 'part-%03d' % j
                out_fn = os.path.join(child_pred_dir, cur_part_prefix+'.png')
                render_pts_with_label_pptk(out_fn, cur_pts[cur_idx_to_visu], cur_mask[cur_idx, cur_idx_to_visu].astype(np.int32))
                out_fn = os.path.join(child_info_dir, cur_part_prefix+'.txt')
                with open(out_fn, 'w') as fout:
                    fout.write('part idx: %d\n' % cur_idx)
                    fout.write('score: %f\n' % cur_conf[cur_idx])
                    fout.write('#pts: %d\n' % np.sum(cur_mask[cur_idx, :]))


def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_ph, _, _, _, _, _ = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_INS, NUM_CLASSES)   # B x N x 3
            is_training_ph = tf.placeholder(tf.bool, shape=())

            net_output = MODEL.get_model(pointclouds_ph, NUM_CLASSES, FLAGS.margin_same, is_training_ph)

            simmat_pred = net_output['simmat']                                      # B x N x N
            conf_pred = tf.reshape(net_output['conf'], [BATCH_SIZE, NUM_POINT])     # B x N
          
            loader = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Load pretrained model
        ckptstate = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckptstate is not None:
            LOAD_MODEL_FILE = os.path.join(CKPT_DIR, os.path.basename(ckptstate.model_checkpoint_path))
            loader.restore(sess, LOAD_MODEL_FILE)
            log_string("Model loaded in file: %s" % LOAD_MODEL_FILE)
        else:
            log_string("Fail to load modelfile: %s" % CKPT_DIR)

        # visu
        if FLAGS.visu_dir is not None:
            cur_visu_batch = 0

        # Start testing
        batch_pts = np.zeros((BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)

        for item in test_h5_fn_list:
            cur_h5_fn = os.path.join(data_in_dir, item)
            print('Reading data from ', cur_h5_fn)
            pts, record = load_data(cur_h5_fn)

            n_shape = pts.shape[0]
            num_batch = int((n_shape - 1) * 1.0 / BATCH_SIZE) + 1
            
            out_mask = np.zeros((n_shape, NUM_INS, NUM_POINT), dtype=np.bool)
            out_valid = np.zeros((n_shape, NUM_INS), dtype=np.bool)
            out_conf = np.zeros((n_shape, NUM_INS), dtype=np.float32)
            out_label = np.zeros((n_shape, NUM_INS), dtype=np.uint8)
            
            bar = ProgressBar()
            for i in bar(range(num_batch)):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, n_shape)

                batch_pts[:end_idx-start_idx, ...] = pts[start_idx: end_idx, ...]
                batch_record = record[start_idx: end_idx]

                feed_dict = {pointclouds_ph: batch_pts,
                             is_training_ph: False}

                simmat_pred_val, conf_pred_val = sess.run([simmat_pred, conf_pred], feed_dict=feed_dict)

                simmat_pred_val = simmat_pred_val[:end_idx-start_idx, ...]  # B x N x N
                conf_pred_val = conf_pred_val[:end_idx-start_idx, ...]      # B x N

                for j in range(end_idx - start_idx):
                    cur_simmat = simmat_pred_val[j, ...]
                    cur_conf = conf_pred_val[j, :]

                    group_ids, _, group_seg, group_conf = GroupMerging(cur_simmat, cur_conf, pw_sim_thres)

                    sid = start_idx + j
                    for gid in group_seg.keys():
                        if np.sum(group_ids == gid) >= min_group_size[group_seg[gid]]:
                            out_mask[sid, gid, :] = (group_ids == gid)
                            out_valid[sid, gid] = True
                            out_conf[sid, gid] = group_conf[gid]

                if FLAGS.visu_dir is not None and cur_visu_batch < FLAGS.visu_batch:
                    gen_visu(start_idx, batch_pts[:end_idx-start_idx, ...], out_mask[start_idx: end_idx, ...], out_valid[start_idx: end_idx, :], \
                            out_conf[start_idx: end_idx, :], batch_record)
                    cur_visu_batch += 1

            save_h5(os.path.join(PRED_DIR, item), out_mask, out_valid, out_conf)

        if FLAGS.visu_dir is not None:
            cmd = 'cd %s && python %s . 1 htmls pts,info:pred,info pts,info:pred,info' % (VISU_DIR, os.path.join(ROOT_DIR, '../utils/gen_html_hierachy_local.py'))
            log_string(cmd)
            call(cmd, shell=True)

# main
log_string('pid: %s'%(str(os.getpid())))
eval()
LOG_FOUT.close()

