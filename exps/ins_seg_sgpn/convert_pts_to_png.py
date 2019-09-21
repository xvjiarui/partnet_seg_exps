import time
import argparse
import os
import sys
import pptk
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
from commons import check_mkdir, force_mkdir

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='log', help='Log dir [default: log]')
parser.add_argument('--eval_dir', type=str, default='eval', help='Eval dir [default: eval]')
parser.add_argument('--visu_dir', type=str, default=None, help='Visu dir [default: None, meaning no visu]')
FLAGS = parser.parse_args()

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    print('ERROR: log_dir %s does not exist! Please Check!' % LOG_DIR)
    exit(1)
LOG_DIR = os.path.join(LOG_DIR, FLAGS.eval_dir)
if FLAGS.visu_dir is not None:
    VISU_DIR = os.path.join(LOG_DIR, FLAGS.visu_dir)
    force_mkdir(VISU_DIR)

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    palette = np.array(palette).reshape(3, -1).transpose()
    return palette

def convert(visu_dir):
    for root, dirs, files in os.walk(visu_dir):
        for file in files:
            if file.endswith('.pts'):
                pts_file = os.path.join(root, file)
                label_file = pts_file.replace('.pts', '.label')
                out_file = pts_file.replace('.pts', '.png')
                print('rendering: {}'.format(pts_file))
                with open(pts_file) as f:
                    pts = np.loadtxt(f)
                if os.path.exists(label_file):
                    print('rendering: {}'.format(label_file))
                    with open(label_file) as f:
                        label = np.loadtxt(f, dtype=np.bool)
                else:
                    label = None
                if label is not None:
                    pts = pts[label]
                pts = np.stack([pts[:, 2], pts[:, 0], pts[:, 1]], axis=1)
                v = pptk.viewer(pts)
                v.set(point_size=0.01, r=5, show_grid=False, show_axis=False, lookat=[.8, .8, .8])
                v.capture(out_file)
                time.sleep(0.5)
                print('saving: {}'.format(out_file))
                # print('camera LA:', v.get('lookat'))
                mask_file = pts_file.replace('.pts', '.mask')
                if os.path.exists(mask_file):
                    with open(mask_file) as f:
                        mask = np.loadtxt(f).astype(np.int)
                else:
                    mask = None
                v.close()
                if mask is not None:
                    palette = get_palette(len(np.unique(mask)))
                    v = pptk.viewer(pts, palette[mask, :])
                    v.set(point_size=0.01, r=5, show_grid=False, show_axis=False, lookat=[.8, .8, .8])
                    mask_out_file = out_file.replace('pts', 'mask')
                    if not os.path.exists(os.path.dirname(mask_out_file)):
                        os.mkdir(os.path.dirname(mask_out_file))
                    v.capture(mask_out_file)
                    time.sleep(0.5)
                    print('saving: {}'.format(mask_out_file))
                    v.close()

convert(VISU_DIR)