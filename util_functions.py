import io
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from imageio import imread

import logging

import torch

logger = logging.getLogger(__name__)
plt.ioff()

class Averaging(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def timing(f):

    """Wrapper for functions to measure time"""

    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        logger.info(
            "%s took %0.3f ms ~ %0.3f min ~ %0.3f sec"
            % (f, (time2 - time1) * 1000.0, (time2 - time1) / 60.0, (time2 - time1))
        )
        return ret

    return wrap


def compute_euclidean_dist(embeddings, prototypes):
    with torch.no_grad():
        dists = (
            torch.sum(embeddings ** 2, dim=1).view(-1, 1)
            + torch.sum(prototypes ** 2, dim=1).view(1, -1)
            - 2 * torch.matmul(embeddings, torch.transpose(prototypes, 0, 1))
        )
        return dists


def get_project_root():
    return Path(__file__).parent.parent.parent


def parse_return_stat(stat, out_file=None):
    keys = ["mof", "mof_bg", "f1", "mean_f1"]
    labels = []
    values = []

    if out_file:
        with open(out_file, "w+") as f:
            for key in keys:
                if key == "f1":
                    _eps = 1e-8
                    n_tr_seg, n_seg = stat["precision"]
                    precision = n_tr_seg / n_seg
                    _, n_tr_seg = stat["recall"]
                    recall = n_tr_seg / n_tr_seg
                    val = 2 * (precision * recall) / (precision + recall + _eps)
                else:
                    v1, v2 = stat[key]
                    if key == "iou_bg":
                        v2 += 1  # bg class
                    val = v1 / v2

                labels.append(key)
                values.append(val)

                logger.info("%s: %f" % (key, val))
                f.write("%s: %f\n" % (key, val))
    else:
        for key in keys:
            if key == "f1":
                _eps = 1e-8
                n_tr_seg, n_seg = stat["precision"]
                precision = n_tr_seg / n_seg
                _, n_tr_seg = stat["recall"]
                recall = n_tr_seg / n_tr_seg
                val = 2 * (precision * recall) / (precision + recall + _eps)
            else:
                v1, v2 = stat[key]
                if key == "iou_bg":
                    v2 += 1  # bg class
                val = v1 / v2

            labels.append(key)
            values.append(val)

            logger.info("%s: %f" % (key, val))

    return labels, values


def plot_to_image(figure):

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = imread(buf)

    return image


def plot_confusion_matrix(q, path):

    fig = plt.figure()
    plot = sns.heatmap(q)
    fig.savefig(path, transparent=False)
    # image = plot_to_image(fig)

    return fig

def bounds(segm):
    start_label = segm[0]
    
    start_idx = 0
    idx = 0
    while idx < len(segm):
        try:
            while start_label == segm[idx]:
                idx += 1
        except IndexError:
            yield start_idx, idx, start_label
            break

        yield start_idx, idx, start_label
        start_idx = idx
        start_label = segm[start_idx]

def plot_segm(path, segmentation, colors, name=''):
    # mpl.style.use('classic')
    fig = plt.figure(figsize=(16, 2))
    plt.axis('off')
    plt.title(name, fontsize=20)
    # plt.subplots_adjust(top=0.9, hspace=0.6)
    gt_segm = segmentation['gt'][0]
    ax_idx = 1
    plots_number = len(segmentation)
    ax = fig.add_subplot(plots_number, 1, ax_idx)
    ax.set_ylabel('GT', fontsize=30, rotation=0, labelpad=40, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # make_axes_area_auto_adjustable(ax)
    # plt.title('gt', fontsize=20)
    v_len = len(gt_segm)
   
    for start, end, label in bounds(gt_segm):
        label= label.numpy()
        
        fc= colors[int(label)]
        ax.axvspan(start / v_len, end / v_len, facecolor=fc, alpha=1.0)
    #print([(k,v) for k,v in segmentation.items()])

    for key,v in segmentation.items():
        if key in ['gt']:
            continue
        segm, label2gt =v
        #print(label2gt)
        ax_idx += 1
        ax = fig.add_subplot(plots_number, 1, ax_idx)
        ax.set_ylabel('OUTPUT', fontsize=30, rotation=0, labelpad=60, verticalalignment='center')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # make_axes_area_auto_adjustable(ax)
        # print(segm)
        # print(label2gt)
        #print("segm", segm)
        #print("label2gt", label2gt)
        # for i in range(len(segm)): 
        #     segm[i]= label2gt[segm[i]]
        segm = list(map(lambda x: label2gt[int(x)], segm))
        segm = [segm[i] if gt_segm[i]!=-1 else -1 for i in range(len(segm))]
        
        #segm = list(map(lambda x: label2gt[x], segm))
        # print(segm)
        for start, end, label in bounds(segm):
            ax.axvspan(start / v_len, end / v_len, facecolor=colors[label], alpha=1.0)


    fig.savefig(path, transparent=False)

if __name__ == "__main__":
    print(get_project_root())