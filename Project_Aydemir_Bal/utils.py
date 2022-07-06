import random
import torch
import os
import logging
import numpy as np
from tqdm import tqdm

import torch.backends.cudnn as cudnn

from model import CRLC

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score


def save_model(args, epoch, optimizer, model):
    checkpoint = {"epoch": epoch,
                  "state_dict": model.state_dict(),
                  "optimizer": optimizer.state_dict()}
    model_save_path = os.path.join(args.model_save_path, "checkpoint.pt")
    torch.save(checkpoint, model_save_path)


def get_model(args):
    model = CRLC(args).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd,
                                nesterov=args.nesterov)
    epoch = 0
    if args.load_checkpoint:
        model_save_path = os.path.join(args.model_save_path, "checkpoint.pt")
        pretrained_dict = torch.load(model_save_path)
        model.load_state_dict(pretrained_dict["state_dict"])
        optimizer.load_state_dict(pretrained_dict["optimizer"])
        epoch = pretrained_dict["epoch"]

    return epoch, model, optimizer


def set_logger(log_path):

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path),
            TqdmHandler()
        ]
    )

    logger = logging.getLogger()
    return logger


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def fix_seed_for_reproducability(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


class AverageMeter(object):
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


# ======== Evaluation Metrics ========


def _make_cost_m(cm):
    s = np.max(cm)
    return (-cm + s)


def ACC(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    rows, columns = linear_assignment(_make_cost_m(cm))
    total = 0
    for i in range(len(rows)):
        row, column = rows[i], columns[i]
        value = cm[row][column]
        total += value
    return (total * 1. / np.sum(cm))


def NMI(labels, predictions):
    return normalized_mutual_info_score(labels, predictions)


def ARI(labels, predictions):
    return adjusted_rand_score(labels, predictions)
