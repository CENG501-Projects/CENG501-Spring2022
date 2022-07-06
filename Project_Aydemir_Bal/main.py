import torch
from tqdm import tqdm
import datetime
import os
import numpy as np

from utils import set_logger, fix_seed_for_reproducability, AverageMeter, ACC, NMI, ARI
from dataset import get_dataset
# model itself
from utils import get_model, save_model
# criterions
from model import PC_loss, Entropy, FC_loss


def train_epoch(args, dataloader, model, optimizer):
    losses = AverageMeter()

    fc_criterion = FC_loss(args)
    pc_criterion = PC_loss(args)
    h_criterion = Entropy(args)

    model.train()
    args.logger.info("Training Started")
    for i, ((inp1, inp2), _) in enumerate(tqdm(dataloader, desc="Training")):
        inp1 = inp1.to(args.device)
        inp2 = inp2.to(args.device)

        rl_i, rl_j, q = model(inp1, inp2)

        l_fc = fc_criterion(rl_i, rl_j)
        l_pc = pc_criterion(q)
        h = h_criterion(q)

        loss = l_pc - args.lambda1*h + args.lambda2*l_fc

        losses.update(loss.item(), inp1.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def eval(args, dataloader, model):
    # ACCMetric = AverageMeter()
    # NMIMetric = AverageMeter()
    # ARIMetric = AverageMeter()

    labels = np.zeros((len(dataloader.dataset), ))
    predictions = np.zeros((len(dataloader.dataset), ))
    model.eval()
    args.logger.info("Evaluation Started")
    with torch.no_grad():
        for i, (inp, label) in enumerate(tqdm(dataloader, desc="Testing")):
            data = inp.to(args.device)
            q = model(data, None)

            batch_size = data.shape[0]
            prediction = q.argmax(dim=-1).cpu().numpy()

            labels[i*args.batch_size: i*args.batch_size
                   + batch_size] = label.cpu().numpy()
            predictions[i*args.batch_size: i
                        * args.batch_size + batch_size] = prediction

        ACCMetric = ACC(labels, predictions)
        NMIMetric = NMI(labels, predictions)
        ARIMetric = ARI(labels, predictions)

    return ACCMetric, NMIMetric, ARIMetric


def main(args):
    now = datetime.datetime.now()
    file_name = f"train_{now.hour}:{now.minute}_{now.day}-{now.month}-{now.year}.log"
    args.logger = set_logger(os.path.join(args.log_file_path, file_name))
    args.print_args()
    # args.print_args()
    fix_seed_for_reproducability(0)

    # TODO: - Set dataset (DONE)
    #       - Set model (DONE)
    #       - Set optimizer (DONE)
    #       - Finish loss functions (DONE)
    #       - Main training loop (DONE)
    #       - Model checkpoint/loading (DONE)
    #       - Evaluation (DONE)
    #       - Training/evaluation integration (DONE)
    #       - Other dataset and backbone settings
    #       - Distributed training

    train_dataloader, test_dataloader = get_dataset(args)

    start_epoch, model, optimizer = get_model(args)
    for epoch in range(start_epoch, args.epochs):
        args.logger.info(f"=========== [Epoch {epoch}] ===========")

        if epoch % 5 == 0:
            avg_acc, avg_nmi, avg_ari = eval(args, test_dataloader, model)
            args.logger.info(f"Average Accuracy: {avg_acc:.4f}")
            args.logger.info(f"Average NMI: {avg_nmi:.4f}")
            args.logger.info(f"Average ARI: {avg_ari:.4f}\n")

        avg_loss = train_epoch(args, train_dataloader, model, optimizer)
        args.logger.info(f"Average Loss: {avg_loss:.4f}")
        args.logger.info(f"=========== =========== ===========\n\n")
        save_model(args, epoch, optimizer, model)
