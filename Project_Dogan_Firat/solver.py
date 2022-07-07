from torchutils.metrics.tracker import AverageMeterFunction
from torchmetrics.functional import accuracy
from torchutils.logging import WandBLogger, ProgressBarLogger
from torchutils.callbacks import ScoreLoggerCallback
from trainer import UMGCN, UMGCNTrainer
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from models import GraphConv, FullyConnected
from loss import LabelSmoothingLoss, Wasserstein
import torch


class TakeInterpreter(Exception):
    pass


dataset = Planetoid(
    './data',
    'CiteSeer',
    transform=T.NormalizeFeatures()
)


# generate random fake samples
citeseer = dataset[0]
fake_edges = torch.randint(0, citeseer.num_nodes, [
    2, int(citeseer.num_edges * 1.0)])

dataset[0].edge_index = torch.cat([citeseer.edge_index, fake_edges], dim=1)

experiment = 'dropout=0.6,random_noise=1.0'
gcn_model = GraphConv(
    in_channels=dataset.num_features,
    hidden_channels=149,
    out_channels=dataset.num_classes,
    use_gdc=False,
    dropout=0.6,
    inference=0.2,
)

fc_model = FullyConnected(
    in_channels=dataset.num_features,
    hidden_channels=1200,
    out_channels=dataset.num_classes
)

adam = optim.Adam([
    dict(params=gcn_model.parameters(), lr=0.01, weight_decay=5e-4),
    dict(params=fc_model.parameters(), lr=1e-4)
])

umgcn = UMGCN(
    model=gcn_model,
    surrogate=fc_model,
    optimizer=adam,
    criterion=Wasserstein,
    label_smoothing=LabelSmoothingLoss(num_classes=dataset.num_classes),
    l_um=0.3,
    l_ls=0.001,
)

trainer = UMGCNTrainer(
    model=umgcn,
    device='cpu',
    ytype=umgcn.dtype
)


# @TODO: allow metric handler to compute a functional at only requested events occured
# @TODO: remove getLoggerGroup() calls to make it smooth instead create a function or a
# class derived from LoggerGroup, ProgressBarLoggerGroup and also UnionGroups
# @TODO: add AverageMeterFunction and AverageMeterModule to __init__ under /metrics
# @TODO: rename AverageMeterModule_Base to AverageMeterModule


def argmax_accuracy(preds, target):
    return accuracy(preds.argmax(dim=1), target)


trainer.compile_handlers(
    callbacks=[ScoreLoggerCallback(
        'Loss', 'Accuracy', 'Cross Entropy', 'Uncertainty', 'Label Smoothing')],
    loggers=ProgressBarLogger.getLoggerGroup(),
    metrics=[AverageMeterFunction(name='Accuracy', fn=argmax_accuracy)]
)
trainer.compile_handlers(
    loggers=WandBLogger.getLoggerGroup(
        username='adnanhd',
        experiment=experiment,
        project='um-gcn'
    )
)

trainer.train(
    num_epochs=200,
    batch_size=1,
    train_dataset=dataset,
    history={'Loss'},
)
