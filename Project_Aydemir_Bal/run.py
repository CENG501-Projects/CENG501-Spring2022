import argparse
import torch.cuda as cuda
from main import main

parser = argparse.ArgumentParser("CRLC")

parser.add_argument('--dataset', type=str, default="CIFAR10")
parser.add_argument('--data_path', type=str,
                    default="/userfiles/hpc-gaydemir/cifar")

parser.add_argument('--model', type=str, default="resnet34")

parser.add_argument('--optimizer', type=str, default="sgd")
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', action="store_true")      # default is false
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=2000)

parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--lambda1', type=float, default=1.0)
parser.add_argument('--lambda2', type=float, default=10.0)
parser.add_argument('--sigma', type=float, default=10.0)

parser.add_argument('--c_subhead_num', type=int, default=10)
parser.add_argument('--c_head_dim', type=int, default=10)
parser.add_argument('--rl_head_dim', type=int, default=128)

parser.add_argument('--load_checkpoint', action="store_true")
parser.add_argument('--log_file_path', type=str,
                    default="logs/")
parser.add_argument('--model_save_path', type=str,
                    default="saved_models/")


args = parser.parse_args()


class Arguments:
    def __init__(self):
        self.dataset = args.dataset
        self.data_path = args.data_path

        self.model = args.model

        self.optimizer = args.optimizer
        self.lr = args.lr
        self.momentum = args.momentum
        self.nesterov = args.nesterov
        self.wd = args.wd
        self.batch_size = args.batch_size
        self.epochs = args.epochs

        self.temperature = args.temperature
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.sigma = args.sigma

        self.c_subhead_num = args.c_subhead_num
        self.c_head_dim = args.c_head_dim
        self.rl_head_dim = args.rl_head_dim
        if self.dataset == "CIFAR10":
            assert self.c_head_dim == 10

        self.load_checkpoint = args.load_checkpoint
        self.log_file_path = args.log_file_path
        self.model_save_path = args.model_save_path

        self.logger = None
        if cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def print_args(self):
        assert self.logger is not None
        self.logger.info(f"lambda1: {self.lambda1:.3f}")
        self.logger.info(f"lambda2: {self.lambda2:.3f}")
        self.logger.info(f"c_subhed_num: {self.c_subhed_num}")
        self.logger.info("================")


if __name__ == '__main__':
    args = Arguments()
    main(args)
