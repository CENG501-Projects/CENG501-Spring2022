import torch
from library.data_structure import *
import sklearn
import numpy as np


class MetricCalculator:
    """
    This class is used to calculate metrics score (AUC, Error Rate and Reconstruction Error)
    """

    def __init__(self):
        self.total_iteration_no = 0.0
        self.total_auc_score = 0.0
        self.total_data = 0.0
        self.total_correct_score = 0.0
        self.total_mse = 0.0

    def calc_metric_score(self, metric_type, input, target, out):
        """
        Main function to calculate scores
        :param metric_type: type of the metric that wanted to calculate
        :param input: input data which is fed to the model
        :param target: label data
        :param out: input data which is calculated from the model
        :return: calculated score
        """
        self.total_iteration_no += 1.0

        # Area Under the ROC Curve
        if metric_type == MetricType.AUC:
            max_out = torch.argmax(out, dim=1)
            self.total_auc_score += sklearn.metrics.roc_auc_score(target.detach().cpu().numpy(),
                                                                  max_out.detach().cpu().numpy())
            avg_auc_score = self.total_auc_score / float(self.total_iteration_no)
            return avg_auc_score

        # Error Rate of the output data
        elif metric_type == MetricType.Error_Rate:
            self.total_data += target.shape[0]
            max_out = torch.argmax(out, dim=1)
            self.total_correct_score += np.sum(max_out.detach().cpu().numpy() == target.detach().cpu().numpy())
            return 1 - self.total_correct_score / self.total_data

        # Reconstruction Error
        elif metric_type == MetricType.Recon_Err:
            criterion = torch.nn.MSELoss(reduction='sum')
            self.total_mse += criterion(input.detach().cpu(), out.detach().cpu())
            avg_mse = self.total_mse.numpy() / self.total_iteration_no
            return avg_mse
