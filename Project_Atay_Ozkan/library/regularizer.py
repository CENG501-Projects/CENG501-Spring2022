import numpy as np
import torch
from .data_structure import *
import scipy

activation = {}


def get_activation(name):
    """
        Get activation value of the layer
        :param name: layer name
        :return: activation value
    """

    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


class Regularizer:
    """
        Adaptive Knowledge Driven Regularizer Class
    """

    def __init__(self, lambda_reg=0.0001, beta_reg=0.01, regularized_param_name=None):
        """
        :param lambda_reg:  λ is the hyperparameter that controls the regularization strength of model parameter
        :param beta_reg: β is used to trade off between correlation(r) and model parameters(w)
        """
        if regularized_param_name is None:
            self.regularized_param_name = 'hidden'
        else:
            self.regularized_param_name = regularized_param_name

        self.lambda_reg = lambda_reg
        self.beta_reg = beta_reg

        # theta denotes the significance weights
        # corr_reg is the Pearson correlation value between
        # r_vector is a normalized correlation value
        self.theta, self.corr_reg, self.r_vector = [], [], []

        # Bottom and Top Model parameter indices
        # These indices are determined by CPR values
        self.top_indices, self.bottom_indices = [], []

        # The length of the top and bottom VPR items in each layer
        self.number_of_top_item, self.number_of_bottom_item = [], []

        # Storing all correlation layers to show hypothesis
        # testing on these B number of correlation values (used in t value calculation)
        # Initialize this layers after each t ratio calculation
        self.all_correlation_layers = []

        # Ratio of pairs of neurons with t-values exceeding the critical values
        self.t_value_ratio = {}

    def init_theta(self, model):
        """
            Initialize CorrReg parameters(normalized correlation vector, theta and pearson correlation)
            :param model: model that will be  trained
            :return: None
        """
        i = 0
        self.theta, self.corr_reg, self.r_vector = [], [], []

        # CORR-Reg is imposed on the model parameters between the hidden fully-connected layers
        # Therefore, parameters that are named as "hidden" are used.
        # We named all Fully Connected Layer as "hidden_layer_"
        # Theta for all hidden neurons and is uniformly initialized
        for model_param_name, model_param_value in model.named_parameters():
            if 'weight' in model_param_name and self.regularized_param_name in model_param_name:
                i += 1
                if i > 1:
                    row = np.ones(model_param_value.shape)
                    corr = np.zeros(model_param_value.shape)
                    row = row * (1 / model_param_value.shape[0])
                    self.theta.append(torch.tensor(row))
                    self.corr_reg.append(corr)
                    self.r_vector.append(corr)


    def compute_correlation(self, activation_val_of_layers, is_t_value_tested):
        """
            Compute Pearson Correlation and Normalized Correlation values between layers
            :param activation_val_of_layers: hidden layers that are used to calculate correlation
            :param is_t_value_tested: boolean value that determines the saving all correlation values or not
            :return: None
        """
        # Add activation values to the list
        activations = []
        for act in activation_val_of_layers:
            activations.append(activation[act])

        # Correlation Calculation Loop
        corr_layers = []
        for i in range(len(activations) - 1):
            # Preprocessing according to the shape of the incoming input
            if len(activations[i].shape) == 3:
                in_layer = activations[i].detach().cpu()
                in_layer = np.average(in_layer, axis=1)
                out_layer = activations[i].detach().cpu()
                out_layer = np.average(out_layer, axis=1)

            else:
                in_layer = activations[i].detach().cpu()
                out_layer = activations[i + 1].detach().cpu()

            # Pearson Correlation
            # 0.9 hyperparameter is proposed by paper. To avoid unstability of batch-wise correlation
            correlation = np.corrcoef(in_layer, out_layer, rowvar=False)
            correlation = correlation[in_layer.shape[-1]:, :in_layer.shape[-1]]
            self.corr_reg[i] = 0.9 * self.corr_reg[i] + 0.1 * correlation
            abs_correg = abs(self.corr_reg[i])
            # Equation 4 in the paper
            self.r_vector[i] = abs_correg / abs_correg.sum(axis=1, keepdims=True)
            corr_layers.append(correlation)

        # Storing all correlation values that are in the batch for calculation t values
        if is_t_value_tested is True:
            self.all_correlation_layers.append(corr_layers)

    def update_theta(self, model, activation_val_of_layers, is_t_value_tested=False):
        """
            Updating Significance Weight(Equation 8 in the paper)
            :param model: the model that is under training
            :param activation_val_of_layers: hidden layers that are used to calculate correlation
            :param is_t_value_tested: boolean value that determines the saving all correlation values or not
            :return: None
        """


        # Add model parameter Weights to list
        i = 0
        layers_weight = []
        for mpn, mpv in model.named_parameters():
            if 'weight' in mpn and self.regularized_param_name in mpn:
                i += 1
                if i > 1:
                    layers_weight.append(mpv)

        # Firstly Compute Correlation Values to update theta(Significance Weight)
        self.compute_correlation(activation_val_of_layers, is_t_value_tested)

        # Loop for updating theta layer by layer
        for i in range(len(self.theta)):
            abs_correg = abs(self.corr_reg[i])
            r_vector = abs_correg / abs_correg.sum(axis=1, keepdims=True)
            alpha_minus_1 = self.lambda_reg * self.beta_reg * r_vector
            sum_alpha_minus_1 = sum(alpha_minus_1)

            weight = self.lambda_reg * pow(layers_weight[i].detach().cpu(), 2)
            sum_weight = sum(weight)

            denominator = sum_weight + sum_alpha_minus_1 + 0.000001
            numerator = weight + alpha_minus_1

            self.theta[i] = numerator / denominator

    def calc_Corr_reg(self, weight, theta, device):
        """
            Calculating CORR-Reg loss (the method is obtained from Equation 6)
            :param weight: model parameter weight
            :param theta: significant weight of model
            :param device: training device cuda or cpu
            :return: regularizer loss
        """
        #return torch.nan_to_num(-1 * (weight.pow(2) * torch.log(theta.to(device))).sum())
        return -1 * torch.log(theta.to(device)).sum()

    ###########################################################################################################
    #     The functions which are below used to Value Percentile Rank (VPR), Correlation Percentile Rank
    #     (CPR) and t-values testing.
    #
    #     The metrics are used to investigate Stability of Correlation Knowledge
    #     and Effects of Adaptive Regularization.
    #
    #     Value Percentile Rank(VPR) is the percentile rank of the absolute value of the model pa-
    #     rameter. Correlation Percentile Rank(CPR), is the percentile rank of the corresponding
    #     correlation value.
    #
    ###########################################################################################################

    def init_t_values(self):
        """
            Initializing correlation layers after t values calculated
            :return: None
        """
        self.all_correlation_layers = []

    def calculate_ratio_of_t_values(self, epoch_ratio, significance_level=0.05):
        """
            Calculating Ratio of pairs of neurons with t-values exceeding the critical values
            :param epoch_ratio: (current epoch)/(total epoch) ratio -> 0.25, 0.5, 0.75 and 1
            :param significance_level: significance level of critical value (specified in paper)
            :return: None
        """
        total_layer_no = np.array(self.all_correlation_layers).shape[1]
        total_batch_no = np.array(self.all_correlation_layers).shape[0]

        # Two-Tailed Critical Value
        two_tailed_t_value = scipy.stats.t.ppf(q=1 - significance_level / 2, df=total_batch_no - 1)

        # Calculate the ratio layer by layer
        for i in range(total_layer_no):
            mini_batches = []

            #  Get the all correlation values at current layer
            for j in range(total_batch_no):
                mini_batches.append(np.array(self.all_correlation_layers[j][i]))

            # Calculate the t scores of correlation values
            t_score = scipy.stats.ttest_1samp(np.array(mini_batches), popmean=False)[0]

            # Calculate the ratio
            exceeding_t_value_indices = np.argwhere(
                abs(t_score) > abs(two_tailed_t_value))
            ratio = (len(exceeding_t_value_indices) / np.array(t_score).size) * 100
            self.t_value_ratio[epoch_ratio, i] = ratio

    def get_t_ratios(self):
        """
            Getter for t values
            :return: Ratio of pairs of neurons with t-values exceeding the critical values
        """
        return self.t_value_ratio

    def compute_cpr(self):
        """
            Computing Correlation Percentile Rank &
            Determining bottom and top model parameter indices
            The details are explained in the paper -> Effects of Adaptive Regularization part
            :return: None
        """
        self.top_indices, self.bottom_indices = [], []

        for i in range(len(self.theta)):
            # Determine 25% and 75% rank boundaries from CPR
            bottom_boundary = np.quantile(self.theta[i], 0.25)
            top_boundary = np.quantile(self.theta[i], 0.75)

            # Determine indices fall in the bottom 25% and fall in the top 75%
            top_indices = np.argwhere(np.array((self.theta[i])) > top_boundary)
            bottom_indices = np.argwhere(np.array((self.theta[i])) < bottom_boundary)
            self.top_indices.append(top_indices)
            self.bottom_indices.append(bottom_indices)

    def compute_vpr(self, weights):
        """
        Computing Value Percentile Rank
        :param weights: model parameter weight
        :return: None
        """

        self.number_of_top_item, self.number_of_bottom_item = [], []

        # Calculating VPR layer by layer
        for i in range(len(self.top_indices)):

            # Getting the weights of at that layer
            weight = weights[i].detach().cpu().numpy()

            # Determine the top and bottom model weights
            top_weight = weight[self.top_indices[i][:, 0], self.top_indices[i][:, 1]]
            bottom_weight = weight[self.bottom_indices[i][:, 0], self.bottom_indices[i][:, 1]]

            #  Dividing VPR, which ranges from 0 to 1, into 20 bins. (like Figure 2)
            bins_for_vpr = np.linspace(0.05, 1, 20)
            top_prcntg, bottom_prcntg = [], []

            # Get initial rank at bin 0
            first_top_boundary = np.quantile(weight, 0)
            first_bottom_boundary = np.quantile(weight, 0)

            # Calculate the number of top and bottom model parameters whose VPR values fall into the corresponding
            # bins interval
            # e.g determine the number of model parameters that VPR values between 0.1 and 0.15
            for bin in bins_for_vpr:
                # Determine the boundary VPR values
                top_bound = np.quantile(weight, bin)
                bottom_bound = np.quantile(weight, bin)

                # Determine the parameter indices that the parameter value between boundaries
                top_indices = np.argwhere(
                    (first_top_boundary < np.array(top_weight)) & (np.array(top_weight) <= top_bound))
                bottom_indices = np.argwhere(
                    (first_bottom_boundary < np.array(bottom_weight)) & (np.array(bottom_weight) <= bottom_bound))

                top_prcntg.append(len(top_indices))
                bottom_prcntg.append(len(bottom_indices))

                # Reset the lower boundaries
                first_bottom_boundary = bottom_bound
                first_top_boundary = top_bound
            self.number_of_top_item.append(top_prcntg)
            self.number_of_bottom_item.append(bottom_prcntg)

    def get_vpr_values(self, vpr_avrg_layers_no, vpr_prcntg_dist_layer_no):
        """
        Getter for Value Percentile Rank
        :param vpr_avrg_layers_no: the layer number that is wanted to get their avg VPR values (for Table 4)
        :param vpr_prcntg_dist_layer_no:  the layer number that is wanted to visualize the percentage distribution
             of VPR value (for Figure 2)
        :return: vpr distributions and average values
        """
        avg_top_vprs, avg_bottom_vprs, dist_avg_vprs = {}, {}, {}

        # Calculating VPR Percentage along layer
        top_vpr_distribution = (np.array(self.number_of_top_item[vpr_prcntg_dist_layer_no]) / sum(
            self.number_of_top_item[vpr_prcntg_dist_layer_no])) * 100
        bottom_vpr_distribution = (np.array(self.number_of_bottom_item[vpr_prcntg_dist_layer_no]) / sum(
            self.number_of_bottom_item[vpr_prcntg_dist_layer_no])) * 100

        # Calculating average VPRs from the percentage distribution of VPR
        for lyr in vpr_avrg_layers_no:
            prcntg_top = (np.array(self.number_of_top_item[lyr]) / sum(self.number_of_top_item[lyr])) * 100
            prcntg_bottom = (np.array(self.number_of_bottom_item[lyr]) / sum(self.number_of_bottom_item[lyr])) * 100

            numerator = sum([prcntg_top[i] * (i + 1) for i in range(len(prcntg_top))])
            denominator = sum(prcntg_top)
            avg_top_vprs[lyr] = numerator / (denominator * 20)

            numerator = sum([prcntg_bottom[i] * (i + 1) for i in range(len(prcntg_bottom))])
            denominator = sum(prcntg_bottom)
            avg_bottom_vprs[lyr] = numerator / (denominator * 20)

            dist_avg_vprs[lyr] = avg_top_vprs[lyr] - avg_bottom_vprs[lyr]

        return top_vpr_distribution, bottom_vpr_distribution, avg_top_vprs, avg_bottom_vprs, dist_avg_vprs
