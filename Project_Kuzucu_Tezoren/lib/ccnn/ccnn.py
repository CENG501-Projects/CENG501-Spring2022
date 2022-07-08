import logging as lg
import math

import numpy as np
import torch.utils.data

from numpy import linalg as la
from sklearn.preprocessing import label_binarize
from .utils import (
    get_pixel_vector,
    zca_whitening,
    transform_and_pooling,
    low_rank_matrix_regression,
    evaluate_classifier,
    central_crop
)

# CCNN is defined for two layers, additional methods are required for adding more.
# Strings indicating the methods:
MULTILAYER_METHODS = ["ZHANG", "TRANSFER_LRN"]
# "ZHANG": Layer generation method proposed in the original CCNN paper by Zhang et al.
# "TRANSF_LRN": The transfer learning method proposed by Du et al.

lg.basicConfig(format='%(levelname)s\t- %(asctime)s\t- %(message)s',
               datefmt='%m/%d/%Y-%H:%M:%S',
               level=lg.DEBUG)


class CCNNState:
    def __init__(self, filter_weights, A_weights):
        self.filter_weights = filter_weights
        self.A_weights = A_weights


class CCNN:
    def __init__(self,
                 train_dl: torch.utils.data.DataLoader,
                 test_dl: torch.utils.data.DataLoader,
                 train_img_cnt: int,
                 test_img_cnt: int,
                 from_state: CCNNState = None,
                 multilayer_method: str = "ZHANG",
                 n_iter: int = 5000,
                 device: torch.device = torch.device("cpu")
                 ):
        if multilayer_method not in MULTILAYER_METHODS:
            raise ValueError("Unrecognized CCNN layer addition method: " + multilayer_method)
        
        self.train_dl = train_dl
        self.test_dl = test_dl
        
        self.train_img_cnt = train_img_cnt
        self.test_img_cnt = test_img_cnt

        self.img_cnt = self.train_img_cnt + self.test_img_cnt

        self.state = CCNNState(filter_weights=[], A_weights=[])
        self.layer_count = 0
        self.last_layer_output = None
        self.train_accuracy = 0
        self.test_accuracy = 0
        self.log_likelihood = 0
        self.probs = None
        self.device = device

        if from_state is None:
            if multilayer_method == "ZHANG":
                # Generate first layer
                self.generate_layer(n_iter=n_iter)  # Increments layer_count too
            elif multilayer_method == "TRANSFER_LRN":
                # TODO: Take a pretrained model and use its state
                
                self.layer_count += 1
            else:
                raise ValueError("Unrecognized CCNN layer addition method in first layer generation: "
                                 + multilayer_method)
        else:
            for fw, aw in zip(from_state.filter_weights, from_state.A_weights):
                # TODO: Save filters with their generation param.s in CCNNState object. Use these params instead of the
                #   default ones below.
                self.generate_layer(filter_weight=fw, A_weight=aw, n_iter=n_iter)

    # TODO: Save reduced input & learned features
    def generate_layer(self,
                       patch_radius: int = 2,
                       nystrom_dim: int = 200,
                       pooling_size: int = 2,
                       pooling_stride: int = 2,
                       gamma: float = 2.,
                       regularization_param: int = 100.,
                       learning_rate: float = 0.2,
                       crop_ratio: float = 1.,
                       n_iter: int = 5000,
                       chunk_size: int = 5000,
                       filter_weight=None,
                       A_weight=None
                       # max_channel: int = 16
                       ):
        """Train and add a layer to the CCNN with the method proposed by Zhang et al."""
        
        lg.info("Begin generating layer #" + str(self.layer_count + 1) + ".")
        
        lg.info("Reading the dataset...")
        x_train = []
        x_test = []
        labels = []
        
        for inp, lbl in self.train_dl:
            x_train.append(inp)
            labels.append(lbl)
        
        for inp, lbl in self.test_dl:
            x_test.append(inp)
            labels.append(lbl)
        
        # Process train & test toghether
        x_train = torch.vstack(x_train).to(self.device)
        x_test = torch.vstack(x_test).to(self.device)
        
        x_raw = torch.cat((x_train, x_test)).to(self.device)
        labels = torch.hstack(labels).to(self.device)

        lg.debug("x_train shape: " + str(x_train.shape))
        lg.debug("x_test shape: " + str(x_test.shape))
        lg.debug("x_raw shape (2D): " + str(x_raw.shape))
        lg.debug("labels shape: " + str(labels.shape))
        
        lg.info("Detecting image parameters...")
        if x_raw.shape[2] != x_raw.shape[3]:
            raise ValueError(f"Expected square images, instead got width {x_raw.shape[2]}, height {x_raw.shape[3]}.")
        
        img_size = x_raw.shape[2]  # == x_raw.shape[3] == L (one side of img.)
        patch_size = 1 + (patch_radius * 2)  # F = 2*radius + 1
        patch_pixel_cnt = patch_size ** 2
        
        patch_cnt_one_side = img_size - (patch_radius * 2)  # == (L - F + 1)
        patch_cnt = patch_cnt_one_side ** 2
        
        pool_cnt = (patch_cnt_one_side // pooling_stride) ** 2
        
        channel_cnt = x_raw.shape[1]
        # feature_dim = nystrom_dim  # Since there is a single channel ...?

        # Vectorize the inputs
        # TODO: Check whether this should be done
        x_raw = x_raw.reshape((x_raw.shape[0], x_raw.shape[1], 1, -1)).squeeze(2)

        lg.debug("x_raw shape (vectorized 1D): " + str(x_raw.shape))

        lg.info("Constructing the patches...")
        
        patch = torch.zeros((self.img_cnt, patch_cnt, channel_cnt, patch_pixel_cnt),
                            dtype=torch.float32,
                            device=self.device)
        
        for y in range(0, patch_cnt_one_side):
            for x in range(0, patch_cnt_one_side):
                for i in range(0, channel_cnt):
                    # TODO: Check image param. usage. May be forming patches wrong
                    indices = get_pixel_vector(x + patch_radius, y + patch_radius, patch_radius, img_size)
                    patch[:, x + y * patch_cnt_one_side, i] = x_raw[:, i, indices]
        
        lg.debug("patch shape: " + str(patch.shape))
        
        lg.info("Applying local contrast normalization and ZCA whitening...")
        # TODO: Process as tensor rather than ndarray
        patch = patch.cpu().numpy()
        patch = patch.reshape((self.img_cnt * patch_cnt, channel_cnt * patch_pixel_cnt))
        patch -= np.mean(patch, axis=1).reshape((patch.shape[0], 1))
        patch /= la.norm(patch, axis=1).reshape(patch.shape[0], 1) + 0.1
        patch = zca_whitening(patch)
        patch = patch.reshape((self.img_cnt, patch_cnt, channel_cnt, patch_pixel_cnt))
        # TODO: Process as tensor rather than ndarray
        patch = torch.Tensor(patch, device=self.device)

        lg.debug("patch shape after normalization & whitening: " + str(patch.shape))
        
        lg.info("Creating features...")
        transformer = [0]
        base = 0
        feature_dim = nystrom_dim
        x_reduced = torch.zeros((self.img_cnt, pool_cnt * feature_dim), dtype=torch.float16, device=self.device)
        
        while base < self.img_cnt:
            lg.info("Processing samples: " + str(base) + "-" + str(min(self.img_cnt, base + chunk_size)))
            x_reduced[base:min(self.img_cnt, base + chunk_size)], transformer = transform_and_pooling(
                patch=patch[base:min(self.img_cnt, base + chunk_size)],
                transformer=transformer,
                selected_group_size=[channel_cnt],  # Always 1?
                gamma=gamma,
                nystrom_dim=nystrom_dim,
                patch_per_side=patch_cnt_one_side,
                pooling_size=pooling_size,
                pooling_stride=pooling_stride
            )
            base = min(self.img_cnt, base + chunk_size)
        
        lg.info("Applying normalization...")
        # TODO: Process as tensor rather than ndarray
        x_reduced = x_reduced.cpu().numpy()
        x_reduced = x_reduced.reshape((self.img_cnt * pool_cnt, feature_dim))
        x_reduced -= np.mean(x_reduced, axis=0)
        # frobenius_norm = la.norm(x_reduced)  # FIXME: Returns inf
        # FIXME: Returns 0 matrix since norm is inf. Skip for now
        # x_reduced /= (frobenius_norm / math.sqrt(self.img_cnt * pool_cnt))
        x_reduced = x_reduced.reshape((self.img_cnt, pool_cnt * feature_dim))
        # TODO: Process as tensor rather than ndarray
        x_reduced = torch.Tensor(x_reduced, device=self.device)
        lg.debug("x_reduced shape: " + str(x_reduced.shape))
        
        lg.info("Creating filters...")
        labels_binarized = label_binarize(labels, classes=range(0, 10))
        if filter_weight is None:
            if A_weight is None:
                # Train from scratch
                filter_weight, train_error, test_error, likelihood, probs, A = low_rank_matrix_regression(
                    # Split and pass concatenated sets
                    x_train=x_reduced[0:self.train_img_cnt],
                    y_train=labels_binarized[0:self.train_img_cnt],
                    x_test=x_reduced[self.train_img_cnt:],
                    y_test=labels_binarized[self.train_img_cnt:],
                    prev_A=None,
                    
                    d1=pool_cnt,
                    d2=feature_dim,
                    n_iter=n_iter,
                    reg=regularization_param,
                    learning_rate=learning_rate,
                    ratio=crop_ratio
                )
            else:
                # Train continuing from previous weights
                filter_weight, train_error, test_error, likelihood, probs, A = low_rank_matrix_regression(
                    # Split and pass concatenated sets
                    x_train=x_reduced[0:self.train_img_cnt],
                    y_train=labels_binarized[0:self.train_img_cnt],
                    x_test=x_reduced[self.train_img_cnt:],
                    y_test=labels_binarized[self.train_img_cnt:],
                    prev_A=A_weight,
    
                    d1=pool_cnt,
                    d2=feature_dim,
                    n_iter=n_iter,
                    reg=regularization_param,
                    learning_rate=learning_rate,
                    ratio=crop_ratio
                )
            
            lg.info("Got train accuracy: " + str(1 - train_error))
            lg.info("Got test accuracy: " + str(1 - test_error))
        else:
            # Use given filter_weight.
            # TODO: Process as tensor rather than ndarray
            filter_weight = filter_weight.cpu().numpy()
            A = None
            
            if A_weight is None:
                # Scores are not calculated
                train_error = 1
                test_error = 1
                likelihood = 0
                probs = None
            else:
                _, train_error, test_error, likelihood, probs = evaluate_classifier(
                    central_crop(x_reduced[0:self.train_img_cnt], pool_cnt, feature_dim, crop_ratio),
                    central_crop(x_reduced[self.train_img_cnt:], pool_cnt, feature_dim, crop_ratio),
                    labels_binarized[0:self.train_img_cnt],
                    labels_binarized[self.train_img_cnt:],
                    A_weight
                )
        
        filter_dim = filter_weight.shape[0]
        
        lg.debug("filter_weight shape: " + str(filter_weight.shape))
        
        # TODO: Should this step be split into forward func?
        lg.info("Applying filters...")
        # TODO: Process as tensor rather than ndarray
        x_reduced = x_reduced.cpu().numpy()
        output = np.dot(x_reduced.reshape((self.img_cnt * pool_cnt, feature_dim)), filter_weight.T)
        output = np.reshape(output, (self.img_cnt, pool_cnt, filter_dim))
        output = np.transpose(output, (0, 2, 1))  # Transpose last 2 dimensions ( torch.transpose(output, 1, 2) )
        # TODO: Process as tensor rather than ndarray
        x_reduced = torch.Tensor(x_reduced, device=self.device)
        
        lg.info("Feature dimension: " + str(output[0].size))
        lg.debug("output shape: " + str(output.shape))
        
        self.layer_count += 1
        # TODO: Process as tensor rather than ndarray
        #self.state.filter_weights.append(torch.from_numpy(filter_weight))
        self.state.filter_weights.append(None)  # FIXME: Temp fix...
        # TODO: Process as tensor rather than ndarray
        self.state.A_weights.append(None if A is None else torch.from_numpy(A))
        self.last_layer_output = torch.from_numpy(output)
        
        lg.info("Done layer generation #" + str(self.layer_count) + ".")
        
        self.train_accuracy = 1 - train_error
        self.test_accuracy = 1 - test_error
        self.log_likelihood = likelihood
        self.probs = probs
