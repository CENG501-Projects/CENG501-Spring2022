import logging as lg
import math
import numexpr as ne
import numpy as np
import time

import torch
from numpy import linalg as la

lg.basicConfig(format='%(levelname)s\t- %(asctime)s\t- %(message)s',
               datefmt='%m/%d/%Y-%H:%M:%S',
               level=lg.DEBUG)

# TODO: Most matrices should be processed as Torch tensors rather than NumPy ndarrays to use GPU


class RandomFourierTransformer:
    transform_matrix = 0
    transform_bias = 0
    n_components = 0
    gamma = 0

    def __init__(self, gamma, n_components):
        self.n_components = n_components
        self.gamma = gamma

    def fit(self, X):
        d = X.shape[1]
        self.transform_matrix = np.random.normal(loc=0, scale=math.sqrt(2*self.gamma), size=(d, self.n_components)).astype(np.float32)
        self.transform_bias = (np.random.rand(1, self.n_components) * 2 * math.pi).astype(np.float32)

    def transform(self, Y):
        ny = Y.shape[0]
        angle = np.dot(Y, self.transform_matrix)
        bias = self.transform_bias
        factor = np.float32(math.sqrt(2.0 / self.n_components))
        return ne.evaluate("factor*cos(angle+bias)")


def get_pixel_vector(center_x, center_y, radius, image_width):
    kernel_size = int(radius * 2 + 1)
    vector = np.zeros(kernel_size ** 2, dtype=int)
    for y in range(0, kernel_size):
        for x in range(0, kernel_size):
            index = (center_x + x - radius) + (center_y + y - radius) * image_width
            vector[x + y * kernel_size] = index
    return vector


def zca_whitening(inputs):
    inputs -= np.mean(inputs, axis=0)
    sigma = np.dot(inputs.T, inputs) / inputs.shape[0]
    u, s, v = np.linalg.svd(sigma)
    epsilon = 0.1
    zca_matrix = np.dot(np.dot(u, np.diag(1.0/np.sqrt(s + epsilon))), u.T).astype(np.float32)

    i = 0
    while i < inputs.shape[0]:
        next_i = min(inputs.shape[0], i+100000)
        inputs[i:next_i] = np.dot(inputs[i:next_i], zca_matrix.T)
        i = next_i

    return inputs


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    _, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w


def project_to_trace_norm(A, trace_norm, d1, d2):
    A = np.reshape(A, (9*d1, d2))
    (U, s, V) = la.svd(A, full_matrices=False)
    s = euclidean_proj_l1ball(s, s=trace_norm)
    return np.reshape(np.dot(U, np.dot(np.diag(s), V)), (9, d1*d2)), U, s, V


def transform_and_pooling(patch, transformer, selected_group_size, gamma, nystrom_dim,
                          patch_per_side, pooling_size, pooling_stride):
    # TODO: Process as tensor rather than ndarray. Change return value too
    patch = patch.cpu().numpy()
    
    n = patch.shape[0]
    patch_per_image = patch.shape[1]
    selected_channel_num = patch.shape[2]
    pixel_per_patch = patch.shape[3]
    group_num = len(selected_group_size)  # TODO: always 1?
    feature_dim = group_num * nystrom_dim

    # construct Nystroem transformer
    patch = patch.reshape((n*patch_per_image, selected_channel_num, pixel_per_patch))
    psi = np.zeros((n*patch_per_image, group_num, nystrom_dim), dtype=np.float32)
    if transformer[0] == 0:
        transformer = np.empty(group_num, dtype=object)
        sum_value = 0
        for i in range(group_num):
            transformer[i] = RandomFourierTransformer(gamma=gamma, n_components=nystrom_dim)
            sub_patch = patch[:, sum_value:sum_value + selected_group_size[i]].reshape((n * patch_per_image, selected_group_size[i] * pixel_per_patch)) / math.sqrt(selected_group_size[i])

            transformer[i].fit(X=sub_patch)
            sum_value += selected_group_size[i]

    # Nystrom transformation
    sum_value = 0
    for i in range(group_num):
        sub_patch = patch[:, sum_value:sum_value + selected_group_size[i]].reshape((n * patch_per_image, selected_group_size[i] * pixel_per_patch)) / math.sqrt(selected_group_size[i])
        psi[:, i] = transformer[i].transform(Y=sub_patch)
        sum_value += selected_group_size[i]
    psi = psi.reshape((n, patch_per_image, feature_dim))
    # tprint("    transformation completes")

    # pooling
    pooling_per_side = int(patch_per_side/pooling_stride)
    pooling_per_image = pooling_per_side * pooling_per_side
    psi_pooling = np.zeros((n, pooling_per_image, feature_dim), dtype=np.float32)

    for pool_y in range(0, pooling_per_side):
        range_y = np.array(range(pool_y*pooling_stride, min(pool_y*pooling_stride+pooling_size, patch_per_side)))
        for pool_x in range(0, pooling_per_side):
            range_x = np.array(range(pool_x*pooling_stride, min(pool_x*pooling_stride+pooling_size, patch_per_side)))
            pooling_id = pool_x + pool_y * pooling_per_side
            index = []
            for y in range_y:
                for x in range_x:
                    index.append(x + y*patch_per_side)
            psi_pooling[:, pooling_id] = np.average(psi[:, np.array(index)], axis=1)

    # normalization
    psi_pooling = psi_pooling.reshape((n*pooling_per_image, feature_dim))
    psi_pooling -= np.mean(psi_pooling, axis=0)
    psi_pooling /= la.norm(psi_pooling) / math.sqrt(n*pooling_per_image)
    psi_pooling = psi_pooling.reshape((n, pooling_per_image*feature_dim))

    return torch.from_numpy(psi_pooling.astype(np.float16)), transformer


def central_crop(X, d1, d2, ratio):
    n = X.shape[0]
    size = int(math.sqrt(d1))
    cropped_size = int(size * ratio)
    X = X.reshape((n, size, size, d2))
    begin = int((size-cropped_size)/2)
    return X[:, begin:begin+cropped_size, begin:begin+cropped_size].reshape((n, cropped_size*cropped_size*d2))


def evaluate_classifier(x_train, x_test, y_train, y_test, A):
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    eXAY = np.exp(np.sum((np.dot(x_train, A.T)) * y_train[:, 0:9], axis=1))  # batch_size-9
    eXA_sum = np.sum(np.exp(np.dot(x_train, A.T)), axis=1) + 1
    probs = eXAY / eXA_sum
    loss = - np.average(np.log(probs))
    
    predict_train = np.concatenate((np.dot(x_train, A.T), np.zeros((n_train, 1), dtype=np.float32)), axis=1)
    predict_test = np.concatenate((np.dot(x_test, A.T), np.zeros((n_test, 1), dtype=np.float32)), axis=1)

    top_pred_train = np.argmax(predict_train, axis=1)
    top_pred_test = np.argmax(predict_test, axis=1)
    
    label_train = np.argmax(y_train, axis=1).astype(int)
    label_test = np.argmax(y_test, axis=1).astype(int)

    error_train = np.average(top_pred_train != label_train)
    error_test = np.average(top_pred_test != label_test)
    
    likelihood_est = -np.sum(label_test * np.log(top_pred_test + 1e-9)) / n_test

    return loss, error_train, error_test, likelihood_est, probs


def low_rank_matrix_regression(x_train, y_train, x_test, y_test, prev_A, d1, d2, reg, n_iter, learning_rate, ratio):
    # TODO: Process as tensor rather than ndarray. Change return value too
    x_train = x_train.cpu().numpy()
    x_test = x_test.cpu().numpy()
    
    n_train = x_train.shape[0]
    cropped_d1 = int(d1*ratio*ratio)
    
    if prev_A is None:
        A = np.zeros((9, cropped_d1 * d2), dtype=np.float32)  # 9-(d1*d2)
    else:
        A = prev_A
    
    A_sum = np.zeros((9, cropped_d1 * d2), dtype=np.float32)  # 9-(d1*d2)
    computation_time = 0
    error_train = 1
    error_test = 1
    dim = 0

    for t in range(n_iter):
        mini_batch_size = 50
        batch_size = 10

        start = time.time()
        for i in range(0, batch_size):
            index = np.random.randint(0, n_train, mini_batch_size)
            x_sample = x_train[index] # batch-(d1*d2)
            y_sample = y_train[index, 0:9] # batch-9

            # stochastic gradient descent
            XA = np.dot(x_sample, A.T)
            eXA = ne.evaluate("exp(XA)")
            # eXA = np.exp(XA)
            eXA_sum = np.sum(eXA, axis=1).reshape((mini_batch_size, 1)) + 1
            diff = ne.evaluate("eXA/eXA_sum - y_sample")
            grad_A = np.dot(diff.T, x_sample) / mini_batch_size
            # grad_A = np.dot((eXA/eXA_sum - y_sample).T, x_sample) / mini_batch_size
            A -= learning_rate * grad_A

        # projection to trace norm ball
        A, U, s, V = project_to_trace_norm(A, reg, cropped_d1, d2)
        end = time.time()
        computation_time += end - start

        A_sum += A
        if (t+1) % 250 == 0:
            dim = np.sum(s[0:25]) / np.sum(s)
            A_avg = A_sum / 250
            loss, error_train, error_test, _, _ = evaluate_classifier(
                central_crop(x_train, d1, d2, ratio),
                central_crop(x_test, d1, d2, ratio),
                y_train,
                y_test,
                A_avg
            )
            
            A_sum = np.zeros((9, cropped_d1*d2), dtype=np.float32)
            
            lg.info("iteration " + str(t+1) + ": loss=" + str(loss) + ", train error=" + str(error_train)
                    + ", test error=" + str(error_test) + ", dim=" + str(dim))
            # lg.info(str(computation_time) + "\t" + str(error_test))
    
    if n_iter < 250:
        dim = np.sum(s[0:25]) / np.sum(s)
        A_avg = A_sum / n_iter

    A_avg, U, s, V = project_to_trace_norm(np.reshape(A_avg, (9*cropped_d1, d2)), reg, cropped_d1, d2)
    
    loss, error_train, error_test, likelihood_est, probs = evaluate_classifier(
        central_crop(x_train, d1, d2, ratio),
        central_crop(x_test, d1, d2, ratio),
        y_train,
        y_test,
        A_avg
    )

    lg.info("iteration " + str(n_iter) + ": loss=" + str(loss) + ", train error=" + str(error_train)
            + ", test error=" + str(error_test) + ", dim=" + str(dim))
    
    dim = min(np.sum((s > 0).astype(int)), 25)
    return V[0:dim], error_train, error_test, likelihood_est, probs, A_avg
