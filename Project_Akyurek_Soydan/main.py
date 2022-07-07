import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.utils import plot_model
from random import random
from numpy import load
import numpy as np

# import zeros, ones, asarray, vstack,

from numpy.random import randint
import matplotlib.pyplot as plt
from os import listdir

from numpy import load
from matplotlib import pyplot

from numpy import savez_compressed

from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.initializers import RandomNormal
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.utils.vis_utils import plot_model

from models import define_generator, define_discriminator, define_gan2, define_gen_fs, define_gen_sf
from define_composite_model import define_composite_model
from dataset import *
from train import *
from utils import *
from datetime import datetime


if __name__ == '__main__':
    # define image shapes
    image_shape = (512, 512, 3)
    image_shape_mask = (512, 512, 1)

    path_dataset = '../_datasets/ISTD/'
    log_time = datetime.now()
    log_time_dir = str(log_time.year) + str(log_time.month).zfill(2) + str(log_time.day).zfill(2) + '_' +  str(
        log_time.hour).zfill(2) + str(log_time.minute).zfill(2) + str(log_time.second).zfill(2)
    project_path = './'
    output_dir = 'output'
    output_path = os.path.join(project_path, output_dir)
    if output_dir not in os.listdir(project_path):
        os.mkdir(output_path)
    save_dir = log_time_dir
    save_path = os.path.join(output_path, save_dir)
    if save_dir not in os.listdir():
        os.mkdir(save_path)
    model_dir = 'model'
    model_path = os.path.join(save_path, model_dir)
    os.mkdir(model_path)

    # create the models
    gen_s = define_generator(image_shape, name='gen_s', type='shadow')
    gen_s.summary()
    plot_model(gen_s, to_file=model_path + '/gen_s_model_plot.png', show_shapes=True, show_layer_names=True)

    gen_f = define_generator(image_shape, name='gen_f')
    gen_f.summary()
    plot_model(gen_f, to_file=model_path + '/gen_f_model_plot.png', show_shapes=True, show_layer_names=True)

    disc_f = define_discriminator(image_shape, name='disc_f')
    disc_f.summary()
    plot_model(disc_f, to_file=model_path + '/disc_f_model_plot.png', show_shapes=True, show_layer_names=True)

    disc_s = define_discriminator(image_shape, name='disc_s')
    disc_s.summary()
    plot_model(disc_s, to_file=model_path + '/disc_s_model_plot.png', show_shapes=True, show_layer_names=True)

    gan2_f = define_gan2(gen_f, disc_f, image_shape, image_shape_mask)
    gen_fs = define_gen_fs(gen_f, gen_s, image_shape)

    gan2_s = define_gan2(gen_s, disc_s, image_shape, image_shape_mask, type='shadow')
    gen_sf = define_gen_sf(gen_s, gen_f, image_shape, image_shape_mask)

    dataset_size = 100
    # load dataset shadow free
    data_train_shadow_free = load_images(path_dataset + 'train/train_C/', size=(512, 512), dataset_size=dataset_size)
    data_test_shadow_free = load_images(path_dataset + 'test/test_C/', size=(512, 512), dataset_size=dataset_size)
    data_shadow_free = np.vstack((data_train_shadow_free, data_test_shadow_free))
    print('Loaded data shadow free: ', data_shadow_free.shape)

    # load dataset shadow
    data_train_shadow = load_images(path_dataset + 'train/train_A/', size=(512, 512), dataset_size=dataset_size)
    data_test_shadow = load_images(path_dataset + 'test/test_A/', size=(512, 512), dataset_size=dataset_size)
    data_shadow = np.vstack((data_train_shadow, data_test_shadow))
    print('Loaded data shadow: ', data_shadow.shape)

    # load dataset mask
    data_train_mask = load_images(path_dataset + 'train/train_B/', size=(512, 512), dataset_size=dataset_size)
    data_test_mask = load_images(path_dataset + 'test/test_B/', size=(512, 512), dataset_size=dataset_size)
    data_mask = np.vstack((data_train_mask, data_test_mask))
    print('Loaded data mask: ', data_mask.shape)

    # load image data
    dataset = load_real_samples(dataset=[data_shadow_free, data_shadow, data_mask])

    image_shape = data_shadow_free[0].shape

    training = sys.argv[1]
    print('******************************************')
    print('training = %s' % training)
    print('******************************************')

    # train models
    train(disc_s, disc_f, gen_f, gen_s, gan2_f, gan2_s, gen_fs, gen_sf, dataset, save_path, training)
