import os
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
from skimage.filters import threshold_otsu
from Losses import perceptual_loss
from numpy.random import randint
import matplotlib.pyplot as plt
from os import listdir
import torch
import torchvision.transforms as transforms


class QueueMask():
    def __init__(self, length):
        self.max_length = length
        self.queue = []

    def insert(self, mask):
        if self.queue.__len__() >= self.max_length:
            self.queue.pop(0)

        self.queue.append(mask)

    def rand_item(self, number):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        mask = self.queue[np.random.randint(0, self.queue.__len__())]
        if number > 1:
            for i in range(1, number):
                n_mask = self.queue[np.random.randint(0, self.queue.__len__())]
                mask = tf.concat((mask, n_mask), axis=0)
        return mask

    def last_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[self.queue.__len__() - 1]

    def mask_generator(self, image_s, image_f):
        to_pil = transforms.ToPILImage()
        to_gray = transforms.Grayscale(num_output_channels=3)
        for i in range(image_s.shape[0]):
            im_f = (to_gray(to_pil(np.array(image_f[i, :, :, :] * 127.5 + 127.5, dtype=np.uint8))))
            im_s = (to_gray(to_pil(np.array(image_s[i, :, :, :] * 127.5 + 127.5, dtype=np.uint8))))
            diff = (np.asarray(im_f, dtype=np.float32) - np.asarray(im_s, dtype=np.float32))
            diff = diff[:, :, 0]
            # difference between shadow image and shadow_free image
            L = threshold_otsu(diff)
            mask = torch.tensor((np.float32(diff >= L) - 0.5) / 0.5).unsqueeze(0).unsqueeze(-1)
            mask.requires_grad = False
            self.insert(mask)


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate ✬real✬ class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return X, y, ix


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape):
    # generate fake instance
    X = g_model.predict(dataset)
    # create ✬fake✬ class labels (0)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# save the generator models to file
def save_models(step, gen_f, gen_s, save_path):
    # save the first generator model
    filename1 = 'generator_free_model_%06d.h5' % (step + 1)
    save_f1 = os.path.join(save_path, 'files_gen_f', filename1)
    gen_f.save(save_f1)
    # save the second generator model
    filename2 = 'generator_shadow_model_%06d.h5' % (step + 1)
    save_f2 = os.path.join(save_path, 'files_gen_s', filename2)
    gen_s.save(save_f2)
    print('>Saved: %s and %s' % (save_f1, save_f2))


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, save_path, n_samples=5, type=None):
    if type == 'shadow':
        X_in, _, idx_x_in = generate_real_samples(trainX[0], n_samples, trainX[1].shape[0])
        sampled_masks = tf.gather_nd(trainX[1], indices=np.expand_dims(idx_x_in, axis=-1))
        X_out, _ = generate_fake_samples(g_model, tf.concat((X_in, sampled_masks), axis=-1), trainX[1].shape[0])
    else:
        # select a sample of input images
        X_in, _, _ = generate_real_samples(trainX, n_samples, 0)
        # generate translated images
        X_out, _ = generate_fake_samples(g_model, X_in, 0)

    # scale all pixels from [-1,1] to [0,1]
    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    # plot real images
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_in[i])
    # plot translated image
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_out[i])
    # save plot to file
    filename1 = '%s_generated_plot_%06d.png' % (name, (step + 1))
    save_path = os.path.join(save_path, filename1)
    plt.savefig(save_path)
    plt.close()


# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don✬t add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)


def predict_fake_samples(g_model, dataset, patch_shape):
    # generate fake instance
    X = g_model.predict(dataset)
    # create ✬fake✬ class labels (0)
    y = np.ones((len(X), patch_shape, patch_shape, 1))
    return X, y


def train(disc_s, disc_f, gen_f, gen_s, gan2_f, gan2_s, gen_fs, gen_sf, dataset, save_path, training=None):
    # paths
    file_dir = 'files_output'
    file_path = os.path.join(save_path, file_dir)
    os.mkdir(file_path)
    model_dir = 'model_saved'
    model_path = os.path.join(save_path, model_dir)
    os.mkdir(model_path)
    loss_dir = 'loss'
    loss_path = os.path.join(save_path, loss_dir)
    os.mkdir(loss_path)
    log_file_name = 'log_loss.txt'
    log_file_path = os.path.join(loss_path, log_file_name)

    mask_queue_size = 50
    mask_queue = QueueMask(mask_queue_size)
    # define properties of the training run
    n_epochs, n_batch, = 10, 2
    # determine the output square shape of the discriminator
    n_patch = disc_s.output_shape[1]
    # unpack dataset
    train_free, train_shadow, train_mask = dataset
    train_mask = train_mask[:, :, :, 0]
    train_mask = np.expand_dims(train_mask, axis=-1)

    log_loss_identity_gen_f, log_loss_identity_gen_s = [], []

    log_loss_gan2_f_gen_f, log_loss_gan2_f_disc_f = [], []
    log_loss_gan2_s_gen_s, log_loss_gan2_s_disc_s = [], []

    log_loss_gen_fs_free, log_loss_gen_fs_mask, log_loss_gen_fs_shadow = [], [], []
    log_loss_gen_sf_free, log_loss_gen_sf_mask, log_loss_gen_sf_shadow = [], [], []

    # calculate the number of batches per training epoch
    bat_per_epo = int(len(train_free) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        if training == 'unpair':
            ''' CYCLE '''
            ''' generator_free + generator_shadow '''
            v, v_patch, ix_v = generate_real_samples(train_shadow, n_batch, n_patch)
            u_ = train_free[ix_v]
            m_u_ = train_mask[ix_v]
            v_ = train_shadow[ix_v]
            gen_fs.trainable = True
            loss_gen_fs = gen_fs.train_on_batch(v, [u_, m_u_, v_, m_u_])
            _, m_f_hat, _, m_s_r = gen_fs.predict_on_batch(v)
            for i_mask in range(len(ix_v)):
                mask_queue.insert(tf.expand_dims(tf.gather(m_f_hat, indices=i_mask), axis=0))
                mask_queue.insert(tf.expand_dims(tf.gather(m_s_r, indices=i_mask), axis=0))

            ''' generator_shadow + generator_free '''
            u, u_patch, ix_u = generate_real_samples(train_free, n_batch, n_patch)
            mask = mask_queue.rand_item(len(ix_u))
            v_ = train_shadow[ix_u]
            m_u_ = train_mask[ix_v]
            u_ = train_free[ix_v]
            gen_sf.trainable = True
            loss_gen_sf = gen_sf.train_on_batch([u, mask], [u_, m_u_, v_, m_u_])
            _, m_s_hat, _, m_f_r = gen_sf.predict_on_batch([u, mask])
            for i_mask in range(len(ix_v)):
                mask_queue.insert(tf.expand_dims(tf.gather(m_s_hat, indices=i_mask), axis=0))
                mask_queue.insert(tf.expand_dims(tf.gather(m_f_r, indices=i_mask), axis=0))

            ''' IDENTITY'''
            ''' genenerator_free '''
            u, u_patch, ix_u = generate_real_samples(train_free, n_batch, n_patch)
            gen_f.trainable = True
            loss_identity_gen_f = gen_f.train_on_batch(u, u)

            ''' generetor_shadow'''
            v, v_patch, ix_v = generate_real_samples(train_shadow, n_batch, n_patch)
            mask = mask_queue.rand_item(len(ix_v))
            gen_s.trainable = True
            loss_identity_gen_s = gen_s.train_on_batch(tf.concat((v, mask), axis=-1), v)

            ''' GAN '''
            ''' generator_free + discriminator_free '''
            v, v_patch, ix_v = generate_real_samples(train_shadow, n_batch, n_patch)
            u_hat = gen_f.predict(v)
            u, u_patch, ix_u = generate_real_samples(train_free, n_batch, n_patch)
            X_disc = tf.concat((u_hat, u), axis=0)
            y_disc = tf.concat(
                (np.zeros((u_hat.shape[0], n_patch, n_patch, 1)), np.ones((u.shape[0], n_patch, n_patch, 1))), axis=0)
            disc_f.trainable = True
            disc_f.train_on_batch(X_disc, y_disc)
            v, v_patch, ix_v = generate_real_samples(train_shadow, n_batch, n_patch)
            y_gan2_f = [train_free[ix_v], np.ones((v.shape[0], n_patch, n_patch, 1))]
            disc_f.trainable = False
            gen_f.trainable = True
            loss_gan2_f = gan2_f.train_on_batch(v, y_gan2_f)

            ''' generator_shadow + discriminator_shadow'''
            u, u_patch, ix_u = generate_real_samples(train_free, n_batch, n_patch)
            mask = tf.gather(train_mask, indices=ix_u)
            v_hat = gen_s.predict(tf.concat((u, mask), axis=-1))
            v, v_patch, ix_v = generate_real_samples(train_shadow, n_batch, n_patch)
            disc_s.trainable = True
            disc_s.train_on_batch(v_hat, np.ones((v.shape[0], n_patch, n_patch, 1)))
            u, u_patch, ix_u = generate_real_samples(train_free, n_batch, n_patch)
            mask = mask_queue.rand_item(len(ix_u))
            y_gan2_s = [train_shadow[ix_v], np.ones((v.shape[0], n_patch, n_patch, 1))]
            disc_s.trainable = False
            gen_s.trainable = True
            loss_gan2_s = gan2_s.train_on_batch(tf.concat((u, mask), axis=-1), y_gan2_s)

        elif training == 'pair':

            ''' IDENTITY'''
            ''' genenerator_free '''
            u, u_patch, ix_u = generate_real_samples(train_free, n_batch, n_patch)
            gen_f.trainable = True
            loss_identity_gen_f = gen_f.train_on_batch(u, u)

            ''' generetor_shadow'''
            v = train_shadow[ix_u]
            mask = tf.gather(train_mask, indices=ix_u)
            gen_s.trainable = True
            loss_identity_gen_s = gen_s.train_on_batch(tf.concat((v, mask), axis=-1), v)

            ''' GAN '''
            ''' generator_free + discriminator_free '''
            v, v_patch, ix_v = generate_real_samples(train_shadow, n_batch, n_patch)
            u_hat = gen_f.predict(v)
            u = train_free[ix_v]
            X_disc = tf.concat((u_hat, u), axis=0)
            y_disc = tf.concat(
                (np.zeros((u_hat.shape[0], n_patch, n_patch, 1)), np.ones((u.shape[0], n_patch, n_patch, 1))), axis=0)
            disc_f.trainable = True
            disc_f.train_on_batch(X_disc, y_disc)

            y_gan2_f = [u, np.ones((v.shape[0], n_patch, n_patch, 1))]
            disc_f.trainable = False
            gen_f.trainable = True
            loss_gan2_f = gan2_f.train_on_batch(v, y_gan2_f)

            ''' generator_shadow + discriminator_shadow'''
            u = train_free[ix_v]
            mask = tf.gather(train_mask, indices=ix_v)
            v_hat = gen_s.predict(tf.concat((u, mask), axis=-1))
            v = train_shadow[ix_v]
            disc_s.trainable = True
            disc_s.train_on_batch(v_hat, np.ones((v.shape[0], n_patch, n_patch, 1)))

            y_gan2_s = [v, np.ones((v.shape[0], n_patch, n_patch, 1))]
            disc_s.trainable = False
            gen_s.trainable = True
            loss_gan2_s = gan2_s.train_on_batch(tf.concat((u, mask), axis=-1), y_gan2_s)

            ''' CYCLE '''
            ''' generator_free + generator_shadow '''
            v, v_patch, ix_v = generate_real_samples(train_shadow, n_batch, n_patch)
            u_ = train_free[ix_v]
            m_u_ = train_mask[ix_v]
            v_ = train_shadow[ix_v]
            gen_fs.trainable = True
            loss_gen_fs = gen_fs.train_on_batch(v, [u_, m_u_, v_, m_u_])

            ''' generator_shadow + generator_free '''
            u = train_free[ix_v]
            mask = tf.gather(train_mask, indices=ix_v)
            v_ = train_shadow[ix_v]
            m_u_ = train_mask[ix_v]
            u_ = train_free[ix_v]
            gen_sf.trainable = True
            loss_gen_sf = gen_sf.train_on_batch([u, mask], [u_, m_u_, v_, m_u_])

        else:
            raise Exception('No such training method')

        print('>%d/%d, '
              'loss_identity_gen_f:%.3f, loss_identity_gen_s:%.3f, '
              'loss_gan2_f_gen_f:%.3f, loss_gan2_f_disc_f:%.3f,'
              'loss_gan2_s_gen_s:%.3f, loss_gan2_s_disc_s:%.3f, '
              'loss_gen_fs_free:%.3f, loss_gen_fs_mask:%.3f, loss_gen_fs_shadow:%.3f, '
              'loss_gen_sf_free:%.3f, loss_gen_sf_mask:%.3f, loss_gen_sf_shadow:%.3f '


              % (i + 1, n_steps,
                 loss_identity_gen_f, loss_identity_gen_s,
                 loss_gan2_f[0], loss_gan2_f[1],
                 loss_gan2_s[0], loss_gan2_s[0],
                 loss_gen_fs[0], loss_gen_fs[1], loss_gen_fs[2],
                 loss_gen_sf[0], loss_gen_sf[1], loss_gen_sf[2]))

        if i % 10 == 0:
            summarize_performance(i, gen_f, train_shadow, 'StoF', file_path)

            log_loss_identity_gen_f.append(loss_identity_gen_f)
            log_loss_identity_gen_s.append(loss_identity_gen_s)

            log_loss_gan2_f_gen_f.append(loss_gan2_f[0])
            log_loss_gan2_f_disc_f.append(loss_gan2_f[1])

            log_loss_gan2_s_gen_s.append(loss_gan2_s[0])
            log_loss_gan2_s_disc_s.append(loss_gan2_s[1])

            log_loss_gen_fs_free.append(loss_gen_fs[0])
            log_loss_gen_fs_mask.append(loss_gen_fs[1])
            log_loss_gen_fs_shadow.append(loss_gen_fs[2])

            log_loss_gen_sf_free.append(loss_gen_sf[0])
            log_loss_gen_sf_mask.append(loss_gen_sf[1])
            log_loss_gen_sf_shadow.append(loss_gen_sf[2])

            f = open(log_file_path, "w")
            f.write('dataset_size = %s, epochs = %s, batch_size = %s, iteration = %s' % (
                train_free.shape[0], n_epochs, n_batch, n_steps))
            f.write('\n\nlog_loss_identity_gen_f = %s' % log_loss_identity_gen_f)
            f.write('\n\nlog_loss_identity_gen_s = %s' % log_loss_identity_gen_s)

            f.write('\n\nloss_gan_f_gen_f = %s' % log_loss_gan2_f_gen_f)
            f.write('\n\nloss_gan_f_disc_f = %s' % log_loss_gan2_f_disc_f)

            f.write('\n\nloss_gan_s_gen_s = %s' % log_loss_gan2_s_gen_s)
            f.write('\n\nloss_gan_s_disc_s = %s' % log_loss_gan2_s_disc_s)

            f.write('\n\nlog_loss_gen_fs_free = %s' % log_loss_gen_fs_free)
            f.write('\n\nlog_loss_gen_fs_mask = %s' % log_loss_gen_fs_mask)
            f.write('\n\nlog_loss_gen_fs_shadow = %s' % log_loss_gen_fs_shadow)

            f.write('\n\nlog_loss_gen_sf_free = %s' % log_loss_gen_sf_free)
            f.write('\n\nlog_loss_gen_sf_mask = %s' % log_loss_gen_sf_mask)
            f.write('\n\nlog_loss_gen_sf_shadow = %s' % log_loss_gen_sf_shadow)
            f.close()
        # summarize_performance(i, gen_s, (train_free, train_mask), 'FtoS', save_path, type='shadow')
        if (i + 1) % (bat_per_epo * 1) == 0:
            # save the models
            save_models(i, gen_f, gen_s, model_path)