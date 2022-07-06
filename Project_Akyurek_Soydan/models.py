import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, LeakyReLU, Conv2DTranspose, Lambda
from tensorflow_addons.layers import InstanceNormalization
import tensorflow_probability as tfp

from tensorflow.keras.layers import Activation, Conv2D, Conv2DTranspose, ReLU, Dropout, Concatenate, Add
from tensorflow.keras.initializers import RandomNormal
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.optimizers import Adam

# generator a resnet block
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first layer convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])
    return g


# define the standalone generator model
def define_generator(image_shape, name='generator', type=None):
    # weight initializationa
    init = RandomNormal(stddev=0.02)
    # image input

    if type == 'shadow':
        new_shape = (image_shape[0], image_shape[1], image_shape[2] + 1)
        in_image = Input(shape=new_shape)
    else:
        in_image = Input(shape=image_shape)

    # L1-64
    g1 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    g1 = LeakyReLU(alpha=0.2)(g1)
    # L2-128
    g2 = Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g1)
    g2 = InstanceNormalization(axis=-1)(g2)
    g2 = LeakyReLU(alpha=0.2)(g2)
    # L3-256
    g3 = Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g2)
    g3 = InstanceNormalization(axis=-1)(g3)
    g3 = LeakyReLU(alpha=0.2)(g3)
    # L4-512
    g4 = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g3)
    g4 = InstanceNormalization(axis=-1)(g4)
    g4 = LeakyReLU(alpha=0.2)(g4)
    g4 = Dropout(rate=0.5, noise_shape=None, seed=None)(g4)
    # L5-512
    g5 = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g4)
    g5 = InstanceNormalization(axis=-1)(g5)
    g5 = LeakyReLU(alpha=0.2)(g5)
    g5 = Dropout(rate=0.5, noise_shape=None, seed=None)(g5)
    # L6-512
    g6 = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g5)
    g6 = InstanceNormalization(axis=-1)(g6)
    g6 = LeakyReLU(alpha=0.2)(g6)
    g6 = Dropout(rate=0.5, noise_shape=None, seed=None)(g6)
    # L7-512
    g7 = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g6)
    g7 = InstanceNormalization(axis=-1)(g7)
    g7 = LeakyReLU(alpha=0.2)(g7)
    g7 = Dropout(rate=0.5, noise_shape=None, seed=None)(g7)
    # L8-512
    g8 = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g7)
    g8 = LeakyReLU(alpha=0.2)(g8)
    g8 = Dropout(rate=0.5, noise_shape=None, seed=None)(g8)
    # L9-512
    g9 = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g8)
    g9 = InstanceNormalization(axis=-1)(g9)
    g9 = ReLU()(g9)
    g9 = Dropout(rate=0.5, noise_shape=None, seed=None)(g9)
    g9 = Concatenate()([g9, g7])
    # L10-512
    g10 = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g9)
    g10 = InstanceNormalization(axis=-1)(g10)
    g10 = ReLU()(g10)
    g10 = Dropout(rate=0.5, noise_shape=None, seed=None)(g10)
    g10 = Concatenate()([g10, g6])
    # L11-512
    g11 = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g10)
    g11 = InstanceNormalization(axis=-1)(g11)
    g11 = ReLU()(g11)
    g11 = Dropout(rate=0.5, noise_shape=None, seed=None)(g11)
    g11 = Concatenate()([g11, g5])
    # L12-512
    g12 = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g11)
    g12 = InstanceNormalization(axis=-1)(g12)
    g12 = ReLU()(g12)
    g12 = Dropout(rate=0.5, noise_shape=None, seed=None)(g12)
    g12 = Concatenate()([g12, g4])
    # L13-256
    g13 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g12)
    g13 = InstanceNormalization(axis=-1)(g13)
    g13 = ReLU()(g13)
    g13 = Concatenate()([g13, g3])
    # L14-128
    g14 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g13)
    g14 = InstanceNormalization(axis=-1)(g14)
    g14 = ReLU()(g14)
    g14 = Concatenate()([g14, g2])
    # L15-64
    g15 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g14)
    g15 = InstanceNormalization(axis=-1)(g15)
    g15 = ReLU()(g15)
    g15 = Concatenate()([g15, g1])
    # L16-3 output
    g16 = Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g15)
    g16 = Conv2D(3, kernel_size=(4, 4), strides=(1, 1), padding='same', kernel_initializer=init)(g16)
    g16 = tf.keras.activations.tanh(g16)
    # define model
    out_image = g16
    model = Model(in_image, out_image, name=name)

    # compile model
    model.compile(loss='mae', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])

    return model


# define the discriminaor
def define_discriminator(image_shape, name='discriminator'):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_image = Input(shape=image_shape)
    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)  # define model
    model = Model(in_image, patch_out, name=name)
    # compile model
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model


def define_gan2(gen_a, disc_a, image_shape, mask_shape, type=None):
    if type == 'shadow':
        input_mask1 = Input(shape=mask_shape)
        gen_in = tf.concat((Input(shape=image_shape), input_mask1), axis=-1)
    else:
        gen_in = Input(shape=image_shape)
    gen_out = gen_a(gen_in)
    disc_out = disc_a(gen_out)
    model = Model(gen_in, [gen_out, disc_out])
    model.compile(loss=['mse', 'mse'], optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5, 0.5])
    return model


def mask_generator_output(image):
    U = image[0]
    V = image[1]

    # Convert the [-1,1] image to [0,1] range
    # Then convert to grayscale
    U_gray = tf.image.rgb_to_grayscale(U / 2 + 0.5)
    V_gray = tf.image.rgb_to_grayscale(V / 2 + 0.5)
    diff = U_gray - V_gray
    median = tfp.stats.percentile(diff, 50)
    mask_2d = tf.where(diff > median, 1, -1)
    return mask_2d


def concat(x):
    return tf.concat((x[0], x[1]), axis=-1)


def define_gen_fs(gen_f, gen_s, image_shape):
    v = Input(shape=image_shape)
    u_hat = gen_f(v)

    m_f_hat = Lambda(lambda x: mask_generator_output(x))([v, u_hat])
    m_f_hat = tf.cast(m_f_hat, tf.float32)
    m_f_hat.trainable = False

    gen_s_in = Lambda(lambda x: concat(x))([u_hat, m_f_hat])
    gen_s_in.trainable = False
    v_r = gen_s(gen_s_in)

    m_s_r = Lambda(lambda x: mask_generator_output(x))([u_hat, v_r])
    m_s_r.trainable = False
    m_s_r = tf.cast(m_s_r, tf.float32)

    model = Model(v, [u_hat, m_f_hat, v_r, m_s_r])
    model.compile(loss=['mse', 'mse', 'mse', 'mse'], optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5, 0.5, 0.5, 0.5])
    return model


def define_gen_sf(gen_s, gen_f, image_shape, mask_shape):
    u = Input(shape=image_shape)
    m = Input(shape=mask_shape)

    gen_s_in = Lambda(lambda x: concat(x))([u, m])
    gen_s_in.trainable = False

    v_hat = gen_s(gen_s_in)

    m_s_hat = Lambda(lambda x: mask_generator_output(x))([u, v_hat])
    m_s_hat.trainable = False
    m_s_hat = tf.cast(m_s_hat, tf.float32)

    u_r = gen_f(v_hat)

    # m_f_r = mask_generator_output(u_r, v_hat)
    m_f_r = Lambda(lambda x: mask_generator_output(x))([u_r, v_hat])
    m_f_r.trainable = False
    m_f_r = tf.cast(m_f_r, tf.float32)

    model = Model([u, m], [v_hat, m_s_hat, u_r, m_f_r])
    model.compile(loss=['mse', 'mse', 'mse', 'mse'], optimizer=Adam(lr=0.0002, beta_1=0.5),
                  loss_weights=[0.5, 0.5, 0.5, 0.5])
    return model
