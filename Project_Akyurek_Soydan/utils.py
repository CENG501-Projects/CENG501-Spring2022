"""

The loss functions and metrics are implemented as described in the paper. 
The functions provided in this files are not fully entegrated with our
implementation. We leave the entegration with VGG16 model and other details
as a future work.

"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Lambda


def PSNR(rmse, max_pixel = 1.0):

    if(rmse == 0):  # MSE is zero means no noise is present in the signal .
        return 100

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / rmse)
    return psnr

def RMSE(predictions, targets):
  return np.sqrt(np.mean((predictions-targets)**2))

def concat(x):
    return tf.concat((x[0], x[1]), axis=-1)

# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()

# See eqns. (6) and (7)
def discriminator_loss_fn(disc_out_real, disc_out_fake):
    real_loss = adv_loss_fn(tf.ones_like(disc_out_real), disc_out_real)
    fake_loss = adv_loss_fn(tf.zeros_like(disc_out_fake), disc_out_fake)
    return 0.5 * (real_loss + fake_loss) 


# Get the feature vectors of an image using VGG16 model
# layers: array of indices of layers to be extracted, if not provided all layers are used
def vgg_content(vgg_model, image, layers = None):

  if layers is None:  
    layers = range(len(vgg_model.layers)) # Get all the layers of the model
  
    outputs = [vgg_model.layers[i].output for i in layers]
    test_model = tf.keras.models.Model(inputs=vgg_model.inputs, outputs=outputs)
    features =  tf.stop_gradient(test_model.predict(image))

  return features


# Gram matrix is often used in style transfer applications
# We will insert the feature vector of an image
#
# Disclaimer: this code snipplet is taken from:
# https://www.tensorflow.org/tutorials/generative/style_transfer?hl=en
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

"""
def tf_MSE(preds, targets, keepbatch=True):
  tmp = tf.math.square(preds-targets)

  if(keepbatch):
    axes = tuple(range(1,tf.rank(tmp)))
    return tf.reduce_mean(tmp, axis=axes)
  else:
    return tf.reduce_mean(tmp)
"""
tf_MSE = keras.losses.MeanSquaredError()

# Color Loss is defined as in the paper (Vasluianu et al., p.4):
# 'Perform a Gaussian filter over the real and fake image
# and compute the Mean Squared Error (MSE)' 
#
# Paper doesn't mention the kernel size of Gaussian filter
def color_loss(fake_image, real_image):

  fake_smoothed =  tfa.image.gaussian_filter2d(fake_image)
  real_smoothed =  tfa.image.gaussian_filter2d(real_image)
  
  return tf_MSE(fake_smoothed, real_smoothed)  # Convert tensor to scalar


# Adjusted function to feed in Keras Model API 
# See https://keras.io/api/models/model_training_apis/ loss function part
def perceptual_loss(I1,I2, vgg16, alpha1=1, alpha2=0.1, alpha3=0):
    
  
  # Keras preprocessing for vgg16 overrides the variables,
  # but we don't want that, so we create copies of inputs
  I1_copy =  tf.identity(I1)
  I2_copy =  tf.identity(I2)

  # Images need to be preprocessed before VGG16
  # See Keras.models.VGG16.preprocess_input()
  # WARNING: I1 & I2 should be in range [0,255]
  tf.stop_gradient(preprocess_input((I1_copy * 0.5 + 0.5) * 255.))
  tf.stop_gradient(preprocess_input((I2_copy * 0.5 + 0.5) * 255.))
  
  # Get the layer outputs of vgg16 for the input images
  C1 =  vgg_content(vgg16, I1_copy)
  C2 =  vgg_content(vgg16, I2_copy)
  
  L_content = tf.zeros(I1.shape[0]) # (batch_size,) loss for feature vecs
  L_style = tf.zeros_like(L_content)
 
  
  style_layers = [0, 1, 2, 3, 4, 5]   # Get the style vectors from lower level features
  content_layers = [len(vgg16.layers)-2]  # Get the target content layer from higher level

  for layer in style_layers:
    L_style += tf_MSE(gram_matrix(C1[layer]), gram_matrix(C2[layer]))

  L_style /= len(style_layers)

  for layer in content_layers:
    L_content += tf_MSE(C1[layer], C2[layer])
  
  L_style = L_content = 0

  L_color = color_loss(I1, I2)
  return alpha1 * L_color + alpha2 * L_content + alpha3 * L_style



# We don't use this class for the current version of the project
# however, this version of the class is also trainable yet produces
# worse results than the current version.
# DISCLAIMER: this code snipplet is based on Keras CycleGAN example:
# https://keras.io/examples/generative/cyclegan/
class CycleGan(keras.Model):
    def __init__(
        self,
        generator_F,
        generator_S,
        discriminator_F,
        discriminator_S,
    ):
        super(CycleGan, self).__init__()

        self.gen_F = generator_F
        self.gen_S = generator_S
        self.disc_F = discriminator_F
        self.disc_S = discriminator_S
       
    def compile(
        self,
        gen_F_optimizer,
        gen_S_optimizer,
        disc_F_optimizer,
        disc_S_optimizer,

        disc_loss_fun,
        percep_loss_fun,
        mask_generator,
        loss_weights,
        vgg_model,

    ):
        super(CycleGan, self).compile()
        self.gen_F_optimizer = gen_F_optimizer
        self.gen_S_optimizer = gen_S_optimizer
        self.disc_F_optimizer = disc_F_optimizer
        self.disc_S_optimizer = disc_S_optimizer

        self.g = loss_weights
        self.mask_generator = mask_generator
        self.pixel_loss = keras.losses.MeanAbsoluteError() # L1 loss is used (see 3.3.1)
        self.mask_loss = keras.losses.MeanAbsoluteError()  # L1 loss is used (see 3.3.7)
        self.gan_loss = disc_loss_fun
        self.percep_loss = percep_loss_fun
        self.vgg16 = vgg_model


    def train_step(self, batch_data):
        # v is shadow, u is shadow-free image, m is for mask
        u, v, m = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # Shadow to fake shadow-free image
            u_hat = self.gen_F(v, training=True)
            u_hat = tf.cast(u_hat, tf.float32)

            m_hat_f = self.mask_generator([u_hat, v])
            m_hat_f = tf.cast(m_hat_f, tf.float32)

            # Shadow-free to fake shadow image
            gen_s_in = Lambda(lambda x: concat(x))([u, m])
            v_hat = self.gen_S(gen_s_in, training=True)

            m_hat_s = self.mask_generator([v_hat, u])
            m_hat_s = tf.cast(m_hat_s, tf.float32)

            # Cycle (s-free to fake shadow to fake s-free) u -> v -> u
            gen_s_in = Lambda(lambda x: concat(x))([u_hat, m_hat_f])
            v_r = self.gen_S(gen_s_in, training=True)

            m_s_r = self.mask_generator([u_hat, v_r])
            m_s_r = tf.cast(m_s_r, tf.float32)

            # Cycle (shadow to fake s-free to fake shadow): v -> u -> v
            u_r = self.gen_F(v_hat, training=True)
            u_r = tf.cast(u_r, tf.float32)

            m_f_r = self.mask_generator([v_hat, u_r])
            m_f_r = tf.cast(m_f_r, tf.float32)

            # Discriminator outputs
            disc_u = self.disc_S(u, training=True)
            disc_u_hat = self.disc_F(u_hat, training=True)
            disc_u_r = self.disc_F(v_hat, training=True)

            disc_v = self.disc_F(v, training=True)
            disc_v_hat = self.disc_S(v_hat, training=True)
            disc_v_r = self.disc_S(v_r, training=True)

            # Losses
            L_pix = self.pixel_loss(u,u_r) + self.pixel_loss(v, v_r)
            L_GAN = self.gan_loss(disc_u, disc_u_hat) + self.gan_loss(disc_v, disc_v_hat) + self.gan_loss(disc_u, disc_u_r) + self.gan_loss(disc_v, disc_v_r)
            L_mask = self.mask_loss(m_hat_f, m_f_r) + self.mask_loss(m_hat_s, m_s_r) # + self.g['b2'] * self.mask_loss(m_hat_star, m_hat_f)
            L_perceptual = self.percep_loss(u,u_r,self.vgg16) +  self.percep_loss(v,v_r,self.vgg16) 

            # Total generator loss
            total_loss_F = self.g['g1'] * L_GAN + self.g['g3'] * L_pix + self.g['g4'] * L_perceptual + self.g['g5'] * L_mask
            total_loss_S = total_loss_F                                            # same loss is provided, afawu from the paper

            # Discriminator loss
            disc_F_loss = self.gan_loss(disc_v, disc_u_hat)
            disc_S_loss = self.gan_loss(disc_u, disc_v_hat)


        # Get the gradients for the generators
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)
        grads_S = tape.gradient(total_loss_S, self.gen_S.trainable_variables)

        # Get the gradients for the discriminators
        disc_F_grads = tape.gradient(disc_F_loss, self.disc_F.trainable_variables)
        disc_S_grads = tape.gradient(disc_S_loss, self.disc_S.trainable_variables)

        # Update the weights of the generators
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )
        self.gen_S_optimizer.apply_gradients(
            zip(grads_S, self.gen_S.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_F_optimizer.apply_gradients(
            zip(disc_F_grads, self.disc_F.trainable_variables)
        )
        self.disc_S_optimizer.apply_gradients(
            zip(disc_S_grads, self.disc_S.trainable_variables)
        )

        return {
            "Gen_F_loss": total_loss_F,
            "Gen_S_loss": total_loss_S,
            "Disc_F_loss": disc_F_loss,
            "Disc_S_loss": disc_S_loss,
        }

# Debugging CycleGAN class and the losses
if __name__ == '__main__':
    # For feature extraction applications, include_top=False. 
    # See https://keras.io/api/applications/vgg/
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=image_shape) # TODO: make it X.shape
    vgg16.trainable = False
    
    """
    # Create cycle gan model
    cycle_gan_model = CycleGan(
        generator_F = gen_f, generator_S=gen_s, discriminator_F=disc_f, discriminator_S=disc_s
    )
    
    opt = Adam(learning_rate= tf.keras.optimizers.schedules.ExponentialDecay(
              initial_learning_rate = 0.005,
              decay_steps=10000,
              decay_rate=0.96,
              staircase=True), beta_1=0.9, beta_2=0.99)
    
    # Compile the model
    cycle_gan_model.compile(
        gen_F_optimizer = opt,
        gen_S_optimizer = opt,
        disc_F_optimizer = opt,
        disc_S_optimizer = opt,
        disc_loss_fun = discriminator_loss_fn,
        percep_loss_fun = perceptual_loss,
        mask_generator = mask_generator_output,
        loss_weights = {'g1':250, 'g2':20, 'g3':60, 'g4':50, 'g5':60, 'b1':10, 'b2':100},
        vgg_model = vgg16
        )
    
    
    batch_size = 2
    data_train_shadow_free =  tf.data.Dataset.from_tensor_slices(dataset[0]).batch(batch_size)
    data_train_shadow =  tf.data.Dataset.from_tensor_slices(dataset[1]).batch(batch_size)
    _masks = np.expand_dims(dataset[2][...,0],-1)
    data_train_mask =  tf.data.Dataset.from_tensor_slices(_masks).batch(batch_size)
    
    dataset = [data_train_shadow_free, data_train_shadow, data_train_mask]
    """
    
    print(">> Debugging Losses...")
    """
    for I1,I2 in zip(data_train_shadow_free, data_train_shadow):
          
          #print("I1 ", I1[0,0:3,0:3,0]) 
          #print("I2 ", I2[0,0:3,0:3,0])
          print("diff: ", tf.reduce_sum(I1-I2).numpy())
          loss = perceptual_loss(I1, I2, vgg16)
          print(loss.numpy())
          break
    """
