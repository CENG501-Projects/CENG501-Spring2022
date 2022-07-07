# Single-View 3D Object Reconstruction from Shape Priors in Memory

# 1. Introduction

The paper “Single-View 3D Object Reconstruction from Shape Priors in Memory” being worked on was published at the CVPR conference in 2021. The purpose of paper is to construct a three-dimensional version of an object with the help of a single-view. My goal was to create and train the network correctly. Unfortunately, due to processing power, time, and poorly understood parts of the paper, I could not reach the result. I made some trainings by making some simplifications in the dataset and network structure. I will explain their cause and effect.

## 1.1. Paper summary

Networks that generally construct 3D objects from images cannot produce high-quality objects due to noisy backgrounds and occlusions. The network used in this paper, called Mem3D, has made progress in this field thanks to its architecture and rendered (R2N2) dataset.
Mem3D, like other networks, makes images more compact with the help of shape encoder. Shape decoder is used in the creation of 3D objects. Thanks to its memory network, Mem3D extracts shape priors and uses these priors in the construction of the object. This memory network provides a great advantage in constructing the object, especially in images with excessive occlusion.
Mem3D is trained with the R2N2 dataset. This dataset contains only rendered images. This trained model has also been tested with the Pix3D dataset containing real images.

# 2. The method and my interpretation

## 2.1. The original method

The Mem3D network described in the paper simply consists of 4 parts. It can be said that shape priors are determined in the network and 3D object is reconstructed by combining image features and shape priors.

### 2.1.1 Image encoder

The image encoder starts with the first three convolutional blocks of the ResNet-50 model ready. In this way, the incoming image is reduced to 512 layers and 28x28 dimensions. Then the ResNet is followed by three sets of 2D convolutional layers, batch normalization layers and ReLU. The kernel size of the convolutional layers is given as 3. In addition to these, after ReLu in the 2nd and 3rd layers, there is a maxpooling layer with a kernel size of 2. The channels of these convolutional layers are 512, 256 and 256 respectively.

### 2.1.2 Memory network

The memory network aims to explicitly construct the shape priors by storing the “image-voxel” pairs, which memorize the correspondence between the image features and the corresponding 3D shapes. To do this, it saves the data in the form of an X. It maps key image features and value shape priors to each other. The process of writing new data to the memory network is only done during training because ground thruth data is only available in this process. In order for data to be written to the memory during training, the similarity between the key and the image features must be below a certain rate. The similarity rate is given below.

$$S_k(F,K_i) = \frac{F \cdot K_i}{|F||K_i|}$$

The similarity for the memory network values can be calculated from the following formula.

$$S_v(v,V_i)= 1-\frac{1}{r_{v}^3} \sum_{j=1}^{r_{v}^{3}} (V_i^j-v^j)^2   $$

The following equation can be used to update the network.

$$ n_1 = argmax(S_k(F,K_i)) $$

After the model is trained, the memory netwok is used only for read operations. Keys and image features with similarity above beta value are sent to LSTM Shape Encoder.

$$ V = \left[ V_n |S_k(F, K_{n_i}) > \beta \right] $$

## 2.2. My interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
