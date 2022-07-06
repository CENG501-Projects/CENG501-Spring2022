# Shadow Removal with Paired and Unpaired Learning

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

The shadows due to occluded light sources are interfering with many computer vision applications, reducing the quality of the results of the algorithms. Various deep learning implementations are utilized for realistic removal of these shadowed areas. This paper, published in Conference on Computer Vision and Pattern Recognition (CVPR) 2021, aims to successfully remove shadows from images while preserving the image content as realistically as possible [1]. Our aim is to use the paper as a guide, and measure the reproducibility of the shadow removal results they obtained.

## 1.1. Paper summary

The authors propose a single image shadow removal solution via self-supervised learning by using a conditional mask. Two types of data are utilized in the learning phase: paired and unpaired. The method proposed in the paper exploits several observations made on the shadow formations process and employs the cyclic consistency and the GAN paradigm as inspired by the CycleGAN [2], a seminal work for learning image-to-image translations between two image domains from unpaired images. 

Different from the existing shadow removal solutions, the method is able to utilize both paired (where the shadow and shadow-free version of the same image is given) and unpaired (where the shadow and shadow-free images are not matching with each other) datasets. Together with this contribution, the method also employs crucial loss functions which improves the state-of-the-art shadow removal process.

# 2. The method and our interpretation

## 2.1. The original method

Given a dataset with shadow images and respective masks, either paired or unpaired settings, the core of the study in the paper is to exploit the given mask information in a self-supervised fashion by using, randomly sampled shadow masks into the training framework (Fig 1.a) and reconstructing the original input, imposing the cycle-consistency (Fig 1.b). The major observation is that there is no need to impose strong pixel-wise loss, rather it is better to focus on color, content and textual losses. 

<img width="460" alt="Screen Shot 2022-07-06 at 20 56 07" src="https://user-images.githubusercontent.com/77360680/177613321-f9c432ad-6055-40a3-9279-17dc25014c3e.png">

**Figure 1**

There are two generators in the architecture: one removes the shadow occlusion given an image, denoted as G_f, the other adds a shadow given an image and corresponding shadow mask. These operations are done in two steps to obtain the cycle consistency of CycleGAN architecture as illustrated in Figure 1. In the first step, called the forward step, the original images are fed to the generators to obtain their fake counterparts. In the second step, reconstruction step, the fake results are fed to the generators to obtain another fake result, which is a reconstruction of their original parts from the forward step. After each pass of the generators, a new shadow mask is constructed by taking the difference between input and output and thresholding the result by its median. 

After obtaining the results for the whole cycle, the paper proposes several loss functions to utilize both real and fake images. The loss functions consider not only pixel-wise L1 loss, but also compares colors by applying a Gaussian filter to smooth the images which makes the loss function more robust to noises. In addition, they utilize Mean Squared Error (MSE) between the content and style of the images by extracting their feature vectors from VGG16 as well as they measure the MSE between the synthetic shadow masks and original ones.

## 2.2. Our interpretation 

The original method utilizes a variety of loss functions which aligns with the overall objectives of the networks. One category of the loss functions, adversarial losses, and its usage within the networks were not clear to us.

We assumed the discriminator loss function is the GAN loss function given in equations (1) and (2) in the supplementary material [3] as it was not explicitly stated in the paper. The GAN loss equations (6) and (7) in the original paper [1] were not matching with the inputs provided in equation (8). We interpreted that part as Mean Squared Error (MSE) between the fake or real label patches (O and J) and the discriminator's output patch given a fake or real sample. To this end, we passed the images one by one to the discriminator, instead of concatenating two of them.


# 3. Experiments and results

## 3.1. Experimental setup

For the experimental setup, we followed the implementation details as provided in both original paper and supplementary material. The generator and discriminator architecture is implemented with respect to Table 1 and Figure 1 in the supplementary material [3] by using Tensorflow and Keras API. On the discriminator side, we changed the input channels from 6 to 3, as we decided to feed in real or fake samples one by one and use the discriminator as a binary classifier instead of feeding real and fakes at the same time. On the other side, training flow of the networks are implemented as closely as possible to impose cycle consistency during the training.

The original paper provides information about the optimizer they used with initial learning rate and its scheduler and number of epochs they trained the network; however, the paper doesn’t mention the batch size they used for the training. Due to hardware limitations, we experimented with a small batch size of 1 or 2 images for the initial experiments with the whole network. We kept the dataset resolution the same as the original paper, 512 by 512 which also affected the required training time.

In addition, although we implemented the loss functions for the networks, we didn’t include VGG16 based losses to the training. There were two losses depending on VGG16’s feature vectors: style and content loss. Style loss for original shadow and shadow-free image pair was on the order of billions which exploded the total loss of the generators. Content loss on the other hand, was suitable; however, extracting VGG16 feature vectors for every step loaded the available RAM too quickly and slowed down the training as we preprocess the results before feeding to VGG16 model. Therefore, we didn’t include these two losses in the experiment, instead we included identity loss functions used in CycleGANs.

In short, our experimental setup follows the flow described in the original paper. The paired and unpaired settings are available for the training and testing. The discriminator input is changed to 3 channels and some of the loss functions we used differ from the original setting. The networks we implemented, together with the non-linearities and skip connections, are provided at Figures 7.1 and 7.2 in the Appendix.

## 3.2. Running the code

The datasets [ISTD](https://drive.google.com/file/d/1I0qw-65KBA6np8vIZzO6oeiOvcDBttAY/view) and [USR](https://drive.google.com/file/d/1PPAX0W4eyfn1cUrb2aBefnbrmhB1htoJ/view) should be downloaded into a dataset folder named “./datasets/” under the working directory, as “./datasets/ISTD/” and “./datasets/USR/”. 

"main.py" is responsible for training the models, see the path variables in the file and adjust them according to your environment. 

For testing purposes, the folder settings are provided in "test" folder. Download the pretrained models: [paired](https://drive.google.com/drive/folders/1EIN752CzIXI_ADpxpmzSnDhquF9h28Zt?usp=sharing) and [unpaired](https://drive.google.com/drive/folders/1qoNzxviQIKRJXtGWMmi3Z-5lO7VMxEeL?usp=sharing). Put them into the locations "./output/pretrained_models/pair/" and "./output/pretrained_models/unpair/" respectively. 

To run the .ipynb file, see [Jupyter Notebook](https://jupyter.org/install). Follow the instructions in the "cycleGAN.ipynb" and run the cells one by one. The notebook also installs the dependencies of the project.


## 3.3. Results

In our experiments, we didn’t include the VGG16 model for the loss functions which is one of the reasons that our results differ from the original paper. The feature vectors obtained from VGG16 provide crucial information about the style (low level features) and content (high level features) of the real and generated images. In the paper, it is mentioned that the content and style of the shadow and shadow-free images should not be very different, which provides informative loss functions for the models and lack of these losses impacts our results. 

In addition, due to the computational limitations, we trained the model with a smaller dataset size, small batch size and less epochs than the original paper. When we took a batch size of 1 to use SGD, we encountered with mode collapse due to dominating gradients of a single image type. 

In the end, we obtained the results as given in Figures 2-5. For some images, our deshadowing network successfully removes the shadow with minimal artifacts as seen in Figures 3 and 4. For some cases, the shadows ghosts are still visible as listed in Figure 5. In general, we obtain results as given in Figure 2.1 and 2.2. The networks are often successful at removing shadows from the consistent environment patterns, such as the flat surface in column two of Figure 2.1 or the tiles in the second column of Figure 2.2. Although it doesn’t change for most of the images, for some cases such as the last two columns of Figure 2.2, training with unpaired data  produces better results. Samples from the original paper are also provided in Figure 6.

![1](https://user-images.githubusercontent.com/77360680/177642533-8dcb000f-c6f2-4442-8d66-ca331578a3d6.png)

**Figure 2.1:** (Top) Original shadow images, (Middle) Original shadow-free images, (Bottom) Generated shadow-free images

![2](https://user-images.githubusercontent.com/77360680/177643441-4ae9f8d3-2dde-470d-8298-c3df1e985204.png)

**Figure 2.2:** (First) Original shadow (Second) Original shadow-free (Third) Generated shadow-free images from paired data (Fourth) Generated shadow-free images from unpaired data


![3](https://user-images.githubusercontent.com/77360680/177643479-ad7d98e9-ee8a-4f05-8872-9c0fc40c20be.png)

**Figure 3:** Selected successful cases (First) Original shadow (Second) Original shadow-free (Third) Generated shadow-free images from paired data (Fourth) Generated shadow-free images from unpaired data

![4](https://user-images.githubusercontent.com/77360680/177643551-2cd78ae6-af5d-4d0d-9d2f-57c7545e9d66.png)

**Figure 4:** Selected successful cases (First) Original shadow (Second) Original shadow-free (Third) Generated shadow-free images from paired data (Fourth) Generated shadow-free images from unpaired data

![5](https://user-images.githubusercontent.com/77360680/177643608-d71dfe54-0a5b-40ee-80b8-ddff9a1dd2ad.png)

**Figure 5:** Selected failure cases (First) Original shadow (Second) Original shadow-free (Third) Generated shadow-free images from paired data (Fourth) Generated shadow-free images from unpaired data

![paper_results](https://user-images.githubusercontent.com/77360680/177645352-8a7d7580-3967-4cde-863a-87ae9dcf7fb5.png)

**Figure 6:** Original paper results with paired and unpaired trainings


# 4. Conclusion

In conclusion, the results of the original paper seem to be reproducible with enough computing power. We observed that the unpaired training works better than paired setting in some cases, which is in parallel with the original paper's results. The differences in loss functions affects the network performance and the experiments illustrated the importance of the various loss functions utilized in the original paper. Selecting suitable loss functions for the networks allowed the generators to generate images that are close to original ones even in the early stages of training, and they were able to learn to bleach the shadow mask area while preserving the semantic details.

The original paper gives almost enough information to rebuild the entire architecture; although training of the models requires adequate computing power as the RAM provided by Google Colab wasn’t enough for training with the whole dataset after a few epochs. In the end, the architecture proposed in this paper is capable of training with both paired and unpaired datasets and with reproducible results. 

# 5. References

[1] Vasluianu, Romero, and Gool, "Shadow Removal with Paired and Unpaired Learning", CVPR2021.

[2] Jun-Yan Zhu*, Taesung Park*, Phillip Isola, and Alexei A. Efros. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", in IEEE International Conference on Computer Vision (ICCV), 2017. 

[3] Vasluianu, Romero, and Gool, "Shadow Removal with Paired and Unpaired Learning - Supplementary Material", CVPR2021.


# Contact

Yusuf Soydan yusuf.soydan@metu.edu.tr

Bartu Akyürek bartu.akyurek@metu.edu.tr

# Appendix

![gen_s_model_plot_copy](https://user-images.githubusercontent.com/77360680/177639618-3dabe3a3-6c66-43fe-9e7e-82ba5973d342.png)

**Figure 7.1:** Generator implementation

![disc_s_model_plot_copy](https://user-images.githubusercontent.com/77360680/177640162-074fd847-abc2-411c-8219-775abf162063.png)

**Figure 7.2:** Discriminator implementation
