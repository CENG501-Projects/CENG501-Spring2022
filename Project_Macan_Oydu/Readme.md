# Efficient Model-Driven Network for Shadow Removal

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

Shadow removol is one of the most important preprocessing application for the computer vision studies.  It brings up challanges for tracking and object detection. Recently developed shadow removel methodologies has problems of ignoring the spatially varient property of shadow images, lacking interpretability of CNN structures and not using masking information in the dataset efficiently resulting in ligthing the nonshadow dark albedo material areas. Our project paper, Efficient Model-Driven Network for Shadow Removal by Zhu et al., proposes deep network combines both model-driven and data-driven CNN-based approaches for shadow removal to overcome aforementioned problems. 


## 1.1. Paper summary

In the paper, drawbacks of currently available shadow removel approaches are shown such as taking shadow effects uniform. To overcome this problem, an illumination transformation matrix is proposed such that non-shadow pıxel values are equal to its value. By this way spatially variant property of shadow images are considered. Apart from that, by introducing model-driven neural network, inerpreability of the network is increased.  Different from using Residual Block proposed, Dynamic Mapping Residual Block is designed as a basic module for the introduced network which increases performance of the model without introducing any parameters.

# 2. The method and my interpretation

## 2.1. The original method

Shadow removing model constructed by using masking information and proposed illumination transfomation mapping. By introducing constraints on this mapping, shadow regions' information is recovered and information of non-shadow regions are preserved. Then, by bayesian recursion, where image with shadow is prior and no shadow image is posterior, variational model is constructed. Illustration of the overall model is provided in the paper as down below.

![image](https://user-images.githubusercontent.com/108632459/177399131-59faaa8a-0ba9-429c-93ef-af0e550836a9.png)


There are two subnetworks that are used in the above algorithm which are "NetworkA and "Networkinit". This networks have a structure of modified U-net inspired from ResNet and Mobilenet-v2. Basic blocks for both networks presented in paper as the following.

![image](https://user-images.githubusercontent.com/82730997/177494801-005ce976-a987-4008-8199-a2cb59ac17b5.png)

In each scale, depth convolutions with shortcuts are used. Iterative algorithm, folmulated from a Bayesian perspective, for the overall model provided in the paper as down below.

![image](https://user-images.githubusercontent.com/108632459/177399514-a443c4ba-003e-4042-8164-fe47fa4f100b.png)


## 2.2. My interpretation 

As explained in the paper, scales of the original UNet are replaced with depth convolution blocks given in the figure 4 of the paper. Since expansion coefficients of the depth convolutions are not given in the paper, we took expansion coefficients as 4 and 6 (same as in ResNet and Mobilenetv2 respectively) in "NetworkA" and "NetworkInit" respectively. Size of these convolutions are taken as 3x3 same as in ResNet and Mobilenetv2. Also, it is mentioned that "NetworkA" and "NetworkInit" involves 4 scales and have channels from the 1st to 4th scale 32, 64, 28, 256 respectively. We assumed that after 4th scale, there are 512 channels. 

Same hyperparameters give in the paper are used for our model. Mean squared error is used as loss function and implemented as given in the paper. Same as in the paper, Adam optimizer and Cosine Annealing scheduler are used during training. However, maximum number of iterations (T_max) for Cosine Annealing scheduler is not mentioned in the paper. Therefore, we took it same as number of epochs (150). Gradients given in the algorithm are calculated analytically.

# 3. Experiments and results

## 3.1. Experimental setup

In the paper, it is denoted that training is conducted by using a single NVIDIA GTX 1080Ti graphics card which has 11 GB VRAM. Since we had NVIDIA RTX 3070Ti graphics card which has only 8 GB VRAM, instead of training with images has a resolution of 256x256 as it has been done in the paper, we trained our network with images with resolution of 128x128. Then we streched the outputs to 256x256. Apart from that, it is mentioned that model converges well after 150 epochs and we took that number as reference for our training stage. It is stated that maximum number of iteration is set to 4 as is it trade-off between accuracy and speed. In that respect we took that value as reference and used that value in our implementation. We used ISTD dataset for training the model.

## 3.2. Running the code

Our trained model, structure of the network and other script that are used for training and testing are provided in the "codes" file. As explained in the  paper, "ShadowRemoverNetwork" consists of 2 subnetworks which are "NetworkA" and "NetworkInit". Implementation of these networks are given in "NetworkInit.py" and "NetworkA.py" scripts. The final model is given in the "ShadowRemoverNetwork.py" script. To make our lifes easier, we constructed another subnetwork named "NetworkA_iter" given in the "NetworkA.py" script. "NetworkA_iter" is written for constructing the "for" loop in the "ShadowRemoverNetwork" in a more understandable way. "dataset.py" contains "dataset" class that is used for reading images. Training algorithm and loss function are implemented in the "train.py". One can use "test.py" to test our trained model. "errorCalculator.py" is used to calculate root mean square error and peak signal noise ratio of the test images.

### Installation

Required python modules can be installed with the following command.
````
pip3 install -r requirements.txt
````

## 3.3. Results

From our trained model, we got the following results on ISTD dataset.

| Original Image (Input) - Non-Shadow Image (Output) - Ground Truth| 
| ------------- |
|![171_result](https://user-images.githubusercontent.com/82730997/177566424-65867079-02a7-4b94-9722-151bf00c341c.jpeg)|
|![378_result](https://user-images.githubusercontent.com/82730997/177566712-7b4eb117-ecd3-4f2f-aa27-6e21ee49ccb2.jpeg)|
|![496_result](https://user-images.githubusercontent.com/82730997/177566880-9eb400d6-c578-4f60-b7d7-5affac94ff96.jpeg)|
| ![25_result](https://user-images.githubusercontent.com/82730997/177533243-f72e768e-08d2-45e8-8518-ea1da2064a4b.jpeg)|
| ![12_result](https://user-images.githubusercontent.com/82730997/177533491-1c1653bb-3a9b-42b0-9621-ce1270f67246.jpeg)|
| ![15_result](https://user-images.githubusercontent.com/82730997/177533544-0d3e100b-e70b-4e9f-8036-2953335729d0.jpeg)|
| ![18_result](https://user-images.githubusercontent.com/82730997/177533626-b6f0f0b7-e142-4fed-b45b-63a2a70a0d61.jpeg)|
| ![6_result](https://user-images.githubusercontent.com/82730997/177533691-a55416a5-c2ca-4c98-96cc-bb79648c9987.jpeg) |


Result images given in the figure 5 of the paper are as the following.

![image](https://user-images.githubusercontent.com/82730997/177547868-cf47de6a-f0da-4e8d-ba5f-b68452be5309.png)

The root mean squared error, peak noise signal ratio and structural similarity are calculated during testing and results are as follows.

| Models        | PNSR           | RMSE  | SSIM|
| ----------------------------------- |:--------:| --------:| ----------|
| Our model (patch size: 128x128)| 78.34 | 0.044 | 0.9773|
| Original model (patch size: 256x256)| 29.85|5.09 | 0.9598|

We can easily notice the huge difference on RMSE and PNSR. This difference is occured due to different patch sizes which is the only difference betwen our trained model and original model. It makes more sense to compare the original model with another model which is trained with images having patch size of 256x256. 

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

It is observed that, with proposed illumination transformation mappping, shadow are removed with more realistic modelling. Obtained shadow free image results shows that this statement is true. Since we did not implement other shadow removel approaches, we cannot make any comment on relative performance of the model presented in the paper compared to other. However, as shown in the paper, from the following figure given in the paper we can easily see that this model performed better than other state-of-the-art shadow removal models.

|![image](https://user-images.githubusercontent.com/82730997/177570294-0da17f22-d26c-433a-9fbb-b7f031962e52.png)|

However, we observed low performance while model tries to remove shadows in which shadows are located in multiple ares with very distinct colours such as these. 
| Original Image (Input) - Non-Shadow Image (Output) - Ground Truth| 
| ------------- |
| ![480_result](https://user-images.githubusercontent.com/82730997/177568421-55e98476-e03f-4960-85a9-206d2f632a9b.jpeg) |
| ![460_result](https://user-images.githubusercontent.com/82730997/177568564-5d275d6f-d37f-417a-9f6c-a28d67381a6f.jpeg) |
| ![518_result](https://user-images.githubusercontent.com/82730997/177568760-d463f668-7644-4672-9e8f-30111cb89685.jpeg) |

This situation is not presented in the paper, so we dont know whether it is caused by our implementation or model itself. This problem may be eliminated with fine-tuning if it not caused due to implementation of the model. One may try increasing the number of epochs because we noticed that loss was still contiuning to drop until the last epoch, if we had taken a larger number of epochs (170-180) we might have gotten better results. 

### INSERT RESULT HERE ####


# 5. References

Zhu, Y., Xiao, Z., Fang, Y., Fu, X., Xiong, Z., & Zha, Z.-J. (2022). Efficient Model-Driven Network for Shadow Removal. Proceedings of the AAAI Conference on Artificial Intelligence, 36(3), 3635-3643.

Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolutional networks for biomedical image segmentation. In MICCAI. Springer.

He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learning for image recognition. In CVPR.


Sandler, M.; Howard, A.; Zhu, M.; Zhmoginov, A.; and Chen, L.-C. 2018. Mobilenetv2: Inverted residuals and linear bottlenecks. In CVPR.


# Contact

Onuralp Maçan - onuralp.macan@metu.edu.tr

Onur Oydu - onur.oydu@metu.edu.tr
