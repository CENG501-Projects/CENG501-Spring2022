# @TODO: Paper title

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

Shadow removol is one of the most important preprocessing application for the computer vision studies.  It brings up challanges for tracking and object detection. Recently developed shadow removel methodologies has problems of ignoring the spatially varient property of shadow images, lacking interpretability of CNN structures and not using masking information in the dataset efficiently resulting in ligthing the nonshadow dark albedo material areas. Our project paper, Efficient Model-Driven Network for Shadow Removal by Zhu et al., proposes deep network combines both model-driven and data-driven CNN-based approaches for shadow removal to overcome aforementioned problems. 


## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

In the paper, drawbacks of currently available shadow removel approaches are shown such as taking shadow effects uniform. To overcome this problem, an illumination transformation matrix is proposed such that non-shadow pıxel values are equal to its value. By this way spatially variant property of shadow images are considered. Apart from that, by introducing model-driven neural network, inerpreability of the network is increased.  Different from using Residual Block proposed, Dynamic Mapping Residual Block is designed as a basic module for the introduced network which increases performance of the model without introducing any parameters.

# 2. The method and my interpretation

## 2.1. The original method

@TODO: Explain the original method.

#### FORMULAS ####



## 2.2. My interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

In the paper, it is denoted that training is conducted by using a single NVIDIA GTX 1080Ti graphics card which has 11 GB VRAM. Since we had NVIDIA RTX 3070Ti graphics card which has only 8 GB VRAM, instead of training with images has a resoılution of 256x256 as it has been done in the paper, we trained our network with images with resolution of 128x128. Then we streched the outputs to 256x256. Apart from that, it is mentioned that model converges well after 150 epochs and we took that number as reference for our training stage. We used ISTD dataset.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

As it can be seen from the results, we obtained parallel results with ones presented in the paper. As mentioned eaerlier, we trained our model with lower resolution and it is observed that, this approach did not affect the results we obtained. However, we observed low performance while model tries to remove shadows which are located in colourful areas such as this one. 

### INSERT RESULT HERE ####

This situation is not presented in the paper, so we dont know whether it is caused by our implementation or model itself.

# 5. References

@TODO: Provide your references here.

Zhu, Y., Xiao, Z., Fang, Y., Fu, X., Xiong, Z., & Zha, Z.-J. (2022). Efficient Model-Driven Network for Shadow Removal. Proceedings of the AAAI Conference on Artificial Intelligence, 36(3), 3635-3643.

Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolutional networks for biomedical image segmentation. In MICCAI. Springer.

He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learning for image recognition. In CVPR.


Sandler, M.; Howard, A.; Zhu, M.; Zhmoginov, A.; and Chen, L.-C. 2018. Mobilenetv2: Inverted residuals and linear bottlenecks. In CVPR.


# Contact

Onuralp Maçan - onuralpmacann@gmail.com

Onur Oydu - onuroydu@gmail.com

@TODO: Provide your names & email addresses and any other info with which people can contact you.
