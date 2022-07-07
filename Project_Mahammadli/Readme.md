# @TODO: Paper title

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).
CNN has great proccesses in many tasks such as image classification in deep learning. However, because of the defficiency of the ability to process large image rotation of images, CNN can fail to classify images. CNN is used many fields such as biomedical,astronomi and industry. Therefore, rotation invariance of neural networks has been more important. There are lots of data augmentation techniques to recognize rotated objects and they may increase parameter. In this paper we make feature maps before and after CNN rotation equivariant, so whole NN is rotation invariant. Main techniques :
      1) Local Binary Pattern Operator based Regional Rotation Layer(RRL) reproduced.
      2) without new parameters, RLL make feature maps .
      3)Evaluated RLL with Lenet-5,ResNet-18,tiny-yolov3

## 1.1. Paper summary

@TODO: First, all of the datasets Cifar-10,Cifar10-rot(rotated with angle 0,90,180,270 images),Cifar10-rot+(rotated with angle in range[0,360)), Plankton dataset, Pascal VOC 2007-12 datasets evaluated in models the the models reproduced with RLL evaluated with same datsets and compared scores,and accuracies. Without new parameters, RLL makes the feature maps before and after convolution equivarience, aand so makes the entire NN rotation invariant With rotation angles 0 , 90,180,270 ; the feature maps are exactly same. With arbitrary rotation angle, there is small distinctions between feature maps. Finaly, RRL with LeNet-5,Resnet-44, ResNet-18 and tiny- yolov3 are evaluated and we get good results.  

# 2. The method and my interpretation

## 2.1. The original method

   The standard convolutional neural networks do not have the property of rotation invariance. Trained by the upright sam- ples, the performance drops significantly when tested by the rotated images. To solve this problem, we add a regional ro- tation layer (RRL) before the convolutional layers and the fully connected layers. The main idea is that we indirectly achieve rotation invariance by restricting the convolutional features to be rotation equivariant. A series of LBP feature values are obtained by rotating the surrounding points, and the minimum of these values is selected as the LBP value of the central pixel
In this paper, the points are rotated to the minimum state of LBP, so as to achieve the rotation invariance of angle. 
LBP is operated in a window, while RRL is operated on the feature maps. The feature maps are sampled one by one in the form of sliding window, and LBP is implemented in each window. So we can rotate the feature maps to the same state even with different input orientations.
## 2.2. My interpretation 

  In the paper there is small unclear place they do not describe places of RLLs in Resnet-18,Resnet-44, and tiny-yolov3 so we do not add to much RLLs before all CNN layers so that not slowing down the training.

# 3. Experiments and results

## 3.1. Experimental setup

In this experimet we use google colab
-Dataset and Lenet-5 Cifar-10 is used.
-Resnet-18 and Cifar-10 is used.
-yolov3-tiny and dataset Pascal VOC 2007-12 is used.
-Plankton and Resnet-44 is mentioned

Required libraries:
-Tensorflow 
-Pytorch
-Numpy
-OpenCv

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
Fatma Ceyda Gökçe: Email:fcg.13@hotmail.com
                   Linkedln:https://www.linkedin.com/in/fatma-ceyda-gokce/
