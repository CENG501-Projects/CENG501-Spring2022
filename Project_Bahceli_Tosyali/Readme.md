# SCIR-Net: Structured Color Image Representation Based 3D Object Detection Network from Point Clouds


This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

 Altough object detection for 2D images is highly examined and there are numerous state of the art tecniques,there aren't any useful object detection techniques
implemented in LIDAR technologies. This paper which is implemented by Qingdong He , Hao Zeng1, Yi Zeng and Yijun Liu and which is published in AAAI22 introduces a new method with neural networks in order to achieve satisfying results in the domain of LIDAR technologies. 

## 1.1. Paper summary

 This paper is actually combination of 3 separate  different neural networks and some smoothing between each of them. Pipeline is further explained below<br/>
 1 - Extract the training data<br/>
 2 - Generate a feature extractor ( which can be also be another neural network , or a more well known extractor )<br/>
 3 - First Neural Network in the paper, which is the feature embedder. After generating the feature extractor, concatenate the feature extractor  with the points <br/>
 in order to add some extra dimensionality. <br/>
 4 - Design two different losses for smoothing <br/>
 5 - After smoothing use an Enforced Detection Network (a complex CNN architecture ) to get 4 different images<br/>
 6 - Use one last neural network to classify those images and generate bounding boxes<br/>
# 2. The method and my interpretation

## 2.1. The original method
# 1 - Feature Embedding Generation
 The paper starts via giving intuition for why do we need feature extractor for point clouds. The reason is to basically differentiate the trining data from other training datas, two points from two different training samples may have the same x,y,z,r coordinates, but it is hard to have same covariance with two different point clouds. After the explanation paper gives us a list of current point cloud extractor which are all designed with multi layer perceptrons. After feature extraction part, paper states that we need to concatanate the feature descriptor values with each of the current points. After concatenation happened we get a matrix of ( N  x (F+4)) where N is the number of points and F is the dimensions of the feature extractor which is 1024 in the paper. Now we can feed this input to our first neural network.  
 
   
 The network consist of Three layers, in each layer except first we concatenate the latest output we have with originial points. Hidden layer and the output is always of the same which is 2 * N, since the objective is to project the points to 2D coordinate system.   <br/>
Ti+1 = σ(Wi(Hi+1)) = σ(Wi([Ti, G])) where Ti is the output of the previous layer (for first layer they are the points) <br/>
 After doing the iterations in the layers, we pass the newly obtained 2D point T, we use a custom loss in order to further normalize data.<br/>
 The custom loss is generated via calculating min||ti - tj||2 for each point where ti is 2D point in T after calculating min||ti - tj||2 which is called di we generate the loss for point ti which is 
 <br/>
 ![Alt text](images/customloss.png?raw=true "")
 <br/>
 
 
 And the the total loss is 
<br/>
  ![Alt text](images/customloss2.png?raw=true "")
<br/>
 
 After calculating the loss we need to normalize the data, the paper suggest two different normalization tehniques. 
 <br/>
  ![Alt text](images/normalization.png?raw=true "")
<br/>
 
 First of those normalizations is called Ball query Normalization. Here we generate the smallest grid M which is of size mxm (128 in the paper ) which fit the points generated T. After generating grid, we search around every point with a radius r and move the selected point to a corner of the grid  via weighted voting with the distances of other points between the radius.
<br/>
Second normalization is the Linear Interpolation normalization, we again get the grid and move every points with respect to the weights of the other points, but the difference here is that we use k nearest neighbours instead of sarching around a radius. The radius is recommended as 5 in this paper. 
<br/>
After normalizations, we get a new image. 
<br/>
  ![Alt text](images/resultimage.png?raw=true "")
  
Now we can freely use CNN based architectures to classify and find bounding boxes, which gets us to the most complex part of the paper. Enforced Detector.
# 2 - Enforced Detection Network


Here we have a complex neural network which aims to downsample and upsample with continous convolutions in order to generate 4 different images. Every convolution aside from downsampling and upsampling is done with 3x3 convolution followed by 1x1 convolution. The image from the beginning is supposed to be 128x128 since m is 128, 
and with the each layer the edges of the image is halved therefore we generate one image with 64x64 resolution one image with 32x32 resolution and one image with 16x16 resolution. Although it is implied the level 0 ( upper row ) is  also convoluted into a lesser resolution paper does not give explicit formula for that.
<br/>
After getting four images we conclude the labeling and bounding boxed via another neural network which is for classifiaction and regression at the same time. And the outputs of the final neural networks are the results
 <br/>
  ![Alt text](images/enforced.png?raw=true "")
<br/>
## 2.2. My interpretation 

During the project, there were some points that we could not comprehend and implement in our work.

·         The first one is the feature extractor. In the paper, the feature extractor module is not described in detail. Writers used their own module, which is not available on the internet. Therefore, we first searched for appropriate feature extractors. Then we found extractors such as principal component analysis (PCA) and support vector machines (SVM). We implement PCA for our feature extractor in our project to create different types of input features.

·         The second problem we encountered is detector and classifier and non-maximum suppression (NMS) with merging and scoring layers in the paper. The reason is that there is insufficient explanation and citations about these layers. Therefore, we implemented and customized a pre-trained neural network resnet34 as our detector and classifier. With the help of this, we could get predictions as bounding boxes and labels. After that, we implemented a layer for the NMS part to merge the final layers.

·         Besides these, since data size is enormous and requires massive computing power, we used only a part of the training data for our model to get the results.

·         The paper has a layer called ‘backbone’ for the enforced detection network. However, it is not clear how it is implemented or the source of this backbone.

.          The difference between the implementations can be seen by comparing these two images gathered after normalization. 
<br/>
 ![Alt text](images/resultimage.png?raw=true "")
 <br/>
 <br/>
 ![Alt text](images/image2.png?raw=true "")
 <br/>
# 3. Experiments and results

## 3.1. Experimental setup

The paper used the KITTI, Waymo Open, nuScenes datasets for the training and test dataset. There are eight classes for object detection in the dataset, and we trained our network for all these classes. We used only KITTI dataset for training and testing our model. In the enforced detection network, we implemented and used the feature normalization module, which is a combination of ball query normalization and bilinear interpolation normalization as in the paper with their constants such as m, ω while taking the same value for each. They used the average precision (AP) metric to compare different evaluation methods. We used labels to compare our model with the proposed model.

KITTI 3D includes 7481 training samples and 7518 test samples. However, due to large file sizes, we have only used a part of the training samples for analysis. We used cross-entropy for loss evaluation and Adam algorithm for the optimizer. We used Colab Pro for the training and obtained the results. The properties of the training model can be seen in the table below.
<br/>
 ![Alt text](images/result.png?raw=true "")
 <br/>
## 3.2. Running the code

You can access the main.ipynb in ths directory, simply download it and run it. However, the directories needed to be changed to your dataset.  

## 3.3. Results

After training and evaluating the results of our implementation of an end-to-end optimized network with Adam optimizer, we have reached quite a satisfactory accuracy with around 50% accuracy for the KITTI dataset, as seen in the table below.
<br/>
 ![Alt text](images/result_.png?raw=true "")
 <br/>
 
Since even with low amount of training data and epochs the training time of this complex network is enormous we had only one real attempt for training the network.

# 4. Conclusion
Although we implemented the paper, there are some mjaor problems that prevents us from using the full potential of the netwrok. The most important being that our gpu ram is not enough for the whole point cloud, since it is approximately takes up 384 GB of data with hidden weights. Therefore we disbanded a huge chunk of the data, via randomly sampling the points in point cloud. Although we though that this will hurt as very hard in terms of accuracy in class prediction, we still got nearly %50 of the classes correctly, but bounding boxes affected heavily because of the data that is dumped ,therefore we did not use bounding boxes as a metric.  
# 5. References
paper : 
He et al., "SCIR-Net: Structured Color Image Representation Based 3D Object Detection Network from Point Clouds" AAAI22

code:
NMS : https://gist.github.com/mkocabas/a2f565b27331af0da740c11c78699185 
# Contact

Batuhan Tosyali email : tosyalibatuhan@gmail.com 
Furkan Bahceli email : frknbhcl@gmail.com
