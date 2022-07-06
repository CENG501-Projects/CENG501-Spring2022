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
   
 The network consist of Three layers, in each layer except first we concatenate the latest output we have with originial points. Hidden layer and the output is always the same which is 2 * N, since the objective is to project the points to 2D coordinate system. 
      T1 = σ(W1(H1)) = σ(W1([P, G])) 
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
