# Joint Semantic-Geometric Learning for Polygonal Building Segmentation

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and my interpretation

## 2.1. The original method
The original method consists of two parts which is a `Multi-task Segmentation Network` that predicts vertices of the polygon to segment the building, next there is a `Polygon Refinement Network` which takes these vertices and the cropped segment of the building and produces displacement values  for each vertex to improve the polygonal segmentation

### 2.1.1 `Multi-task Segmentation Network`

< Add info here >

### 2.1.2 `Polygon Refinement Network`
The Polygon Refinement Network Consists of two major parts. The first major part is the **Resnet Backbone & Vertex Embedding** which acts as the feature extractor for each vertix. The second major part is the **Propogation Model & Coordinate Transforming & Refined Polygon Vertices** model which is actually just one Gated Graph Neural Network.

#### **Resnet Backbone & Vertex Embedding**
The resnet backbone is a variant of the ResNet50 where skip-connection structure are used to up-sample and concatenate the feature maps of four skip layers. The size of the final feature map for vertex embedding. The input to the Resnet Backbone is 224x224x3 cropped image of the building segment from the original picture through the bounding boxes corresponding to each building instance and rescaled into the same size. The skip concatination leads to a feature map of 112x112x256 for the same building segment. The coordinates of building vertices obtained from Vertex Generation Model  are transformed accordingly for vertex embedding on the final feature map of the backbone. Each vertex is assigned with the features extracted from the channel direction of the cube.

#### **Propogation Model & Coordinate Transforming & Refined Polygon Vertices**

The polygonal vertices obtained from the Vertex Embedding step are considered as nodes of a graph, and every two neighbouring vertices constitutes an edge of the graph. A gated graph neural network is used to learn the offset needed for each vertex , the displacement needed to move the vertex to the ground truth. After the propagation process through the Gated Graph Neural Network,two two fully-connected layers are used to output a displacement value for each vertex.	The prediction of displacement value is also formulated as a classification issue and the model is trained with cross-entropy loss. In the coordinate transforming step, the output displacement classes of PRN are converted to the displacement coordinates, and added to the corresponding vertex coordinates.

## 2.2. My interpretation 

### 2.2.1 `Multi-task Segmentation Network`

< Add info here >

### 2.2.2 `Polygon Refinement Network`

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
