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

The polygonal vertices obtained from the Vertex Embedding step are considered as nodes of a graph, and every two neighbouring vertices constitutes an edge of the graph. A gated graph neural network is used to learn the offset needed for each vertex , the displacement needed to move the vertex to the ground truth. After the propagation process through the Gated Graph Neural Network, two fully-connected layers are used to output a displacement value for each vertex.	The prediction of displacement value is also formulated as a classification issue and the model is trained with cross-entropy loss. In the coordinate transforming step, the output displacement classes of PRN are converted to the displacement coordinates, and added to the corresponding vertex coordinates.

## 2.2. My interpretation 

### 2.2.1 `Multi-task Segmentation Network`

< Add info here >
@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

### 2.2.2 `Polygon Refinement Network`

#### **Resnet Backbone & Vertex Embedding**
 
* The renet backbone structure was not explained in the paper, but rather could be found from one of the relevant literature mentioned which is from: [Efficient Interactive Annotation of Segmentation Datasets with Polygon-RNN++](https://arxiv.org/pdf/1803.09693.pdf)
* It is mentioned that the input to the resnet backbone is is a cropped image from the bounding box of the building segment. However, in the illustrations and the logic implies that it is not exactly cropped to the bounidng box , such as if the predicted vertex for the corner is underestimating the corner and within the building, the cropped input according to the bounding box would not include the building corner which would be the ideal point for the vertex to be moved. To solve this, a tolerance was introduced to the bounding box as a sort of padding to the bounding box, where the image croped for building segment is larger than the bounding box, which was not mentioned in the paper.
* For the training stage, the bounding box coordinates are provided from the dataset, however, in the inference stage of the model, the output of the Vertex Generation Model are the polygon vertices, not the bounding box, so there needs to be a stage of getting bounding box from segmentation vertices, this is interpreted as finding the maximum and minimum vertices of the segmentation in the x and y axis and adding the pre-defined tolerance to it.
* As the feature map generated from the encoder is of size 112x112 with a feature length of 256, the predicted vertices need to be scaled down to this scale.
* The arrow in the Main Figure 1 of the paper shows that the intermediate result generated from the Vertex Generation Model is used for the vertex embedding, however, as the input picture to the encoder is not the Intermediate result from the VGM but a cropped and rescaled image of one building instance from the intermediate results, the vertex coordinates used for the vertex generation model are interpreted as to be from the cropped and rescaled image and not from the intermediate result of full picture as using vertex coordinates scaled from the 300x300 picture to 112x112 would not represent the same coordinate but the vertex coordinates scaled down from the 224x224 cropped image would represent the same coordinate in the 112x112 version.

#### **Propogation Model & Coordinate Transforming & Refined Polygon Vertices**

* It is mentioned that every two neighbouring vertices constitutes an edge of the graph, however, it is not mentioned what the edge attributes are, are they directed edges or if they are weighted edges, or which information is integrated into these edges.
* Further, it is not specified what format the graphs are treated in, if they are as adjaceny matrices or if they are used as dictionaries or a different method.
* Tne implementation details of the Gated Graph Neural Network are also not given other than the information that the result from it is passed through two fully connected networks to generate the displacement prediction for each vertex. The vertices of the graph are also passed to the GGNN one at a time for each graph, it is not mentioned how the graph is interpreted to be passed thorugh this Graph Neural Network used as a Recurrent Neural Network.
* It is mentioned that the model predicts a displacement value and this is treated as a classification issue and trained with cross entropy loss. It is also mentioned that the model predicts a final output of format 15x15 incorporating a displacement og +7 and -7 from the centre of the 15x15 grid. However, it is not mentioned how this 15x15 output is mapped to a classification problem. This information is interpreted as converting each element of the 15x15 grid to a label from the range of 0 to 224 classes.
* It is also not mentioned if the output from the propogation model of [-7,7] pixels is to be appllied on the intermediate result coordinates or to the cropped building instance, or if it is to be added to the coordinate of the feature map of 112x112 dimentions. It is interpreted that the Coordinate Transforming step is added to the coordinates of the segment which is cropped, so these vertex points are displayed.

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
