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

### 3.1.1 `Multi-task Segmentation Network`

< Add info here >
@TODO: Describe the setup of the original paper and whether you changed any settings.

### 3.1.2 `Polygon Refinement Network`

* As the dataset of the paper,OpenAI Mapping Challenge Dataset, was available on Kaggle, it was utilized through Kaggle directly to the Google Colab File.
* The Polygon Refinement Netwok is trained independtly of the Multi-task Segmentation Network, artificial training data was generated for the Polygon Refinement Network through the Dataset directly.
  * From the dataset, the first 500 instances of building segments are iterated over, which lead to 3985 vertex predictions.
  * As this is the ground truth data, artificial test data is created from this by adding random noise to the predicted vertex. 
  * It was also noted that the bounding box data of certain building segments were erroneous, where the ground truth segmentation and ground truth bounding box did not have any overlapping area. To solve this, bounding box information was generated from the segementation data , as would be done in inference phase, through the minimum and maximum of all the vertices of the polygon in x and y axis. From these min and max values, a tolerance is added to get the bounding box. 
  * This tolerance is added above to the bounding box to enlarge it, as further now in the cropepd building segment from the original picture, noise will be added to deviate the vertices from the true points. As the prediction of the vertex displacement is to be done in the range of [-7,7], this is also taken as the range of the noise added.
  * When the noise is added, the noisy polygon vertex located in the 112x112 feature map with 256 features is chosen as the data to analyse, and the noise added in the format of [A,B] , where A and B are values between +7 and -7, is trated as the truth label of this new data.
* As cross entropy is used as the loss, and the task is treated as a classfication problem, the label data is to be reformated from the [A,B], where A and B are values between +7 and -7, to multiclass labels and as this is a 15x15 grid, it converts to 255 individual labels from range 0 to 224.
  * This mapping function from [A,B] format to a label is performed through: $$Label=[(A+7)*15] + (B+7)$$.
  * Through this, the top left element,[-7,-7], of the 15x15 grid is formulated as 0, the bottom right element ,[7,7], is the last element with label 224, and the centre of the 15x15 grid which account for [0,0] displacement is labaled 112.
  * This mapping function also works to convert the label back to displacement which is the purpose of the Coordinate Transforming Model.
* The vertex and image originate in the format of a 300x300 image. After the bounding box is determined through the segmentation from the Multiclass Segmentation Network, The bouding box is cropped out and the vertex coordinates are transformed from the original 300x300 frame to this new arbitrary bounding box shape. Further this cropped image is now rescaled to 224x224 to be suitable as an input with resnet, and now the vertex coordinates are transformed with the ratio of: $$X_{new} = {X_{old}*224 \over X_{length} }$$ $$Y_{new} = {Y_{old}*224 \over Y_{length} }$$ These vertices are then halved to be applicable to the final feature map of 112x112
* These vertices , now transformed to the 112x112 shape, are then used to extract node-level 256 features for each vertex from the feature map.
* These nodes could then be used as inputs to the propogation model.


## 3.2. Running the code
As the two parts of Multi-Task Segmentation Network  and Polygon Refinement Network are trained seperately, they would also be explained seperately.
### 3.2.1 `Multi-task Segmentation Network`

< Add info here >
@TODO: Explain your code & directory structure and how other people can run it.

### 3.2.2 `Polygon Refinement Network`
A Google colab file is prepared which extensively explains the steps to run it, however they will be summarzied below:
* The dataset used is downloaded from Kaggle and the model does not require any other uploads/inputs from the user.
* The data is preprocessed and converted to a unified pandas dataframe to hold the filen details and also the annotations details.
![del1](https://user-images.githubusercontent.com/69632507/177655663-ea14e713-22ae-4b94-b173-0b8913f1f7d7.jpg)
* Some of the data is visualized to show the user what nature of data they are dealing with. 
![del](https://user-images.githubusercontent.com/69632507/177655397-3b7e037b-a93f-4305-b0f6-47fb6266e104.jpg)
* It is then illustrated how the building instance images are transformed from 300x300 to Bounding Box Dimensions to 224x224 Images
![image](https://user-images.githubusercontent.com/69632507/177657816-154b8fd2-d5c1-4a7d-8182-624d3caf8cca.png)
* The vertices in the 224x224 image are then exposed to noise to generate training data, with their orginal location being the ground truth location, and the displacement between these two acting as the label for the propogation model. The blue polygon is the ground truth and the red + symbols are the noisy segmentation generated to be finetuned by the Polygon Refinement Network
![image](https://user-images.githubusercontent.com/69632507/177658122-b2cbe42a-cbfc-4565-9ad5-dabb7bb5fb63.png)
* These red vertices are then passed through vertex embedding to be associated with their 256 features for their location. This data is now ready for the Propogation Model
* As enough implementation details  were not provided to formulate the vertices of the polygon effectively into a graph and could not be found on external literature referenced in the paper as well on how to pass the the nodes of the graph individually into the GRU based RNN model that takes graph nodes sequentially, other propogation methods were attempted to get results.
* Two Ideas were tested with the same target of predicting the displacement of the Vertice given the Vertice with 256 node features.
* The first model tested was based on the idea proposed in the paper where a fully connected network was made that would take these 256 features and predict a class of the displacement among the labels of 0 to 224, this model was trained with cross entropy loss.
  * Something that was realized during this training was that treating this problem as a clasification problem may not be the best idea as there are no details given on how the 15x15 grid was converted to class labels, the method chosen for duplication was arbitrary to map each element of the grid into a class label. But a problem that arises here is that if the actual displacement was [7,7] for example, the loss function should be in such a way that a guess of [7,6] should be penalized much less than a guess of [0,0]. This is an issue with the choice of loss function, and the problem nature suggests to treat it as a regression problem instead where if Mean Squared Error is used as the loss function, the guess of [7,6] would be penalzied much less than the guess of [0,0]. 
* The second model was now made in such a way that the criterion was changed to Mean Squared Error from Categorical Cross Entropy and the labels were kept in the format of [A,B] instead of class labels. The output layer of the fully connected network now has 2 neurons to predict the X and Y displacement. 


## 3.3. Results

### 3.3.1 `Multi-task Segmentation Network`

< Add info here >
@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

### 3.3.2 `Polygon Refinement Network`
* As a building segment is passed through the encoder to get a 112x112x256 feature map, some of the feature layers of this feature map are illustarted below:
* The input image to the encoder is: 
  * ![image](https://user-images.githubusercontent.com/69632507/177661083-d5a55044-fc6c-4686-8620-4d448f33c1bb.png)
* Six randomly selected feature layers of the feature map
  * ![image](https://user-images.githubusercontent.com/69632507/177661218-eebf0c70-0509-4487-b5b4-33ebf4905468.png)
* Regarding the result of the First Propogation Model which was trained as a classfication model with Categorical Class Entropy with 225 different classes, the model quickly went to overfitting and memorizing the training data and no learning was happening on the testing data. As there are 225 classes to predict from, a random classifier would attain an accuracy of 0.44% accuracy and in this model's case, the training accuracy reached 88%, however, the testing accruacy is still fluctuating around the accuracy of a random classifier, suggesting no learning is happening and the model was interrupted early on.
  * ![image](https://user-images.githubusercontent.com/69632507/177661928-e9d7ae03-6d5d-4728-aac8-fa7a47819fce.png)
* Regarding the result of the Second Propogation Model which was trained as a regression model with Mean Squared Error, it was expected to perform better than the first model as there was more room to learn with the possibility of predicting displacements closer and closer to the ground truth. However, despite several iterations and efforts, the model was not learning as shown in the image illustarted below.
  * ![image](https://user-images.githubusercontent.com/69632507/177662271-8c4ac60f-5608-49da-9b16-192d369ec874.png)


# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
