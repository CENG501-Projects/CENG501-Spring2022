# OSOP: A Multi-Stage One Shot Object Pose Estimation Framework

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

In computer vision, labelled datasets are expensive to create, often contains a small subset of real data. This problem is more prominent in 6D Pose estimation problems since it is tricky to generate grounf truth data. Due to these limitations, new one-shot or few shot methods are emerging in the literature. These methods try to learn generalizations about the task and using the learned representations, they try to accomplish their task by either zero training or really small size training. 

Our paper[1] which was published in CVPR2022 claims to be state of the art in one shot 6D pose estimation problems. In this problem, you are given a 3D model of an object and network is asked to localize the object and estimate the 6D pose of the object. They provide results on popular datasets found in BOP benchmark[2] challenge like Linemod, TLESS. We tried to reproduce the results of the paper on TLESS dataset.

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

Main purpose of the paper is to present a method for solving one-shot 6D Pose Estimation problem. One-shot means that after initial training, there is no retraining and method tries to generalize well to all unseen objects. In literature, really few one-shot pose estimators are present. So this paper extends existing one-shot object detection methods for pose estimation. 

In order to solve this problem, authors have built upon the existing literature of one-shot Object segmentation and 2D-3D Correspondence problems. Authors have proposed a multi-stage network architecture that consists of 4 layers. The proposed framework can be seen in Figure 1. First layer is responsible for generating the one-shot object segmentation map. Here, authors have proposed improvements over the existing correlation based 2D segmentation methods where a pixelwise attention mechanism was also proposed. Also, full 3D information about the object was exploited. Second stage of the network is the template matching network where they increased the latent vector size of a existing encoder-decoder based template matcher. In third stage, given the template and the segmentation mask, 2D-3D Correspondences are estimated. This network is similar to the existing literature. In fourth stage, existing classical CV methods such as ICP,PnP are applied to refine the outputs of the third stage.

# 2. The method and my interpretation

## 2.1. The original method

@TODO: Explain the original method.

In the original method, first stage network is similar to the existing one-shot object detectors such as Os2D[3]. The idea is to render a dictionary of poses from the given 3D object model and extract features from those views using existing feature extractors. The Encoder Architecture can be seen in Figure 2. The dictionary consists of 2880 predefined views which corresponds to 16 elevation angle, 36 azimuth angle and 5 in plane rotations. The Feature computation method can be seen in Figure 3. In the paper, ResNet50 was used as a feature extractor in all parts of the network. Features from 10,22,40th layers are extracted with sizes 120X160X256, 60x80x512, 30x40x1024. These features are then averaged in spatial coordinates to get feature vectors of size 256,512,1024. So at the end, we have three tensors of sizes 16x36x5x256, 16x36x5x512, 16x36x5x1024. These tensors are called $o_1$, $o_2$, $o_3$. Features of the reference image are also extracted from these layers as well called $f_1$, $f_2$, $f_3$. These feature tensors and image features are then fed into Attention Block. This block calculates the correlation between a feature tensor and image feature vector. Output of this block is processed using convolutions and then fed back into the rest of the ResNet. Dice Loss was used to train the network.


Second stage network part does not involve a lot of information. We were only able to infer that they were using a network similar to a network from their reference list. Because of that, we are not really sure about the full architecture.

In third stage, a dense corresponce is calculated using a network with architecture given in Figure. This network is said to be trained using Dice loss and per pixel cross entropy classification loss.

Fourth stage uses the data from the third stage to make refinements using PnP,ICP algorithms. There are no further specifics of this part.

## 2.2. My interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

For the first stage, we have used renderer libraries to generate the required feature tensors. However, loading the tensors into the memory of the GPU proved to be impossible for the GPU's that we owned(~27GB). Because of this reason, we have designed the full architecture and verified that it was able to do forward pass and backwards pass without problem however we did not train our first stage network. A rendered image sample can be seen in Figure 4.

For the second stage, we have decided to implement the method given in [4]. This network is called variational autoencoder network where given an image, it tries to isolate the query object in the image. In order to train the network, we have used the TLESS dataset to get object data. We use domain randomization techniques described in [4]. Specifically, we take a random background image from VOC Pascal dataset and set it as the background of the object image. After this step, a series of random blur, noise, affine transformations etc are performed on the image to generate a random image. Network is then asked to reconstruct the original image given the randomized image. We use a loss function called Bootstrapped L2 Loss first defined in [5]. To calculate this loss, we first perform the standard L2 loss. After this step, we only take the top 100/k% of the pixels contributing to the loss. For example, if k = 4, only 25% of the pixels with the highest loss is considered in the L2 loss. Loss for the other pixels are discarded. We used a latent size of 128. We used broadcast factor of 8 in the implementation. A sample input-output pair is given in Figure 6.

For the third stage, we have prepared the backbone of the architecture however we could not implement it fully. Reason is that paper claims that they are using the same attention block used in the first layer however this causes shape inconsistencies in the network provided by the authors. We could not come up with a reasonable explanation for the attention block so we could not proceed onto the training.

Fourth layer had no implementation since it was only used for refinement and post-processing.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

Original paper did not provide really detailed experimental setup. There were several ambiguities and incorrect architecture explanations in the paper.
For the first stage, we followed the descriptions given in the paper to implement the model. The first stage is comprised of a network with a resemblance to a U-Net. The authors have used Attention Blocks and Pearson correlation as described in the paper.
We have implemented these custom blocks as nn.Modules and built the architecture based on the descriptions. For training, we used Adam Optimizer with lr=2e-4 and Dice Loss, a loss function we implemented. The reason for Dice Loss is to handle imbalanced class data as described in the paper. However, our hardware was not capable of handling the feature vectors o_k. We have faced Cuda memory problems along the training procedure. Therefore, the training was not completed.
For the second stage, we have mostly used the setup of the respective paper. We use Adam Optimizer with lr=2e-4. Paper trained for 30k iterations. We have trained for 150k iterations since it took overnight to reach these numbers. The loss curve can be seen in Figure 5.



## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.
The required python packages to run the code is given in requirements.md file. Creating a conda virtual environment is highly recommended.
In order to run each of the stages described in the paper, you need to find the corresponding stage_<stage_no>.ipynb file and execute the required cells.
The code is separated into execution blocks in Jupyter notebooks and models, auxillary modules in Python scripts. The custom models implemented for the paper are given in stage_<stage_no>_models.py file, whereas the helper modules are located in their corresponding scripts.


## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.
![Framework](/assets/framework.png "Framework")
![Encoder Architecture](/assets/encoder.png "Encoder Architecture")
![Feature Computation](/assets/feature_comp.png "Feature Computation")
![Rendered Image](/assets/image_438.png "Rendered Image")
![Loss curve for Stage 2](/assets/loss_history_stage2.png "Loss History Curve")
![Merged Input/Output](/assets/merged_io.png "Merged Input/Output")

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

We were succesful in the implementation of the second stage. We believe that paper was poor at explaining the specifics of the architecture. One-shot pose estimation is a complicated problem with a lot of details. Training a network for any part of the architecture is tricky with a lot of local minimas and other problems. Experimental setup was poorly explained, architecture explanations had faults in it. A complicated network structure and training procedure in third stage was summarized in half a page. We feel sorry that despite our sincere efforts, we were not able to accomplish our initial goals. 

# 5. References

@TODO: Provide your references here.

[1] Ivan Shugurov, Fu Li, Benjamin Busam and Slobodan Ilic. OSOP: A Multi-Stage One Shot Object Pose Estimation Framework. In CVPR, 2022
[2] Tomas Hodan and Antonin Melenovsky. Bop: Benchmark for 6d object pose estimation: https://bop.felk.cvut.cz/home/, 2019
[3] Anton Osokin, Denis Sumin, and Vasily Lomakin. Os2d: One-stage one-shot object detection by matching anchor features. In ECCV, 2020
[4] Martin Sundermeyer, Maximilian Durner, En Yen Puang, Zoltan-Csaba Marton, Narunas Vaskevicius, Kai O Arras and Rudolph Triebel. Multi-path learning for object pose estimation across domains. In CVPR, 2020.
[5] Wu Z, Shen C, Hengel Avd (2016) Bridging category-level and instance-level semantic image segmentation. arXiv preprint arXiv:160506885
# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
Erhan Ege Keyvan - ege.keyvan@gmail.com
Şahin Umutcan Üstündaş - ustuntasumutcan@gmail.com
