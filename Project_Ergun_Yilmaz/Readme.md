# Paper Title: Deep Metric Learning with Self-Supervised Learning

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper “Deep Metric Learning with Self-Supervised Learning” was published at the AAAI21 conference. It basically aims to provide Deep Metric Learning networks with better performance so that it can learn deep embedding space better. In the paper, it is proposed that learning not only inter-class variance, but also intra-class variance can be exploited for this purpose. Moreover, it was very useful to reproduce the results and see the effects of intra-class variance within the scope of the CENG501 course.        

## 1.1. Paper summary

Main idea of the paper is that existing deep metric learning approaches only search for pairwise similarities and seek higher margin of different class neighbors in the embedding space, however they dismiss the local structures of the same class-negihbors and count them totally same. Consequently, it is claimed that existing methods in the literature learn the embeddings in a less effective way. They offer novel auxiliary framework which can be integrated with any existing metric learning network. It is also shown that by using this novel technique retrival and ranking performances of the state-of-the-art approaches increases by 2%-4%.  

# 2. The method and my interpretation

## 2.1. The original method

The original method first introduces the deep metric learning setup where auxilary framework is to be integrated. Pretrained ResNet50 with imagenet and replacing its last layer with randomly initiliazed fully connnected layer are preferred to accomplish this task.  On the other hand, auxiliary framework is constructed by conecing MLP with one hidden layer to the called backbone network. This backbone network is the same former ResNet50. In other words, a common network is constructed and used as the backbone network, but two different branches are splitted for main deep metric learning and auxilary network tasks.

  But the most importantly, auxiliary network is trained in a self supervised manner. Simulative transformations with specific strength on the provided dataset are used to generate syntethic dataset for this network to exploit intra-class variance. The main idea is that higher the strength is, the distance between the original image and its transformed versions should be larger. Accordingly, loss function is defined to influence this idea to the network.
Moreover, this offered method brings computational overhead and to cope with the exponetial increase in the computationla complexity, two suggesstions are made. First, number of different transformations used for syntethic dataset generation is limited to four. Secondly, execution of the auxiliary framework is performed with some probability while training.  


## 2.2. My interpretation 

There are also few points which is unclear for ablation studies. Firstly, usage of batch and mini batch conncepts in the paper is the most confusing. Two different values are mentioned in the paper under the names of mini-batch and batch. However, there is a single minibatch shown in the pseudoalgorithm. We prefer to use higher valued batch definition to train the general embedding framework and perform random sampling as figured and use the other one to form self supervied learning input. Secondly, there are also some missing definitions about training parameters. Number of epeoch to train whole network is absent and there is only one explicit initial learning rate definition and nothing about how to schedule it. How to use them will be mentioned in the following section. Finally, there is also confusing point about self supervised procedure. How to select simulative transformation parameters is unclear and we leave them as a parametric by defining transformation strength.     

# 3. Experiments and results

## 3.1. Experimental setup

Our team starts to build experimental setup by adapting Easily Extendable Basic Deep Metric Learning Pipeline taken from https://github.com/Confusezius/Deep-Metric-Learning-Baselines. Then we have removed and modified some parts and added some extra functionality to have experimental setup as described in the Paper. As a result, explicitly defined all parameters about experimantal environment in the paper are involved in our implementation as they are. Epoch secimi ve learning rate secimi
However, due to limited time and resources, we focus on some test scenarios only. We could test our implementation only using CUB200 dataset and margin loss and inspect the metrics namely recall@K, normalized mutual information score(NMI) and F1. In the paper, there are also two extra datasets and losses whose results are shared. In addition, we have covered all the metrics provided in the paper about retrieval results, but we have changed ranking metric from R-Precision, Precision Recall Curve and Mean Average Precision to F1.     


## 3.2. Running the code

Our main script is contained in Standard_Training.py file which is adapted from the url given above. Before running this file, datasets should be downloaded. Various arguments controlling the hyperparameters of the model and loss functions can be passed to this file while running. If not specified, default values are used. Standard_Training.py creates and trains the models described in “Deep Metric Learning with Self-Supervised Ranking” for a specified number of epochs and logs the metric loss, ranking loss and performance metrics to Tensorboard during training. Once the training completes, trained models are saved to specified directory for inference and reuse. In order to visualize the results via Tensorboard run the following command in terminal after completion:
tensorboard –-logdir= <tensorboard_save_path>

## 3.3. Results

After training 15 epochs, the resulting metric we have obtained with and without auxiliary network are tabulated as follows.
Metric	With Auxiliary Network	Without Auxiliary Network
Recall@1	0.5944	0.5908
Recall@2	0.7130	0.7208
Recall@4	0.8118	0.8172
Recall@8	0.8891	0.8864
NMI	0.6679	0.6621
F1	0.3618	0.3511

Note that, we have evaluated and verified our model by looking the relative difference between these two results. Moreover, the results in the paper are higher than the peresented results here, but they can be accessible by training the model longer with appropriate tuning of confusing parts. The loss curves for metric loss and what the paper called ranking loss can be seen below to show that our model can be able to decrease ranking loss also.

![image](https://user-images.githubusercontent.com/56062641/177702265-0a20ca15-037f-4f94-acc0-9ae27e705279.png)
![image](https://user-images.githubusercontent.com/56062641/177702290-040c881d-e71c-4e49-906e-4d27b7d13dac.png)




# 4. Conclusion

After several works on the model implementation, the slight increase in the metrics with auxiliary framework addition seen in the table 1 shows us that our implemented model works when training with 15 epochs. Moreover, when we increase epoch number, we notice that metric learning loss approaches to 0 as seen in the figure 3 and 4, and network without auxiliary framework seems to work well. However, we consider this result unreliable. Because model gets overfitted with the setup our implementation has for the higher epochs. Since everything is almost the same as in the paper, we thought that tuning of transformation strength and learning rate scheduling play important role for this overfitting issue. As we have limited resources, we can not be able to make several trials with higher epochs to avoid overfittinng with playing the parameters. Also increase caused by auxiliary framework can be further improved with the tuning mentioned here.

![image](https://user-images.githubusercontent.com/56062641/177702312-a9b55fa7-25fb-40a0-b80f-f8a51ee81100.png)

![image](https://user-images.githubusercontent.com/56062641/177702317-b9f4800c-2e4d-45c8-8eb6-c624c9884ebf.png)

![image](https://user-images.githubusercontent.com/56062641/177702338-09786e7e-45c4-401d-9088-da7a4cfd88bc.png)

The paper we covered here comes up with the concrete idea that can be easily integrated to existing deep metric learning algorithm. This is the most attention-grabbing features of the paper proposals. Besides, it is written in an understandable way so that people from other disciplinary can benefit from it. These can be considered as an advantagaes. On the other hand, some parts could create confusion as explained before and should have managed carefully. Moreover, training auxiliar framework intorduces somewhat significant overhead despite the suggestions of the paper. At this point, we have also some remarks. If this intra-class variance extraction can be handled without defining new loss and inserted already used metric learning losses, this overhead can be avoided. Also, another remark can be freezing some initial layers of the backbone network and avoiding to gradient flow there. Because generally, the first layers of the pretrained CNN models are responsible to extract general features. This freezing operation is added to our implementation with a run parameter called num_frozen_submodules.


# 5. References

[1] Fu, Z., Li, Y., Mao, Z., Wang, Q., & Zhang, Y. (2021). Deep Metric Learning with Self-Supervised Ranking. Proceedings of the AAAI Conference on Artificial Intelligence, 35(2), 1370-1378. Retrieved from https://ojs.aaai.org/index.php/AAAI/article/view/16226

# Contact

Musa Selman ERGUN
selman.ergun@metu.edu.tr

Onur YILMAZ 
yilmaz.onur_01@metu.edu.tr
