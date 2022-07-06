# Convolutional Neural Network Pruning with Structural Redundancy Reduction

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction
Our project paper, Convolutional Neural Network Pruning with Structural Redundancy Reduction by Wang et al., was accepted to CVPR 2021. In the paper, authors propose a filter pruning method, which highlights the importance of pruning filters from 'redundant' layers rather than pruning filters from the network as a whole. The main contribution of the paper is the method for computing structural redundancies of convolutional layers, where the proposed algorithm differs from and outperforms the state-of-the-art approaches [1-5]. As the dataset, the authors used CIFAR-10 and ImageNet ILSVRC-2012, with the well-known networks such as ResNet, AlexNet, or VGG16. Although the paper is reproducible as a whole, we aim to compare different prunning approaches on CIFAR-10 only, while using VGG16 as our network.

## 1.1. Paper summary

Compared to the existing works on network pruning that focus on removing the least important filters, Wang et al. proposed a method that prunes filters in the layers with the most structural redundancies [1-4]. To analyze structural redundancies of convolutional layers, authors represent each layer as a graph and use 1-covering-number along with the quotient space size. A graph with a higher 1-covering-number and larger quotient space size indicates less redundancy. After identifying the most redundant layer using the constructed graphs, the filters to be pruned are selected in that layer. Although there are different filter selection strategies [2], the authors used a simple strategy by pruning the filters with smaller absolute weights [1]. This pruning method resulted in 44.1% reduction in FLOPs while losing only 0.37% top-1 accuracy on ResNet50 with ImageNet ILSVRC-2012 dataset.   

# 2. The method and my interpretation

## 2.1. The original method

Initially, authors start by representing each convolutional layer as a graph so that they can find the layer with the most structural redundancy. To construct the graph, weights are flattened and normalized to change their lengths to 1, so that each filter is a unit vector. The filters _(of an individual layer)_ form the vertices of the graph and if the Euclidean distance between any two filters is smaller than a threshold γ, an edge is formed between the two vertices to represent the similarity between kernels. This algorithm forms an undirected, unweighted graph which may or may not be connected.

The redundancy value of a layer is then computed by utilizing quotient space size, and 1-covering-number of the graph. Since computinng the ℓ-covering-number is NP-Hard, authors estimated 1-covering-number to achieve reasonable computation complexities. A weighted average of _estimated_ 1-covering-number and quotient space size is used, where the weights are also parameters but the authors observed no drastic impact of values, and eventually used 0.65 and 0.35 in the favor of 1-covering-number _estimation_.

_High ℓ-covering-number and large quotient space size indicates less redundancy, i.e. intuitively, the filters in the layer can be considered as linearly independent._

With the identification of the most structurally redundant layer, the next step is to find the filters to be pruned. Here, different metrics can be used, yet the authors used a simple method to prune the filters with smaller absolute weights. The reasoning is provided as a theoretical proof in the paper. In this proof, five different prunings schemes and resulting accuracies are compared. In a setting with two layers where η is a much more redundant layer than ξ;

- p<sub>o</sub> denotes the accuracy with no pruning
- p<sub>ηr</sub> denotes the accuracy with a random pruning from layer η
- p<sub>η'</sub> denotes the accuracy with pruning the least important filter from layer η
- p<sub>ξ'</sub> denotes the accuracy with pruning the least important filter from layer ξ
- p<sub>g</sub> denotes the accuracy with pruning the least important overall filter

The relationship between these five accuracies are given as p<sub>ξ'</sub> ≤ p<sub>g</sub> ≤ p<sub>ηr</sub> ≤ p<sub>η'</sub> ≤ p<sub>o</sub>. However, with large enough n, where n denotes the filter number of η, p<sub>ηr</sub> ≈ p<sub>η'</sub> ≈ p<sub>o</sub>. Thus, the authors avoided calculating the least important filter and preferred a method that might yield a result similar to p<sub>ηr</sub> while improving the performance [1].

## 2.2. My interpretation 

The paper is well-written and there are not much problems with the explanations, however the filter selection is the most simplified aspect of the paper **apart from the missing Appendix**. Although the theoretical proof provides a good reasoning, the performance improvement by simplification of this part is still not clear. Also, there is no clear explanation of how many filters to prune at each iteration. Accordingly, we've decided to calculate the number of to-be-pruned filters based on the redundancy of a layer.

# 3. Experiments and results
## 3.1. Experimental setup

For our network, we preferred to use VGG16 instead of ResNet since adjusting residual connections after pruning might be complicated. With VGG16, a pruning on a convolutional layer only affects that layer and a single layer following it.

Since the [VGG16 model](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html) that Torch provides is pre-trained and uses an MLP head suitable for ImageNet dataset, we've decided to implement our own VGG16 that is suitable for CIFAR-10. We've then trained the network on CIFAR-10 for 50 epochs with:
- CrossEntropy loss
- SGD with momentum (=0.9)
- Batch size of 256, and learning rate of 1e-2

**Although we didn't have enough resources/time to tune the network to the optimum, it provided 77.7% accuracy on the test set, which is a reasonable baseline performance to compare, without any pruning.**

We started by implementing a graph construction function. Here we used flattening and normalization similar to that in the paper. For the similarity threshold γ, we manually tested a few different values and decided to use 0.02, which is in range of the values used in the paper. Using weighted average of ℓ-covering and quotient space size we found the layer with the most redundancy. We used same weights as the authors (0.65 for ℓ-covering and 0.35 for quotient space size). 

After finding the layer, we used single shot pruning as mentioned in the paper. As the number of filters to prune is not specified in the paper, we came up with our own formulation at this step. We pruned number of filters equal to $2 * \sqrt{r}$, where $r$ denotes the redundancy of the layer. After each pruning step, we reconstructed the graph and calculated the redundancy only for the pruned layer, since the graph construction is the most costly part. 

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

Implemented Paper: Zi Wang, Li Chengcheng, and Wang Xiangyang. Convolutional neural network pruning with structural redundancy reduction. 
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. arXiv:2104.03438, 2021.

[1] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and
Hans Peter Graf. Pruning filters for efficient convnets. arXiv
preprint arXiv:1608.08710, 2016.

[2]  Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila,
and Jan Kautz. Pruning convolutional neural networks for resource efficient inference. arXiv preprint arXiv:1611.06440,
2016.

[3] Xiaohan Ding, Guiguang Ding, Yuchen Guo, Jungong Han,
and Chenggang Yan. Approximated oracle filter pruning for destructive cnn width optimization. arXiv preprint
arXiv:1905.04748, 2019.

[4] Yang He, Ping Liu, Ziwei Wang, Zhilan Hu, and Yi Yang.
Filter pruning via geometric median for deep convolutional
neural networks acceleration. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition,
pages 4340–4349, 2019. 

[5] Chaoqi Wang, Roger Grosse, Sanja Fidler, and Guodong
Zhang. Eigendamage: Structured pruning in the kroneckerfactored eigenbasis. arXiv preprint arXiv:1905.05934, 2019.

# Contact

Mustafa Duymuş (mduymus@ceng.metu.edu.tr) <br />
Erce Guder     (guder.erce@gmail.com)
