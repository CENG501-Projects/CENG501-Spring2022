# @TODO: Paper title

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction
Our project paper, Convolutional Neural Network Pruning with Structural Redundancy Reduction by Wang et al., was accepted to CVPR 2021. In the paper, authors propose a network prunning method, which improves the performance of CNNs with removing filters from unnecessery layers. The main contribution of the paper is the method of filter selection for prunning, where the proposed algorithm differs from and outperforms the state-of-the-art approaches [1-5]. As the dataset, the authors used CIFAR-10 and ImageNet, with the well-known networks such as ResNet or AlexNet. Although the paper is reproducible as a whole, we aim to compare different prunning approaches on CIFAR-10 only, while using VGG16 as our network.

## 1.1. Paper summary

Compared to the existing works on network prunning those focus on removing the least important filters, Wang et al. proposed a method that pruns a filter in the layer with the most structural redundancy [1-4]. To analyze the structural redundancy of the layers, authors represent each layer as a graph and use ℓ-covering along with the quotient space. A graph with a higher ℓ-covering number and larger quotient space size indicates less redundancy. After identifying the most redundant layer using the constructed graphs, the filters to be pruned must be selected in that layer. Although there are different filter selection strategies [2], the authors used a simple strategy by pruning the filters with smaller absolute weights [1]. This pruning method resulted in 44.1% reduction in FLOPs while losing only 0.37% top-1 accuracy on ResNet50 with ImageNet dataset.   

# 2. The method and my interpretation

## 2.1. The original method

Initially, authors start by representing each layer as a graph so that they can find the layer with the most structural redundancy. For constructing the graph, weights are flatten and normalized to change their lengths to 1, so that each filter is unit vector. The filters form the vertices of the graphs and if Euclidean distance between two different filters is smaller than a threshold γ an edge is added between the two vertices representing those filters. This forms an undirected, unweighted graph which may or may not be connected. By calculating ℓ-covering and quotient space values the authors find the most structurally redundant layer. Since calculation of ℓ-covering is NP-Hard, authors simply selected ℓ = 1 or ℓ = 2 to achieve a reasonable computation time. High covering number can be presumed as most of the filters in that layer are linearly independent.

With the identification of the most structurally redundant layer, next step is finding the filters to be pruned. Here, different metrics can be used, yet the authors use a simple method of pruning the filters with smaller absolute weights. The reasoning is provided as a theoretical proof in the paper. In this proof, five different prunnings and resulting accuracies are compared. In a setting with two layers where η is a more redundant layer than ξ;

- p<sub>o</sub> denotes the accuracy with no pruning
- p<sub>ηr</sub> denotes the accuracy with a random pruning from layer η
- p<sub>η'</sub> denotes the accuracy with pruning the least important filter from layer η
- p<sub>ξ'</sub> denotes the accuracy with pruning the least important filter from layer ξ
- p<sub>g</sub> denotes the accuracy with pruning the least important overall filter

The relationship between these five accuracies are given as p<sub>ξ'</sub> ≤ p<sub>g</sub> ≤ p<sub>ηr</sub> ≤ p<sub>η'</sub> ≤ p<sub>o</sub>. However, with large enough n, where n denotes the filter size of η, p<sub>ηr</sub> ≈ p<sub>η'</sub> ≈ p<sub>o</sub>. Thus, the authors avoided calculating the least important filter and preferred a method that might yield a result similar to p<sub>ηr</sub> while improving the performance. 

## 2.2. My interpretation 

The paper is well-written and there are not much problems with the explanations, however the filter selection is the most simplified aspect of the paper. Although the theoretical proof provides a good reasoning, the performance improvement by simplification of this part is still not clear. Also there is not a clear explanation of how many filters are pruned at each iteration. As a result, we preferred to select to-be-pruned filters randomly based on the proof and the relationship p<sub>ηr</sub> ≈ p<sub>η'</sub>. Furthermore, we preferred to prune a single filter at each iteration.

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

Mustafa Duymuş (mduymus@ceng.metu.edu.tr) 
Erce Guder     (guder.erce@gmail.com)
