# @TODO: Paper title

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction
Our project paper, Convolutional Neural Network Pruning with Structural Redundancy Reduction by Wang et al., was accepted to CVPR 2021. In the paper, authors propose a network prunning method, which improves the performance of CNNs with removing filters from unnecessery layers. The main contribution of the paper is the method of filter selection for prunning, where the proposed algorithm differs from and outperforms the state-of-the-art approaches [1-5]. As the dataset, the authors used CIFAR-10 and ImageNet, with the well-known networks such as ResNet or AlexNet. Although the paper is reproducible as a whole, we aim to compare different prunning approaches on CIFAR-10 only, while using ResNet as our network.

## 1.1. Paper summary

Compared to the existing works on network prunning those focus on removing the least important filters, Wang et al. proposed a method that pruns a filter in the layer with the most structural redundancy [1-4]. To analyze the structural redundancy of the layers, authors represent each layer as a graph and use ℓ-covering along with the quotient space. A graph with a higher ℓ-covering number and larger quotient space size indicates less redundancy. After identifying the most redundant layer using the constructed graphs, the filter to be pruned must be selected in that layer. Although there are different filter selection strategies [2], the authors used a simple strategy by pruning the filters with smaller absolute weights [1]. This pruning method resulted in 44.1% reduction in FLOPs while losing only 0.37% top-1 accuracy on ResNet50 with ImageNet dataset.   

# 2. The method and my interpretation

## 2.1. The original method

To analyze the structural redundancy of the layers, authors represent each layer as a graph and use l-covering along with the quotient space. For the graph, weights are flatten and normalized to change their lengths to 1, so that each filter is unit vector. Euclidian distance 

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

@TODO: Provide your names & email addresses and any other info with which people can contact you.
