# Task Aligned Generative Meta-learning for Zero-shot Learning
This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction
In this project, a novel method of zero-shot learning as described in the paper [1] is attempted to be implemented(*). The paper was published in [AAAI-21](https://aaai.org/Conferences/AAAI-21/).

(*): The paper uses certain architectures that do not have public implementations:
* a type of generative adversarial network called Meta-conditional Generative Adversarial Network (MGAN)
* a type of autoencoder called Task-adversarial AutoEncoder (TAE)

Moreover, the papers uses four benchmark datasets: AWA1, AWA2, aPY and CUB.
* [AWA1](https://paperswithcode.com/dataset/awa-1) is retracted due to copyright issues.
* [AWA2](https://paperswithcode.com/dataset/awa2-1) is available.
* [aPY](https://paperswithcode.com/dataset/apy) has parts from Yahoo and Pascal, Pascal images are unavailable.
* [CUB](https://paperswithcode.com/dataset/cub-200-2011) is entirely unavailable.

Thus,
1. We will be working with AWA2.
2. We will be using approximations of the architectures to the top.

Therefore, we will not be able to perfectly replicate the paper's results.

## 1.1. Paper summary
The paper described a novel method of zero-shot learning in classifier models. Zero-shot learning is the problem of being able to classify instances from unknown classes that were absent in the training set.

Existing methods [2-4] for zero-shot learning tries to predict possible new classes using features of the existing classes. This method can introduce a bias towards existing classes and the distribution of their features.

The method in this paper tries to solve this problem by training the network in tasks and introducing another network to align the tasks to a uniform distribution. Hence, reducing bias.

Moreover, there are uses of adversarial networks to synthesize examples of unseen classes to be used for training. Note that even if we do not have any examples of those classes, the paper requires us to know their labels and attributes.

# 2. The method and my interpretation
## 2.1. The original method

The papers proposes two phases of training. 

The first phase is task distribution alignment. In this phase, a special type of autoencoder is trained together with a discriminator and a classifier. Purpose of the autoencoder is to embed the input data in the embedding space in a uniform manner. This is achieved by the discriminator inventing pseudo-labels for the encodings, so that the encoder spreads out the embeddings. Classifier helps by contributing an objective to the loss functions in this phase. Later, the classifier weights are transferred to the second phase classifier.

The second phase is generative zero-shot learning. In this phase, a special type of GAN is trained together with a classifier. The input data is passed through the autoencoder to get their encodings and the encodings are used in this stage. The generator produces encodings from attribute and noise vectors. The discriminator tries to discern by looking at an encoding and its supposed attribute vector. The classifier only gets the encoding. There is a meta-learning aspect here that optimizes the GAN and the classifier.

Finally, the training is done through tasks in an episodic manner. Data is split into classes, seen and unseen. Examples of unseen classes only show up in test time. The paper splits the training data into support and query disjoint subsets. Examples in the query subsets are known but we act as if they are unseen so that the model can be trained to do zero-shot classification.

## 2.2. My interpretation

I had to check a lot of the reference material and existing literature to have an idea about what is happening in the paper. It appears that most of the process has been established in other papers. The two main novelties are the special types of autoencoder and GAN uses. Both of which lack adequate material or source code.

Moreover, the architecture was very advanced for me. I did not have to deal with one module feeding into multiple modules and the modules playing a minimax game between themselves before.

I dealt with these problems by approximating and simplifying them. I used standard autoencoders and GANs to account for the lack of knowledge about the special types used here. I also trained modules separately where they were supposed to be trained together.

Overall, this simplified things but reduced accuracy.

# 3. Experiments and results
## 3.1. Experimental setup

The original paper provides results on four benchmark datasets along with comparisons with other state-of-the-art models. To repeat, the datasets are AWA1, AWA2, aPY and CUB. The only available one is AWA2. The experiments run on features extracted from AWA2 through the use of a pretrained ResNet-101 model. In this implementation, I use the same data.

AWA2 comes with extracted features, labels for 50 classes and 85 attributes. Briefly, Animals With Attributes 2 (AWA2) dataset is a collection of images with a table of animals to animal features such as black, white or brown.

As I mentioned, I had to alter the architecture. Therefore, the experiments are only an approximation.

## 3.2. Running the code

To run the code, please run the provided Jupyter Notebook. It should download the data and run the experiments as you run the cells. There are some constants you can alter to tune the models.

## 3.3. Results

Please refer to the Jupyter Notebook for figures.

In words, the method was somewhat successful. Final analysis of the zero-shot classifier has shown that examples of unseen classes perform about as well as examples of the seen classes in my simplified implementation. The more comprehensive, original method should perform better.

# 4. Conclusion

The paper proposes a multi-faceted architecture for zero-shot learning. According to the benchmarks, the architecture outperforms other existing zero-shot learning models. This experiment I have done shows that it is feasible. However, there are many steps to the training and the models involved are large. So, there is a big computational cost.

Overall, I believe the paper provides a promising method.

# 5. References
[1]: [Liu, Zhe, et al. "Task aligned generative meta-learning for zero-shot learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 10. 2021.](https://ojs.aaai.org/index.php/AAAI/article/download/17057/16864)

[2]: [Romera-Paredes, B.; and Torr, P. 2015. An embarrassingly simple
approach to zero-shot learning. In International Conference on
Machine Learning, 2152–2161.](https://proceedings.mlr.press/v37/romera-paredes15.html)

[3]: [Zhang, L.; Xiang, T.; and Gong, S. 2017. Learning a deep em-
bedding model for zero-shot learning. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition, 2021–
2030.](https://www.semanticscholar.org/paper/Learning-a-Deep-Embedding-Model-for-Zero-Shot-Zhang-Xiang/c4f67bb310ef9b5dca6623d3aa890182d9e828e7)

[4]: [Liu, S.; Long, M.; Wang, J.; and Jordan, M. I. 2018a. Generalized
zero-shot learning with deep calibration network. In Advances in
Neural Information Processing Systems, 2005–2015.](https://papers.neurips.cc/paper/2018/file/1587965fb4d4b5afe8428a4a024feb0d-Paper.pdf)

# Contact
Mert Alp Taytak (mert.taytak@metu.edu.tr)
