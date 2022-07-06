# From Label Smoothing to Label Relaxation

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

This study is an implementation of the paper "From Label Smoothing to Label Relaxation" from AAAI 2021. The authors are Julian Lienen and Eyke Hüllermeier. An alternative loss function called "label relaxation" is proposed as an alternative to the "label smoothing". The proposed function is tested on several networks using different datasets. Our goal is to create our own algorithm and reproduce the presented results.

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and my interpretation

## 2.1. The original method

@TODO: Explain the original method.

## 2.2. My interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

MNIST, FMNIST, CIFAR10, and CIFAR100 datasets are used in this study. MNIST and FMNIST datasets are used for 2-layer dense architecture, and the rest is used for VGG16, ResNet56(v2) and DenseNet-BC(100-12).

We obtained results for 2-layer architecture using MNIST and FMNIST datasets. Architectures for VGG16 and ResNet56(v2) were also prepared and presented, however, the results could not be presented due to lack of time and computational power.

We imported the datasets using “keras”. After the data was imported, the training and test set were combined and augmentation was applied by creating symmetrical images. After augmentation, 1/7 of the whole data was labeled as the test set and 1/6 of the rest was labeled as the validation set using "train_test_split". To train the network, the batch size is taken as 64 as stated in the article.

|                    |  **MNIST**   |  **MNIST**    |        **Fashion-MNIST**        | **Fashion-MNIST**   |
|--------------------|--------------|----------------|---------------------------------|---------------------|
|     **Loss**       |   **Acc**    |    **ECE**     |            **Acc**              |       **ECE**       |
|--------------------|--------------|----------------|---------------------------------|---------------------|
|     CE (a=0)       |              |                |                                 |                     |
|LS (a opt. for acc.)|              |                |                                 |                     |
|LR (a opt. for acc.)|              |                | $LR=0.005 , Decay Rate = 0.99$  |                     |


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
