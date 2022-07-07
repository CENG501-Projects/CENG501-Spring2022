# From Label Smoothing to Label Relaxation

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

In this report, the implementation of the study "From Label Smoothing to Label Relaxation, AAAI 2021" is presented. In the study, a new loss function called "Label Relaxation" is proposed as an alternative to the conventional "Label Smoothing", where it is performed on several networks using different datasets. Our goal is to create our own algorithm and reproduce the presented results.   

## 1.1. Paper summary

In this paper, they proposed a new loss function, label relaxation (LR) as an alternative to the conventional label smoothing (LS). They tried to prove the accuracy of the method they presented by comparing it with the previous methods. The proposed method, label relaxation is mainly compared with the label smoothing. In addition, cross entrophy (CE), confidence penalizing (CP), and focal loss (FL) are also used in order to evaluate the proposed method. Various datasets and learning architectures were also used within the scope of the study.

![image](https://user-images.githubusercontent.com/108774445/177691233-4ebdbea1-a33c-4f26-a68e-3e4dfcc015c3.png)


# 2. The method and our interpretation

## 2.1. The original method

A parametric evaluation is conducted in order to evaluate the accuracy of the proposed method. Various datasets and learning architectures were used within the scope of the study. Different loss functions such as cross entrophy (CE), label smoothing (LS), confidence penalizing (CP), and focal loss (FL) are used for the evaluation.

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

MNIST, FMNIST, CIFAR10, and CIFAR100 datasets are used in this study. MNIST and FMNIST datasets are used for 2-layer dense architecture, and the rest is used for VGG16, ResNet56(v2) and DenseNet-BC(100-12).
We imported the datasets using “Keras” and 1/6 of the training set was labeled as the validation set using "train_test_split". Then, we performed preprocessing by subtracting the mean from the inputs. The batch size is taken as 64 as stated in the article. We used Pytorch's Cross Entropy loss and implemented the Label Smoothing and Label Relaxation losses. In training phase, a code piece is taken from PyTorch examples and modified for our case.
We obtained results for 2-layer architecture using MNIST and FMNIST datasets. Moreover, architecture for VGG16 is also prepared and presented, however, the results could not be presented due to lack of time and computational power.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

The results of the implementation and original results are presented in the following tables. Table 1 and Table 2 are optimized for accuracy whereas Table 3 and Table 4 are optimized for ECE.


Table 1: Implemented Results on MNIST and Fashion-MNIST (Opt. for acc.)
|                    |  **MNIST**  |  **MNIST**   |**Fashion-MNIST**|**Fashion-MNIST**|
|--------------------|-------------|--------------|-----------------|-----------------|
|     **Loss**       |  **Acc.**   |   **ECE**    |    **Acc.**     |     **ECE**     |
|     CE (α=0)       |   0.983     |    0.016     |      0.875      |      0.094      |
|LS (α opt. for acc.)|   0.960     |    0.218     |      0.898      |      0.014      |
|LR (α opt. for acc.)|   0.972     |    0.017     |      0.901      |      0.029      |


Table 2: Original Results on MNIST and Fashion-MNIST (Opt. for acc.)
|                    |  **MNIST**  |  **MNIST**   |**Fashion-MNIST**|**Fashion-MNIST**|
|--------------------|-------------|--------------|-----------------|-----------------|
|     **Loss**       |  **Acc.**   |   **ECE**    |    **Acc.**     |     **ECE**     |
|     CE (α=0)       |   0.985     |    0.010     |      0.912      |      0.129      |
|LS (α opt. for acc.)|   0.988     |    0.106     |      0.915      |      0.155      |
|LR (α opt. for acc.)|   0.985     |    0.007     |      0.912      |      0.059      |
--------------------------------------------------

Table 3: Implemented Results on MNIST and Fashion-MNIST (Opt. for ECE)
|                    |  **MNIST**  |  **MNIST**   |**Fashion-MNIST**|**Fashion-MNIST**|
|--------------------|-------------|--------------|-----------------|-----------------|
|     **Loss**       |  **Acc.**   |   **ECE**    |    **Acc.**     |     **ECE**     |
|  CE (α=0, T opt.)  |   0.976     |    0.024     |      0.894      |      0.095      |
|LS (α opt. for ECE) |   0.982     |    0.017     |      0.875      |      0.024      |
|LR (α opt. for ECE) |   0.976     |    0.006     |      0.887      |      0.111      |


Table 4: Original Results on MNIST and Fashion-MNIST (Opt. for ECE)
|                    |  **MNIST**  |  **MNIST**   |**Fashion-MNIST**|**Fashion-MNIST**|
|--------------------|-------------|--------------|-----------------|-----------------|
|     **Loss**       |  **Acc.**   |   **ECE**    |    **Acc.**     |     **ECE**     |
|  CE (α=0, T opt.)  |   0.983     |    0.003     |     0.908       |      0.030      |
|LS (α opt. for ECE) |   0.987     |    0.014     |     0.915       |      0.016      |
|LR (α opt. for ECE) |   0.985     |    0.003     |     0.911       |      0.015      |

# 4. Conclusion

In this study, values close to the original results were obtained. Since the difference is in the acceptable range, it can be said that the method has been applied successfully. However, all analyzes could not be performed due to the lack of time and the long duration of the analyzes.

# 5. References

- Guo, C.; Pleiss, G.; Sun, Y.; and Weinberger, K. Q. 2017. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, August 6-11, 2017, volume 70 of Proceedings of Machine Learning Research, 1321–1330. PMLR.

- Lienen, J.; Hüllermeier, E. 2021. From Label Smoothing to Label Relaxation. Proceedings of the AAAI Conference on Artificial Intelligence, 35(10), 8583-8591. Retrieved from https://ojs.aaai.org/index.php/AAAI/article/view/17041

# Contact

- Sezer Mutlu: szrmutlu@gmail.com
- Tareq Rabaia: tareqrabai3a@gmail.com
