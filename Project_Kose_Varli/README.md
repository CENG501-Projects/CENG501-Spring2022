# SMG: A Shuffling Gradient-Based Method with Momentum

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

This is an unofficial implementation of the paper "SMG: A Shuffling Gradient-Based Method with Momentum" from ICML 2021. The paper proposes an alternative scheme for the non-convex finite-sum optimization problem. While most of the necessary experimentation details are given as a search grid, we did not have enough computation power to perform the grid search. Moreover, details like batch sizes and normalization of the data-set were not given, so we had to determine them ourselves. Hence, we focused on obtaining the results whose trends match as much as possible with the paper's. 

## 1.1. Paper summary

The paper proposes an alternative optimization scheme based on batch shuffling and momentum techniques. The method is inspired by existing momentum techniques, but its update fundamentally differs from the existing momentum-based methods. Moreover, the authors claim that they are the first to analyze convergence rate guarantees of shuffling-type gradient methods with momentum under standard assumptions. Besides, the paper proposes a single shuffle variant of the algorithm. The technique shows encouraging performance among existing shuffle-based methods.

# 2. The method and our interpretation

## 2.1. The original method

SMG: The exact algorithm will be given below. Thus, we will be focusing on what makes the scheme different from existing shuffle-based momentum techniques. Unlike existing momentum methods where the momentum is updated recursively in each iteration, the method fixes the momentum at the beginning of each epoch. The momentum is only updated at the end of each epoch by averaging all the gradient components evaluated. To avoid storing the gradients, an auxiliary variable v is introduced, which keeps track of the gradient average. Lastly, in the algorithm, there is a hyperparameter   $\beta $   which is fixed, but in the paper, it is stated that it is also possible to make $\beta$ adaptive. The exact algorithm can be seen below.


![SMG](https://user-images.githubusercontent.com/44121631/177314075-06f40c29-65a0-4c2a-9e16-772615d465be.png)



SSMG: Again the exact algorithm will be given below. SSMG is the single shuffle variant of SMG. In SMG, the batch is reshuffled at the beginning of each epoch, but in SSMG, the dataset is shuffled only once at the beginning of the training and held fixed throughout the training. Unlike SMG, SSMG directly incorporates the momentum term in each iteration and updates it recursively. The rest of the algorithm is similar to SMG. The exact algorithm can be seen below.



![SSMG](https://user-images.githubusercontent.com/44121631/177314110-b62ab3e0-a73f-4772-8bdd-f7678cbd4865.png)


## 2.2. Our interpretation 

We had no problem implementing the algorithm. As stated in the introduction part, the part that is not clearly explained is the experimental setup. In the experimental setup, there are four different datasets that have been tested on different neural networks. For Cifar-10 and Fashion-MNIST the networks were stated clearly, so we made no assumptions. For w8a and ijcnn1, the authors have conducted a nonconvex logistic regression. Since the network they used is not clearly stated, we have used a single layer fully connected network with activation function tanh to conduct the experiments on these datasets. 

Dataset normalization can have a significant impact on performance. The authors did not state how they normalized the datasets in the paper. Therefore, we have tried different normalizations and inspected the results. For the Fashion-MNIST dataset, we have taken normalization values from an open-source implementation which can be found in the references part.

Like in the normalization part, the batch sizes were not given. Therefore, we have tried several batch sizes and examined the results in terms of speed and performance. 

As stated in the introduction, the essential hyperparameter setup for optimizers is given as a search grid. The authors did not clearly state which parameter worked best for them. While this does not look like a problem, at first sight, it is expensive because the chosen dataset-architecture pairs start showing their actual performance after 150-170 epochs. Considering the results shown for 200 epochs in the paper, the grid search means training the network fully for each combination of hyperparameters. Unfortunately, it was impossible for us because we did not have enough computational power to do that. Therefore, we did a logical grid search by hand by analyzing tradeoffs between the hyperparameters. We believe that we have chosen the best hyperparameters in the setup we have configured.

# 3. Experiments and results

## 3.1. Experimental setup

They conduct numerical experiments on four different datasets and 3 networks. Convolutional LeNet-5 is used to train CIFAR-10 dataset with 50,000 samples, classic LeNet-300-100 model is used to train Fashion-MNIST with 60,000 images and w8a (49,749 samples), ijcnn1 (91,701 samples) classification datasets are trained on a network that implements nonconvex logistic regression. As we discussed in the previous sections, the architecture of nonconvex logistic regression model is not defined precisely. Thus, we have used a single layer fully connected network with activation function tanh to conduct the experiments on these datasets.

Their main approach to conduct experiments is that they repeatedly run experiments for 10 random seeds and report the average results. Also, the essential hyperparameter setup for optimizers is given as a search grid. They select the parameters that performs best for the datasets. There is no clear definition of what they mean by best performing and according to our understanding that they select the best performing model by hand from the plots. 

As we discussed in the section 2.2, we were lack of computational power to do grid search for hyperparameter tuning and running each experiment for 10 different random seeds. Instead, we random seeded the model and did a logical grid search by hand to choose the best one that shows a similar trend in paper and run that setup one time. Also, to provide some reproducibility of our result, we changed some torch.backend settings not to choose best algorithm for currently used hardware.

We conducted 2 experiments on each dataset:
- Experiment 1: Run SMG with different learning rate schedulers and see how it performs
- Experiment 2: Run `ADAM, SGD, SGD-M, SSMG(Single Shuffling Gradient Momentum)` with constant learning rate scheduler and compare them with SMG

The hyperparameter $\beta$ was proposed as 0.5 in the paper, thus we used as it is. Besides, all learning schedulers were given as formulas in appendix. We directly implemented them without any interpretation.

For the experiment 1, we used the hyperparameters below:
| **Dataset**   | **Constant** | **Diminishing**        | **Exponential**               | **Cosine**          |
|---------------|--------------|------------------------|-------------------------------|---------------------|
| Fashion-MNIST | $LR=0.1$       | $LR=0.5 , \lambda=8$   | $LR=0.5 , Decay Rate = 0.99$    | $LR=0.2 , T = 200$    |
| CIFAR-10      | $LR=0.1$       | $LR=0.5 , \lambda=8$   | $LR=0.5 , Decay Rate = 0.99$    | $LR=0.1 , T = 200$    |
| ijcnn1        | $LR=0.001$     | $LR=0.001 , \lambda=8$ | $LR=0.005 , Decay Rate = 0.99$  | $LR=0.005 , T = 200$  |
| w8a           | $LR=0.001$     | $LR=0.01 , \lambda=8$  | $LR=0.005 , Decay Rate = 0.99$  | $LR=0.005 , T = 200$  |

For the experiment 2, we used the hyperparameters below:
| **Optimizer** | Fashion-MNIST                            | CIFAR-10                                 | ijcnn1                                   | w8a                                       |
|---------------|------------------------------------------|------------------------------------------|------------------------------------------|-------------------------------------------|
| ADAM          | $LR=0.001 , \beta_{1}=0.9 , \beta_{2}=0.999$ | $LR=0.001 , \beta_{1}=0.9 , \beta_{2}=0.999$ | $LR=0.001 , \beta_1=0.9 , \beta_2=0.999$ | $LR=0.0001 , \beta_1=0.9 , \beta_2=0.999$ |
| SGD           | $LR=0.1$                                   | $LR=0.04$                                  | $LR=0.001$                                 | $LR=0.001$                                  |
| SGD-M         | $LR=0.1 , \beta=0.5$                     | $LR=0.02 , \beta=0.5$                    | $LR=0.001 , \beta=0.5$                   | $LR=0.001 , \beta=0.5$                    |
| SSMG          | $LR=0.2, \beta=0.5$                      | $LR=0.1 , \beta=0.5$                     | $LR=0.001 , \beta=0.5$                   | $LR=0.001 , \beta=0.5$                    |

## 3.2. Running the code

Our directory tree of Github can be seen as:
```bash
├── README.md
├── experiments
│   ├── cifar10_experiments.ipynb
│   ├── fashionmnist_experiments.ipynb
│   ├── ijcnn1_experiments.ipynb
│   └── w8a_experiments.ipynb
└── results
    └── overall_results.ipynb
```
We conducted our work in different ipynb files for each dataset due to lack of computational power. 
Explanations of directories can be found below:

- `Experiments` directory includes seperate iypnb files for the experiments that we conducted for each dataset. 
- `Results` directory includes a single ipynb file that plots overall results for the conducted experiments.

Each of the files in experiments includes our implementation of SMG (Shuffling Momentum Gradient) and SSMG(Single Shuffling Momentum Gradient) algorithms for training. We supported our implementation with comments to ease understandibility of the codes. 

It can be directly accesible through `Open in Colab` button that can be found top of each ipynb files.

<img width="206" alt="image" src="https://user-images.githubusercontent.com/44034966/177547754-3ef783af-c0f1-4777-8eab-7682707f2f82.png">

If you want to reproduce our results, you need to run each experiment file first and then run `overall_results.ipynb` to see the results.
Since we used Colab, we saved our results to Google Drive. If you want to save it to Google Drive, please create the following directory structure to your Google Drive. If you do not want, please change the corresponding paths.
```bash
└── SMGExperiments
    ├── Cifar10-Other-Optimizers
    ├── Fashion-Other-Optimizers
    ├── ijcnn1-Other-Optimizers
    ├── w8a-Other-Optimizers
    ├── SMG-Cifar10-History
    ├── SMG-Fashion-History
    ├── SMG-ijcnn1-History
    └── SMG-w8a-History
```

You can also use, Jupyter Notebook to use your own hardware instead of the assigned CPU/GPU in Colab.

## 3.3. Results

In the paper, there exists no numerical results. Instead, they present their results using graphs obtained from the average of 10 runs. Thus, we used the same method to compare our results. We used both train loss and the norm of grad squares obtained from the history of training, same as in the paper.

**Experiment 1:**

The first experiment compares SMG itself by using different learning rate schedulers. They use 4 distinct learning schedulers namely, `constant`, `exponential`, `diminishing` and `cosine`. We used the same formula for schedulers they used in the paper. You can see the plotting of the acquired `train loss` results presented in the paper and our results below:

![imgonline-com-ua-twotoone-PRA65LYd0cnGOWKr](https://user-images.githubusercontent.com/44034966/177620586-e5ca1da2-8d50-4f13-892c-e7f3e82576aa.jpg)
<p align="center"> Figure 1. The train loss produced by SMG using four different learning rate schedulers (Paper Results) </p>

<img width="1383" alt="image" src="https://user-images.githubusercontent.com/44034966/177621433-10d5683d-c95f-4ebb-ad22-aab60fa19569.png">
<p align="center"> Figure 2. The train loss produced by SMG using four different learning rate schedulers (Our Results) </p>


You can see the plotting of the acquired `the squared norm of gradient` results presented in the paper and our results below:

<img width="1386" alt="image" src="https://user-images.githubusercontent.com/44034966/177624213-c8ec116b-3179-4158-8728-07d7f413b77d.png">
<p align="center"> Figure 3. The squared norm of gradient produced by SMG under four different learning rate schedulers (Paper Results) </p>

<img width="1382" alt="image" src="https://user-images.githubusercontent.com/44034966/177624730-f602a006-8f36-4efb-9021-d610d5589d4d.png">
<p align="center"> Figure 4. The squared norm of gradient produced by SMG under four different learning rate schedulers (Our Results) </p>


**Experiment 2:**

The second experiment compares SMG with other optimizers namely, `ADAM`, `SGD`, `SGD-M`, `SSMG (Single Shuffling Gradient Momentum)`. This experiment is conducted with constant learning scheduler as stated in the paper. We used optimizers directly through PyTorch. You can see the plotting of the obtained `train loss` results presented in the paper and our results below:

<img width="1374" alt="image" src="https://user-images.githubusercontent.com/44034966/177627290-7a778d66-7fc4-4e9b-a529-1747f5203a61.png">
<p align="center"> Figure 5. The train loss produced by SMG, SSMG, SGD, SGD-M, and ADAM (Paper Results) </p>

<img width="1375" alt="image" src="https://user-images.githubusercontent.com/44034966/177627520-8e1919a2-909f-4ee6-a6b3-054228e09185.png">
<p align="center"> Figure 6. The train loss produced by SMG, SSMG, SGD, SGD-M, and ADAM (Our Results) </p>


You can see the plotting of the acquired `the squared norm of gradient` results presented in the paper and our results below:

<img width="1373" alt="image" src="https://user-images.githubusercontent.com/44034966/177627662-cb720147-8f66-4486-b04f-8ffd8ad66565.png">
<p align="center"> Figure 7. The squared norm of gradient produced by SMG, SSMG, SGD, SGD-M, and ADAM (Paper Results) </p>

<img width="1373" alt="image" src="https://user-images.githubusercontent.com/44034966/177627827-a7f9927f-4971-4ed7-91ea-196a995b7c2e.png">
<p align="center"> Figure 8. The squared norm of gradient produced by SMG, SSMG, SGD, SGD-M, and ADAM (Our Results) </p>

We have very similar results on the datasets CIFAR-10 and Fashion-MNIST compared to the paper. There are some small changes in graphs. On the contrary, we were not able to catch the numbers of w8a and ijcnn1 classification datasets like we did in CIFAR-10 and Fashion-MNIST. However, we were able to catch the trends.


# 4. Conclusion

In general, even though we could not get the same numbers, we were able to catch the same trend with the results that are obtained in the paper. This is due to following reasons:
- They reported the results as average of 10 runs with different random seeds.
- Each run gives them different starting point for the model. 
- They were able to do exhaustive grid search to find hyperparameters that performs best according to their definition.
- We used different hyperparameters.
- Normalization of datasets differs with respect to each other.
- There might be architectural differences in the networks.

We were successful on the side of CIFAR-10 and Fashion-MNIST where we trained them on widely used models LeNet5 and Lenet-300-100. Since the implementation details are explicitly stated for these, there are some small differences in the graphs. Main reasons are difference in normalization and choices of hyperparameters. We were probably able to choose the ones that performs similar to authors. 

Although the graphs of train loss are similar at first sight for w8a and ijcnn1, there are some big numerical differences. Since these classification datasets are scaled to ranges [-1,1] or [0,1], we do not think normalization makes the big difference here. Since the network for nonconvex logistic regression is not stated clearly, we had to decide the architecture ourselves. There might be distinct decisions on the activation functions and other architectural details on the side of authors. 

We think that our implementation of the proposed SMG (Shuffling Gradient Momentum) method is as successful as it is compared to the results shared in the paper.

# 5. References

- Nouman. (2022, January). Writing Lenet5 from scratch in Pytorch. Paperspace Blog. Retrieved June 26, 2022, from https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/ 
- Tran, T. H., Nguyen, L. M., & Tran-Dinh, Q. (2020). SMG: A Shuffling Gradient-Based Method with Momentum. arXiv preprint arXiv:2011.11884.
- Yu, J. (2018). LeNet-300-100 https://github.com/jiecaoyu/scalpel-1/

w8a and ijcnn1 classification datasets are downloaded from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html


# Contact

For any further questions and suggestions, you can contact us at omer.kose@metu.edu.tr and yigit.varli@metu.edu.tr
