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

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

- Nouman. (2022, January). Writing Lenet5 from scratch in Pytorch. Paperspace Blog. Retrieved June 26, 2022, from https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/ 
- Tran, T. H., Nguyen, L. M., & Tran-Dinh, Q. (2020). SMG: A Shuffling Gradient-Based Method with Momentum. arXiv preprint arXiv:2011.11884.
- Yu, J. (2018). LeNet-300-100 [https://github.com/jiecaoyu/scalpel-1/]

w8a and ijcnn1 classification datasets are downloaded from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html


# Contact

For any further questions and suggestions, you can contact us at omer.kose@metu.edu.tr / yigit.varli@metu.edu.tr
