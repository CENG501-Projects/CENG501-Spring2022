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

@TODO: Describe the setup of the original paper and whether you changed any settings.

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
