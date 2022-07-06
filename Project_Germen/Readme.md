

<p align="center">
<mark> = = = THIS FILE MUST BE READ WITH LIGHT THEME/VIEW = = = </mark>
</p>

              Otherwise equations and some formulas are not clearly visible!

# Slot Machines: Discovering Winning Combinations of Random Weights in Neural Networks

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

Slot Machines written by [Maxwell Mbabilla Aladago](https://scholar.google.com/citations?user=ekf53bkAAAAJ&hl=en ) and [Lorenzo Torresani](https://scholar.google.com/citations?user=ss8KR5gAAAAJ&hl=en), and published in [ICLR2021](https://iclr.cc/Conferences/2021). It builds on the idea proposed by Ramanujan et al. (2020) and Zhou et al. (2019) that even random weighted networks pruned correctly can perform well without any training. They are showing a possible way to harness over expressiveness of the networks. In Slot Machines, instead of pruning the network, weights of the network are limited to a potential "k" amount of discrete options per weight drawn from a distribution. Training is done upon this "discrete weighted" network. They show that even with two (k=2) discrete options per weight can perform test set accuracy of 98.2% on CIFAR10. With the larger number of weight options (8 - 32), discrete weighted networks perform similarly or better than traditional continuously weighted networks.

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

The paper proposes to change the weight system of existing networks by giving a limited amount of discrete weight options per weight. Traditional networks can change their weight continuously according to chosen optimization methods. In slot machines, networks can only "choose" from predefined, limited, and initialized discrete values.

This change brings two questions:
- How to initialize the discrete weights?
- How to optimize, choose the best weights?

Weights are initialized from the Glorot Uniform distribution (Glorot & Bengio, 2010). Each discrete weight has appointed a score to be changed by the optimization method according to the gradient of the loss with respect to the weight's score. With these changes, slot machines act like traditional networks. Forward pass happens according to "best scored" weights. In backpropagation, gradients are calculated for weights' scores, and the optimization method updates the current weights' scores. Then "best scored" weights are chosen as new "current" weights to be used in the next forward pass. 

"Best scored" weights are chosen according to proposed two methods:

**Greedy Selection:** Weight with maximum score is selected.

**Probabilistic Sampling:** Possible weights' scores are sampled with multinomial distribution to determine the new "current" weight.

This new weight system allows it to be generically used, essentially every network using the traditional weight system.

Priorly experiment networks' weights and training methods were changed to slot machines' defined methods and investigated in the following topics:

- Performance compared to traditional networks
- Performance compared to pruning networks
- Performance with different amounts of optional discrete weights
- Fine tunning traditionally after slot machine training
- Affect of "best score" selection function
- Weight sharing

---
The paper brings a new way to harness over expressiveness of the traditional networks by limiting the options of weights, in essence, creating an implicit regularization on the network. It also reinforces the idea that networks are about "ideal" weight combinations rather than ideal weights. 

Even though it is not explicitly stated in the paper, the proposed method is a new way to push weight-wise optimization to network-wise optimization; The weights do not immediately respond to the error signal, needing to reach a threshold before responding. In this way, each backpropagation doesn't change all of the network but the ones near the threshold. This way network doesn't have to adapt to a "completely" new network each iteration.

# 2. The method and my interpretation

## 2.1. The original method

@TODO: Explain the original method.
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">  (Eq. 4)
</p>
In the Slot Machines (SM), the number of options per connection (k) amount of weights is initialized per traditional weight. Initialization is made by the uniform sampling of a Glorot Uniform distribution bound by the standard deviation of the Glorot Normal distribution. The standard deviation of the Glorot Normal distribution is calculated according to traditional weight size, ignoring the "k." This is because the SM network's capacity is identical to a traditional network for a forward pass.


Each weight has its own score. The scores are initialized independently from a uniform distribution, upper bound as constant lambda multiplied by the standard deviation of the Glorot Normal distribution. Lambda is chosen as "1" for convolution layers and "0.1" for fully connected layers.

Active, current weights are found by processing corresponding potential weights' scores. Two methods are proposed for processing. Greedy Selection(GS), using the maximum scored weight and Probabilistic Sampling (PS), the scores are sampled as a multinomial distribution, resulting score's weight is chosen.

Finally, the gradient of the score is calculated by the multiplication of its corresponding weight's gradient and the weight itself. The found gradient is decreased from the corresponding score after multiplication with the learning rate. Completely ignoring any memory, momentum if there is any.

## 2.2. My interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

The description of the method leaves little to no place for interpretation. Most of the interpretation, guesswork is done on the experiment settings, which can be called as vague.

The only place open to interpretation was the initialization of weights. It is stated as:

>Sampled uniformly at random from a Glorot Uniform distribution where bounds are the standard deviation of the Glorot Normal distribution. 

In classic, already implemented Glorot Uniform Distribution bounds are the standard deviation of the Glorot Normal distribution multiplied with the square root of three. 

It isn't clear which one the authors refer to as the bound:
- The standard deviation of the Glorot Normal distribution
- The standard deviation of the Glorot Normal distribution multiplied by the square root of three

In the implementation, the former was used.

This difference may not be seen as a "part" of the model/method due to being written in the experiment section. However, due to the nature of the SMs, the wrong initialization can decrease converging speed and even make it stagnant, and vice versa. 


# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

### 3.1.1 Paper experimentel setup

| Network                   | Lenet        | CONV-2       | CONV-4         | CONV-6         | VGG-19          |
|---------------------------|--------------|--------------|----------------|----------------|-----------------|
| Convolutional Layers      |              |              |                |                | 2x64, pool      |
|                           |              |              |                |                | 2x128, pool     |
|                           |              |              |                |  64, 64, pool  | 2x256, pool     |
|                           |              |              | 64, 64, pool   | 128, 128, pool | 4x512, pool     |
|                           |              | 64, 64, pool | 128, 128, pool | 256, 256, pool | 4x512, avg-pool |
| Fully-connected Layers    | 300, 100, 10 | 256, 256, 10 | 256, 256, 10   | 256, 256,10    |       10        |
| Epochs: Slot Machines     |     200      |     200      |     200        |      200       |     220         |
| Epochs: Learned Weights   |     200      |     200      |     330        |      330       |     320         |
| Dataset                   |    MNIST     |   CIFAR-10   |   CIFAR-10     |   CIFAR-10     |   CIFAR-10      |
|  Validation % of training |      15%     |     10%      |      10%       |      10%       |      10%        |

<p align="center">
<b>Table 1: Architecture specifications of the networks in experiments conducted in the paper</b>
</p>

The SMs are tested on 5 different networks in the paper that can be seen in Table 1. Lenet, CONV-2, CONV-4, CONV-6 were tested with all combinations of K (2, 4, 8, 16, 32, 64) and scoring functions (GS, PS). Batch size of 128 and stochastic gradient descent with warm restarts (at epoch 25 and 75), a momentum of 0.9 and a l2 penalty of 0.0001. When training GS SMs, learning rate was set to 0.2 for K =< 8 and 0.1 otherwise. When training PS SMs, learning rate was increased to 25. Data augmentation was applied on CIFAR-10 and dropout (with a rate of p = 0.5). Early stopping according to validation test accuracy was used. All convulution layers have 3 x 3 filters.

### 3.1.2 Implementation experimentel setup


| Network                   | CONV-2       | CONV-4         | CONV-6         |
|---------------------------|--------------|----------------|----------------|
| Convolutional Layers      |              |                |                |
|                           |              |                |                |
|                           |              |                |  64, 64, pool  |
|                           |              | 64, 64, pool   | 128, 128, pool |
|                           | 64, 64, pool | 128, 128, pool | 256, 256, pool |
| Fully-connected Layers    | 256, 256, 10 | 256, 256, 10   | 256, 256,10    |
| Epochs: Slot Machines     |     200      |     200        |      200       |
| Epochs: Learned Weights   |     200      |     330        |      330       |
| Dataset                   |   CIFAR-10   |   CIFAR-10     |   CIFAR-10     |
|  Validation % of training |     10%      |      10%       |      10%       |

<p align="center">
<b>Table 2: Architecture specifications of the networks in experiments conducted in this implementation/repository</b>
</p>

The SMs are tested on 3 different networks in the paper that can be seen in Table 2.  Combinations of the conducted experiments can be seen in Table 3. Batch size of 128 and stochastic gradient descent, a momentum of 0.9 and a l2 penalty of 0.0001. When training GS SMs, learning rate was set to 0.2 for K =< 8 and 0.1 otherwise. When training PS SMs, learning rate was increased to 25. CIFAR-10 and dropout (with a rate of p = 0.5). Early stopping according to validation test accuracy was used. All convulution layers have 3 x 3 filters.

| K\Net. | Conv 2 SM             | Conv 4 SM           | Conv 6 SM           |
|--------|-----------------------|---------------------|---------------------|
| 2      |    :heavy_check_mark: | :heavy_check_mark:  | <ul><li>- [ ] </li> |
| 4      | :heavy_check_mark:    | <ul><li>- [ ] </li> | <ul><li>- [ ] </li> |
| 8      | :heavy_check_mark:    | <ul><li>- [ ] </li> | <ul><li>- [ ] </li> |
| 16     | :heavy_check_mark:    | <ul><li>- [ ] </li> | <ul><li>- [ ] </li> |
| 32     | :heavy_check_mark:    | <ul><li>- [ ] </li> | <ul><li>- [ ] </li> |
| 64     | :heavy_check_mark:    | <ul><li>- [ ] </li> | <ul><li>- [ ] </li> |

<p align="center">
<b>Table 3: Experiment combinations of networks and weight options for PS SMs. Check mark indicating experiment is conducted.</b>
</p>

**Explicit differences with paper**

Parts that are not implemented here eventhough it is explicitly said to be implemented in the paper.
- Data augmentation was ommmited to decrease training time and due to lack of spesification on applied augmentations.
- Warm restarts were ommitted. (Ready implementation doesn't work with this implementation. Due to changed stepping method.) 

**Implicit implementations**

Parts that are not explicitly stated, however stated in 1 - 2 level deep references.

- RELU activation was used after every hidden layer.
- Dropout applied to only first two fully connected layer.
- Size is invariant to convolutions. (Padding is one.)
- Pool max spesifications: 2 x 2 kernel, 2 stride.

**Interpreted implementations**

- Current weights are set just before forwarding. This could have been after updating scores too. This doesn't affect the results, however can affect computation time. 
- Weights are initialized inside the modul. If wanted, makes implementing weight sharing  harder.
- Number of optional weights are passed as initilization variable. Easier to implement varying number of optional weights.
- Score selection function type is passed as initilization variable. Easier to implement different functions for different layers/modules.
- :warning: "cuda" is hardcoded inside the module. :warning: This means that code will :warning: ONLY :warning: run on "cuda" avaliable and activated environment. This includes validation and testing.


## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

Directory consists of 3 folders data, images, models_layers_trainers.

**data:** Holds data, CIFAR-10 dataset


## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.

Deniz Germen 
