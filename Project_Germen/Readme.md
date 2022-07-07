

<p align="center">
<mark> = = = THIS FILE MUST BE READ WITH LIGHT THEME/VIEW = = = </mark>
</p>

              Otherwise equations and some formulas are not clearly visible!

# Slot Machines: Discovering Winning Combinations of Random Weights in Neural Networks

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

[Slot Machines](https://arxiv.org/abs/2101.06475) written by [Maxwell Mbabilla Aladago](https://scholar.google.com/citations?user=ekf53bkAAAAJ&hl=en ) and [Lorenzo Torresani](https://scholar.google.com/citations?user=ss8KR5gAAAAJ&hl=en), and published in [ICLR2021](https://iclr.cc/Conferences/2021). It builds on the idea proposed by Ramanujan et al.[^raman] (2020) and Zhou et al. [^zhou] (2019) that even random weighted networks pruned correctly can perform well without any training. They are showing a possible way to harness over expressiveness of the networks. In Slot Machines, instead of pruning the network, weights of the network are limited to a potential "k" amount of discrete options per weight drawn from a distribution. Training is done upon this "discrete weighted" network. They show that even with two (k=2) discrete options per weight can perform test set accuracy of 98.2% on CIFAR10[^kriz09]. With the larger number of weight options (8 - 32), discrete weighted networks perform similarly or better than traditional continuously weighted networks.

## 1.1. Paper summary

The paper proposes to change the weight system of existing networks by giving a limited amount of discrete weight options per weight. Traditional networks can change their weight continuously according to chosen optimization methods. In slot machines, networks can only "choose" from predefined, limited, and initialized discrete values.

This change brings two questions:
- How to initialize the discrete weights?
- How to optimize, choose the best weights?

Weights are initialized from the Glorot Uniform distribution[^glo]. Each discrete weight has appointed a score to be changed by the optimization method according to the gradient of the loss with respect to the weight's score. With these changes, slot machines act like traditional networks. Forward pass happens according to "best scored" weights. In backpropagation, gradients are calculated for weights' scores, and the optimization method updates the current weights' scores. Then "best scored" weights are chosen as new "current" weights to be used in the next forward pass. 

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
The paper brings a new way to harness over expressiveness of the traditional networks by limiting the options of weights, in essence, creating an implicit regularization on the network. It also reinforces the idea that networks are about "ideal" weight combinations rather than ideal weights, builds on top of similar methods[^raman][^zhou][^lee][^tanaka].

Even though it is not explicitly stated in the paper, the proposed method is a new way to push weight-wise optimization to network-wise optimization; The weights do not immediately respond to the error signal, needing to reach a threshold before responding. In this way, each backpropagation doesn't change all of the network but the ones near the threshold. This way network doesn't have to adapt to a "completely" new network each iteration.

# 2. The method and my interpretation

## 2.1. The original method

In the Slot Machines (SM), the number of options per connection (k) amount of weights is initialized per traditional weight (Eq. 1). Initialization is made by the uniform sampling of a Glorot Uniform distribution[^glo] bound by the standard deviation of the Glorot Normal distribution(Eq. 2-3). The standard deviation of the Glorot Normal distribution is calculated according to traditional weight size, ignoring the "k." This is because the SM network's capacity is identical to a traditional network for a forward pass.

<p align="center">
<img src="https://bit.ly/3arEO46" align="center" border="0" alt="W_{ij}  \Longrightarrow  ( W_{ij1},...,W_{ijK} )  " width="179" height="21" /> (Eq. 1)
</p>

<p align="center">
<img src="https://bit.ly/3yKy2zL" align="center" border="0" alt="std =  \sqrt{ \frac{2}{ fan_{in} +  fan_{out}  } } " width="187" height="57" /> (Eq. 2)
</p>

<p align="center">
<img src="https://bit.ly/3nGKVF1" align="center" border="0" alt="\pazocal{U}( - \sigma _{x} ,  \sigma _{x} )" width="90" height="18" /> (Eq. 3)
</p>

Each weight has its own score(Eq. 4). The scores are initialized independently from a uniform distribution, upper bound as constant lambda multiplied by the standard deviation of the Glorot Normal distribution(Eq. 2, 5). Lambda is chosen as "1" for convolution layers and "0.1" for fully connected layers.

<p align="center">
<img src="https://bit.ly/3yjNQYV" align="center" border="0" alt="( W_{ij1},...,W_{ijK} ) \Longrightarrow ( s_{ij1},...,s_{ijK} )" width="254" height="21" /> (Eq. 4)
</p>

<p align="center">
<img src="https://bit.ly/3o2dQ6F" align="center" border="0" alt="\pazocal{U}( 0 ,   \lambda \sigma _{x} )" width="79" height="18" /> (Eq. 5)
</p>

Active, current weights are found by processing corresponding potential weights' scores(Eq. 6). Two methods are proposed for processing. Greedy Selection(GS), using the maximum scored weight (Eq. 7) and Probabilistic Sampling (PS), the scores are sampled as a multinomial distribution, resulting score's weight is chosen (Eq. 8).

<p align="center">
<img src="https://bit.ly/3uwOjpM" align="center" border="0" alt=" k^{*}  =  \rho (s_{ij1},...,s_{ijK})" width="157" height="22" /> (Eq. 6)
</p>

<p align="center">
<img src="https://bit.ly/3bQT72G" align="center" border="0" alt=" \rho   =  argmax(s_{ij1},...,s_{ijK})" width="204" height="21" />(Eq. 7)
</p>

<p align="center">
<img src="https://bit.ly/3RfHB1a" align="center" border="0" alt=" \rho   =  Mult(SoftMax(s_{ij1},...,s_{ijK}))" width="261" height="21" />(Eq. 8)
</p>

Finally, the gradient of the score is calculated by the multiplication of its corresponding weight's gradient and the weight itself(Eq. 9). The found gradient is decreased from the corresponding score after multiplication with the learning rate. Completely ignoring any memory, momentum if there is any(Eq. 10).

<p align="center">
<img src="https://bit.ly/3R8Jj4a" align="center" border="0" alt="  \nabla _{ s_{ijk} }   \leftarrow  \frac{ \partial \pazocal{L}}{ \partial  a(x)_{i}^{(l)} } h(x)_{j}^{(l-1)} W_{ijk} ^{l} " width="226" height="53" /> (Eq. 9)
</p>

<p align="center">
<img src="https://bit.ly/3RfuaOE" align="center" border="0" alt="    \widetilde{s_{ijk}}  = s_{ijk} - \alpha \nabla _{ s_{ijk} }   " width="139" height="24" /> (Eq. 10)
</p>

## 2.2. My interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

The description of the method leaves little to no place for interpretation. Most of the interpretation, guesswork is done on the experiment settings, which can be called as vague.

The only place open to interpretation was the initialization of weights. It is stated as:

>Sampled uniformly at random from a Glorot Uniform distribution where bounds are the standard deviation of the Glorot Normal distribution. 

In classic, already [implemented](https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_uniform_) Glorot Uniform Distribution bounds are the standard deviation of the Glorot Normal distribution multiplied with the square root of three. 

It isn't clear which one the authors refer to as the bound:
- The standard deviation of the Glorot Normal distribution(Eq. 3)
- The standard deviation of the Glorot Normal distribution multiplied by the square root of three(Eq. 11 - 12)

In the implementation, the former was used.

<p align="center">
<img src="https://bit.ly/3utM6eD" align="center" border="0" alt="a =  std  \times \sqrt{3} " width="110" height="22" /> (Eq. 11)
</p>

<p align="center">
<img src="https://bit.ly/3yprxRC" align="center" border="0" alt="\pazocal{U}( - a ,  a)" width="74" height="18" /> (Eq. 12)
</p>

This difference may not be seen as a "part" of the model/method due to being written in the experiment section. However, due to the nature of the SMs, the wrong initialization can decrease converging speed and even make it stagnant, and vice versa. 


# 3. Experiments and results

## 3.1. Experimental setup

### 3.1.1 Paper experimentel setup

|          Network          |     Lenet    |    CONV-2    |     CONV-4     |     CONV-6     |      VGG-19     |
|:-------------------------:|:------------:|:------------:|:--------------:|:--------------:|:---------------:|
|    Convolutional Layers   |              |              |                |                |    2x64, pool   |
|                           |              |              |                |                |   2x128, pool   |
|                           |              |              |                |   64, 64, pool |   2x256, pool   |
|                           |              |              |  64, 64, pool  | 128, 128, pool |   4x512, pool   |
|                           |              | 64, 64, pool | 128, 128, pool | 256, 256, pool | 4x512, avg-pool |
|   Fully-connected Layers  | 300, 100, 10 | 256, 256, 10 |  256, 256, 10  |   256, 256,10  |        10       |
|   Epochs: Slot Machines   |      200     |      200     |       200      |       200      |       220       |
|  Epochs: Learned Weights  |      200     |      200     |       330      |       330      |       320       |
|          Dataset          |     MNIST    |   CIFAR-10   |    CIFAR-10    |    CIFAR-10    |     CIFAR-10    |
|  Validation % of training |      15%     |      10%     |       10%      |       10%      |       10%       |    10%       |      10%        |

<p align="center">
<b>Table 1: Architecture specifications of the networks in experiments conducted in the paper</b>
</p>

The SMs are tested on 5 different networks in the paper that can be seen in Table 1. Lenet, CONV-2, CONV-4, CONV-6 were tested with all combinations of K (2, 4, 8, 16, 32, 64) and scoring functions (GS, PS). Batch size of 128 and stochastic gradient descent with warm restarts[^hutter] (at epoch 25 and 75), a momentum of 0.9 and a l2 penalty of 0.0001. When training GS SMs, learning rate was set to 0.2 for K =< 8 and 0.1 otherwise. When training PS SMs, learning rate was increased to 25. Data augmentation was applied on CIFAR-10[^kriz09] and dropout (with a rate of p = 0.5). Early stopping according to validation test accuracy was used. All convulution layers have 3 x 3 filters.

### 3.1.2 Implementation experimentel setup


|          Network          |    CONV-2    |     CONV-4     |     CONV-6     |
|:-------------------------:|:------------:|:--------------:|:--------------:|
|    Convolutional Layers   |              |                |                |
|                           |              |                |   64, 64, pool |
|                           |              |  64, 64, pool  | 128, 128, pool |
|                           | 64, 64, pool | 128, 128, pool | 256, 256, pool |
|   Fully-connected Layers  | 256, 256, 10 |  256, 256, 10  |   256, 256,10  |
|   Epochs: Slot Machines   |      200     |       200      |       200      |
|  Epochs: Learned Weights  |      200     |       330      |       330      |
|          Dataset          |   CIFAR-10   |    CIFAR-10    |    CIFAR-10    |
|  Validation % of training |      10%     |       10%      |       10%      |

<p align="center">
<b>Table 2: Architecture specifications of the networks in experiments conducted in this implementation/repository</b>
</p>

The SMs are tested on 3 different networks in the paper that can be seen in Table 2.  Combinations of the conducted experiments can be seen in Table 3. Batch size of 128 and stochastic gradient descent, a momentum of 0.9 and a l2 penalty of 0.0001. When training GS SMs, learning rate was set to 0.2 for K =< 8 and 0.1 otherwise. When training PS SMs, learning rate was increased to 25. CIFAR-10[^kriz09] and dropout (with a rate of p = 0.5). Early stopping according to validation test accuracy was used. All convulution layers have 3 x 3 filters. In total 20 models were trained.

|      K\Net.     |       Conv 2 SM       |      Conv 2 SM      |      Conv 4 SM      |      Conv 6 SM      |
|:---------------:|:---------------------:|:-------------------:|:-------------------:|:-------------------:|
|        2        |    :heavy_check_mark: |  :heavy_check_mark: |  :heavy_check_mark: |  :heavy_check_mark: |
|        4        |   :heavy_check_mark:  |  :heavy_check_mark: |  :heavy_check_mark: |  :heavy_check_mark: |
|        8        |   :heavy_check_mark:  |  :heavy_check_mark: |  :heavy_check_mark: |  :heavy_check_mark: |
|        16       |   :heavy_check_mark:  | <ul><li>- [ ] </li> |  :heavy_check_mark: |  :heavy_check_mark: |
|        32       |   :heavy_check_mark:  | <ul><li>- [ ] </li> | <ul><li>- [ ] </li> | <ul><li>- [ ] </li> |
|        64       |   :heavy_check_mark:  | <ul><li>- [ ] </li> | <ul><li>- [ ] </li> | <ul><li>- [ ] </li> |
| Score Selection |           GS          |          PS         |          GS         |          GS         |

<p align="center">
<b>Table 3: Experiment combinations of networks and weight options for PS SMs. Check mark indicating experiment is conducted.</b>
</p>

**Explicit differences with paper**

Parts that are not implemented here eventhough it is explicitly said to be implemented in the paper.
- Data augmentation was ommmited to decrease training time and due to lack of spesification on applied augmentations.
- Warm restarts were ommitted. (Ready implementation doesn't work with this implementation. Due to changed stepping method.) 

**Implicit implementations**

Parts that are not explicitly stated, however stated in 1 - 2 level deep references.

- RELU activation was used after every hidden layer. [^karen]
- Dropout applied to only first two fully connected layer. [^karen]
- Size is invariant to convolutions. (Padding is one.) [^karen]
- Pool max spesifications: 2 x 2 kernel, 2 stride. [^karen]

**Interpreted implementations**

- Current weights are set just before forwarding. This could have been after updating scores too. This doesn't affect the results, however can affect computation time. 
- Number of optional weights are passed as initilization variable. Easier to implement varying number of optional weights.
- Score selection function type is passed as initilization variable. Easier to implement different functions for different layers/modules.
- :warning: "cuda" is hardcoded inside the module. :warning: This means that code will :warning: ONLY :warning: run on "cuda" avaliable and activated environment. This includes validation and testing.


## 3.2. Running the code

### 3.2.1 Directory

Directory consists of 3 folders `data`, `images`, `models_layers_trainers` and `custom_experiments.py `file in main directory.

`data:` Holds data, CIFAR-10[^kriz09] dataset. Data is empty in Github, however with the first run of the experiment, it iwll be automatically filled.

`images:` Holds images used in `readme.md` and `slotmachines.ipynb`.

`models_layers_trainers:` Holds `SM_Layers.py`and `SM_Models.py`.

- `SM_Layers.py:`  Holds the core implementation, derived versions of 2d convolution layer and fully connected layer
- `SM_Models.py:` Holds the models created with custom SM layers.

`custom_experiments.py:` Parametricly runnable experiment python code file

### 3.2.1 Running the code

- :warning: Since "cuda" is hardcoded in the `SM_Layers.py` on modules itself. It can only run on a "cuda capable" environment.

**Environment Spesifications**

**Windows 10**

- [x] Python 3.8.13
- [x] Cuda 11.3
- [x] Nvidia GeForce Driver 512.15

**macOS Monterey 12.3.1** 

- Hardcoded "cuda"s were removed, dataset and model were on CPU.
- [x] Python 3.8.13 (arch-arm)


**Parametric run command**

Run the following command in directory of the repository. This will run the custom experiment with its defaults parameters. 

```
python custom_experiments.py 
```

Following parameters can be added after the command to customize the experiment:

`--model {X}`
- Name of the wanted model.
- Currently, CONV_2, CONV_4, CONV_6, CONV_2_SM, CONV_4_SM, CONV_6_SM are avaliable.

`--K {X}`
- Number of optional weights per weight
- Any integer above 0 is accepted.

`--selector {X}`
- Selector function
- "GS" and "PS". (GS: Greedy Selector, PS: Probability Sampling)

`--lr {X}`
- Learning rate
- Any float above 0

`--epochs {X}`
- Number of epochs
- Any integer above -1

Example querries:

```
python custom_experiments.py --model CONV_4_SM --K 64 --selector "GS" --lr 0.1 --epochs 200

python custom_experiments.py --model CONV_2_SM --K 8 --selector "PS" --lr 0.2 --epochs 200

python custom_experiments.py --model CONV_6 --lr 0.001 --epochs 330

python custom_experiments.py --model CONV_2 --epoch 200
```

**Expected result:**

```
python custom_experiments.py --model CONV_4_SM --K 64 --selector "GS" --lr 0.1 --epochs 200
Cuda (GPU support) is available and enabled!
Files already downloaded and verified
Files already downloaded and verified
Epoch 0 / XXX: avg. loss of last 5 iterations x.xxxxxxxxxxxx
...
Epoch XXX / XXX: avg. loss of last 5 iterations x.xxxxxxxxxxxx
Accuracy out of 5000 images is 0.xxxx
Best epoch for validation and its test accuracy: XXX - 0.xxxx
```

After its done it will save the fully trained model and best model inside the fully trained model according to validation. For high "k"s, model size can exceed 5 GB.

**Run Times:**

- 2080 Super, GPU - ~2 hours, increasing with size of the model and k's value increases run time
- M1 Pro, CPU: 3 - 4 hours, 5+ hours for high value ks (k>8). Probably due to lacking memory starting to swap.


## 3.3. Results


In the following figures results from the paper and the implemntations can be seen. Odd numbered figures represent the results from the paper. Even numbered figures represent the results from the implementation. 

![alt text](https://github.com/CENG501-Projects/CENG501-Spring2022/blob/main/Project_Germen/images/fig1.png?raw=true)

<p align="center">
<b>Figure 1: K = 2 weight options per connection vs. randomly initialized network for CONV-2, CONV-4, CONV-6; paper</b>
</p>

|                                                                                                                         |                                                                                                                         |                                                                                                                         |
|-------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| ![alt text](https://github.com/CENG501-Projects/CENG501-Spring2022/blob/main/Project_Germen/images/fig2-1.png?raw=true) | ![alt text](https://github.com/CENG501-Projects/CENG501-Spring2022/blob/main/Project_Germen/images/fig2-1.png?raw=true) | ![alt text](https://github.com/CENG501-Projects/CENG501-Spring2022/blob/main/Project_Germen/images/fig2-1.png?raw=true) |

Comparing the Figure 1 and Figure 2, it can be seen that eventhough trained networks being significantly performing better than random network, implementation compares worse to the paper. 

In paper, 2 SMs surpass learned weighted networks (Figure 3; Plots 1 and 3). However, none of the SMs in implementation surpasses learned weight networks.

<p align="center">
<b>Figure 2: K = 2 weight options per connection vs. randomly initialized network for CONV-2, CONV-4, CONV-6; implementation</b>
</p>

![alt text](https://github.com/CENG501-Projects/CENG501-Spring2022/blob/main/Project_Germen/images/fig3.png?raw=true)

<p align="center">
<b>Figure 3: Performance of slotmachines with increasing K and traditional networks; paper</b>
</p>

| ![alt text](https://github.com/CENG501-Projects/CENG501-Spring2022/blob/main/Project_Germen/images/fig4-1.png?raw=true) | ![alt text](https://github.com/CENG501-Projects/CENG501-Spring2022/blob/main/Project_Germen/images/fig4-2.png?raw=true) | ![alt text](https://github.com/CENG501-Projects/CENG501-Spring2022/blob/main/Project_Germen/images/fig4-3.png?raw=true) |
|-------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|

Comparing the Figure 3 and Figure 4, for all models, performance is reported better in the paper. Looking at the trends of change, implementation and paper are similiar. Both, implementation and paper's performance first increases with increasing K and reaches a stagnation. 

<p align="center">
<b>Figure 4: Performance of slotmachines with increasing K and traditional networks; implementation</b>
</p>

![alt text](https://github.com/CENG501-Projects/CENG501-Spring2022/blob/main/Project_Germen/images/fig5.png?raw=true)

<p align="center">
<b>Figure 5: Performance of slotmachines with different score selection methods. Greedy Selection vs. Probabilistic sampling; paper</b>
</p>

![alt text](https://github.com/CENG501-Projects/CENG501-Spring2022/blob/main/Project_Germen/images/fig6.png?raw=true)

<p align="center">
<b>Figure 6: Performance of slotmachines with different score selection methods. Greedy Selection vs. Probabilistic sampling for CONV-2; implementation</b>
</p>

Comparing the second plot of the Figure 5 to Figure 6. Trend of SMs with GS are similar, with implementation being lower. In Figure 6, an unexpected behaviour is observed, SMs with PS's performance drops as K is increased.
# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

Overall, all models of the implementation perform worse than their paper pairs. However, trends are similar, except for Figure 6, SMs with PS's performance.

The immediate difference between traditional, canon learned weight models indicate a difference in training method, which was expected with the omission of data augmentation. Also, this could be due to differences in the unspecified part of the models. However, having such a significant difference was unexpected (Figure 1, 2) for SM models. This could show that SMs benefit more from augmentation compared to the traditional method.

The similarity of the trends can indicate a matching underlying mechanism, thus good reproduction in layers.

Figure 6's unexpected trend can be due to the nature of probabilistic sampling. In PS, selection for weight does not necessarily choose the best-scored weight; higher scored weights are just more "likely" to be chosen. As a result, the network can choose a bad-scored weight, leading to divergence during training. Similar behavior can be observed in Figure 5, Plot 2, SMs with PS; As can be seen, even for a small number of trials (4) model shows high variance (up to 5%) for the same hyperparameters. Figure 6's case can be a more pronounced version of this.

The SM models are too delicate since they can only use the initialized discrete weights; wrong initialization can mean an untrainable or limited model. In the implementation, slow adaptability was observed compared to a traditional network due to significant score thresholds compared to score gradients.

Its unique property of implicit regularization with weight option limitation can regularize it to not converge to minimums. 



# Contact

Deniz Germen (germen.deniz@ceng.metu.edu.tr , deniz.germen@icloud.com)

# 5. References

[^raman]: Ramanujan, V., Wortsman, M., Kembhavi, A., Farhadi,
A., and Rastegari, M. What’s hidden in a randomly weighted neural network? In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.
URL https://openaccess.thecvf.com/content_CVPR_2020/papers/Ramanujan_Whats_Hidden_in_a_Randomly_Weighted_Neural_Network_CVPR_2020_paper.pdf.

[^zhou]: Zhou, H., Lan, J., Liu, R., and Yosinski, J. Deconstructing lottery tickets: Zeros, signs, and thesupermask. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alche-Buc, F., Fox, E., and Garnett, R. ´(eds.), Advances in Neural Information ProcessingSystems, volume 32, pp. 3597–3607. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper/2019/file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf.

[^kriz09]: Krizhevsky, A. Learning multiple layers of features fromtiny images. Technical report, University of Toronto, 2009.

[^glo]:Glorot, X. and Bengio, Y. Understanding the difficulty of training deep feedforward neural networks. In Teh, Y. W.and Titterington, M. (eds.), Proceedings of the Thirteenth International Conference on Artificial Intelligence andStatistics, volume 9 of Proceedings of Machine LearningResearch, pp. 249–256, Chia Laguna Resort, Sardinia,Italy, 13–15 May 2010. PMLR.

[^lee]: Lee, N., Ajanthan, T., and Torr, P. SNIP: Single-shot
Network Pruning based on connection sensitivity. In
International Conference on Learning Representations,2019. URL https://openreview.net/forum?id=B1VZqjAcYX.

[^tanaka]: Tanaka, H., Kunin, D., Yamins, D. L. K., and Ganguli,
S. Pruning neural networks without any data by
iteratively conserving synaptic flow. In Advances
in Neural Information Processing Systems, volume 33, 2020. URL https://proceedings.neurips.cc/paper/2020/file/46a4378f835dc8040c8057beb6a2da52-Paper.pdf.

[^karen]: Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image
recognition. arXiv preprint arXiv:1409.1556, 2014.
[^hutter]: Loshchilov, I. and Hutter, F. Sgdr: Stochastic gradient descent with warm restarts. In International Conference on
Learning Representations (ICLR), 2017. URL https://openreview.net/pdf?id=Skq89Scxx.
