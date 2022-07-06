# @TODO: Paper title

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

Spiking neural networks (SNNs) are artificial neural networks that lead to promising solutions for low-power computing. They include the idea of time in their working model in addition to neuronal and synaptic states. Through binary values, neurons in SNNs asynchronously communicate with one another. The main idea behind their working principle is that neurons in the SNN only communicate information when a membrane potential reaches a particular value rather than at the beginning of each propagation cycle. Due to the event-driven manner they have, SNNs create a possibility to process inputs sequentially at low-power consumption so, SNNs with leaky integrate and (LIF) neurons provide opportunities for energy-efficient computing and they are convenient for sequential learning. 

In their paper which was published in AAAI22, Ponghiran and Roy [1] show that SNNs can be trained for sequential tasks such that the internal states of a network of LIF neurons can be modified to learn extended sequences and are resilient to the vanishing gradient problem. In the experiment, TIMIT (TIMIT Acoustic-Phonetic Continuous Speech Corpus) and LibriSpeech 100h speech recognition datasets were used. TIMIT is a typical dataset that is employed in the testing of automatic speech recognition models. It consists of recordings of 630 speakers reading 10 phonetically dense lines in each of eight American English dialects [2]. And the LibriSpeech corpus is a group of approximately 1,000 hours of audiobooks. The training data on this dataset is split into 3 divisions such as 100hr, 360hr, and 500hr sets [3].

## 1.1. Paper summary

To demonstrate the vanishing gradient problem of a layer in SNN, the authors derived the gradient for each update, then they show that a term in their derivation converges to zero in some conditions. This situation in the spiking neuron layer is the result of the vanishing gradient problem which prevents the SNN from learning a long sequence. So, a need for improving inherent recurrence dynamics has arisen. The proposed method for internal states of the spiking neuron layer is that while the synaptic current equation which carries information over time since they are the main path for gradients to flow backward is modified, the membrane potential equation which tracks errors when the spiking neurons produce outputs is kept the same.

# 2. The method and my interpretation

## 2.1. The original method

To overcome the vanishing gradient problem two modifications are proposed in the paper. As in GRUs and LSTMs, one modification called “the SNNs with improved dynamics v1” is to selectively update the state of the neurons based on the equations below:

$$F[n]=σ(X[n] W_{fi} )$$

$$C[n]=ReLU(X[n] W_{ci} ) $$

$$I[n]=F[n]⊙I[n-1]+(1-F[n])⊙C[n]$$

where ⊙ operator corresponds to element-wise multiplication, $W_{fi}$ and $W_{ci}$ correspond to weight matrices for input messages to determine forget and candidate signal, respectively. Because changes in synaptic currents are made based on inputs at every time-step independent of the current synaptic current values, this strategy would eliminate the advantage of having sparse inputs. 

Another approach called “the SNNs with improved dynamics v2” is to use Y[n] which is a sparse and provides an approximation of I[n] as in following equations to update the synaptic current equation:

$$F[n]=σ(X[n] W_{fi}+Y[n-1]W_{fr} )$$

$$C[n]=ReLU(X[n] W_{ci}+Y[n-1]W_{cr} )$$

$$I[n]=F[n]  ⊙I[n-1]+(1-F[n])⊙C[n]$$

where ⊙ operator again corresponds to element-wise multiplication, $W_{fr}$ and $W_{cr}$ correspond to weight matrices for previous outputs to determine forget and candidate signal, respectively.

The authors also found that employing a constant threshold does not work effectively for various learning problems since the maximum range of the membrane potentials varies for distinct datasets. They use an exponential moving average (EMA) to monitor the membrane potential statistic. And estimate the membrane potential during the training to reach an appropriate threshold which is simplified as a linear function of membrane potential.

Another problem which should be solved during the training is gradient mismatch caused by using surrogate function to solve the non-differentiability problem of spiking neurons. The approach is to use multi-level mapping, where membrane potentials are mapped several values rather than zeros and ones. The idea is to map the values which are below the threshold to zero so that only sufficiently large membrane potentials are mapped to non-zero outputs by we assuming uniform mapping between membrane potentials and the outputs values.

## 2.2. My interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

LSTM and GRU models are used as a baseline. LSTMs, GRUs, and spiking neurons were stacked into different two-layer networks with 550 non-spiking or spiking units for each layer. Then, a fully connected layer received the final outputs and generated a probability of the most likely used words for speech recognition purpose. Inputs for neural networks were created based on Kaldi principle [4]. Feature space maximum likelihood linear regression [5] was used to convert raw audios into acoustic features. The features were calculated using 25 ms windows with a 10 ms overlap. All weight matrices excpect recurrent weight matrices which had orthogonal initialization were initialized based on Glorot’s scheme [6]. Adam used as the optimizer, and recurrent dropout used as the regularization method [7]. On all architectures, a dropout rate of 0:1 was discovered to provide the best performance. All architectures were trained for 24 epochs with a batch size of 64 and the starting learning rate was set to $10^{-3}$. The error on the development set was monitored every epoch and the learning rate was halved after 3 epochs of improvement less than 0.1. 

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

LSTM architecture surpass GRU architecture by a small amount.  First approach called “the SNNs with improved dynamics v1” increases the accuracy of the prediction by 8.98% and 9.84% on TIMIT and LibriSpeech 100h datasets, respectively compared to conventional SNNs. The increase in accuracy shows the significance of learning without the problems occurred on training such as gradient mismatch and vanishing gradient. Second approach called “the SNNs with improved dynamics v2” further boosted the values reached in v1 version by 1.64% and 5.57% on TIMIT and LibriSpeech 100h datasets, respectively. So, the result is that maximizing the recognition accuracy was made possible by choosing the forget and candidate signals depending on the past synaptic current. 

When it comes to computational savings, 4.37-5.40 x fewer number of sparse outputs multiplications caused by the proposed v1 SNN compared to GRUs. However, the SNN with improved dynamics v2 led to 10.13 and 11.14 x fewer multiplication operations compared to the GRUs on TIMIT and LibriSpeech 100h datasets, respectively. The difference in recognition accuracy between SNNs and the baseline LSTMs became 1.10% and 0.36% using all the proposed architecture and training on TIMIT and LibriSpeech 100h datasets, respectively with halved parameter numbers.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

[1] Ponghiran, W., & Roy, K. (2022, June). Spiking Neural Networks with Improved Inherent Recurrence Dynamics for Sequential Learning. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 7, pp. 8001-8008).

[2] Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. R. (2012). Improving neural networks by preventing co-adaptation of feature detectors. arXiv preprint arXiv:1207.0580.

[3] Han, K. J., Prieto, R., & Ma, T. (2019, December). State-of-the-art speech recognition using multi-stream self-attention with dilated 1d convolutions. In 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU) (pp. 54-61). IEEE.

[4] Povey, D., Ghoshal, A., Boulianne, G., Burget, L., Glembek, O., Goel, N.K., Hannemann, M., Motlícek, P., Qian, Y., Schwarz, P., Silovský, J., Stemmer, G., & Veselý, K. (2011). The Kaldi Speech Recognition Toolkit.

[5] Gales, M. J. F. (1998). Maximum Likelihood Linear Transformations for HMM-Based Speech Recognition. COMPUTER SPEECH AND LANGUAGE, 12, 75–98.

[6] Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. AISTATS.

[7] Semeniuta, S., Severyn, A., & Barth, E. (2016). Recurrent dropout without memory loss. arXiv preprint arXiv:1603.05118.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
