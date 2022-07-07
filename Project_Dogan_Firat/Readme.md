# Uncertainty-Matching Graph Neural Networks to Defend Against Poisoning Attacks

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

This is a preliminary version paper, which was published in September 2020. It aims to defend the model from poisoning attacks using uncertainty-matching graph neural networks (UM-GNN). Our goal is to implement the UM-GNN defined in the paper.

## 1.1. Paper summary

Graph neural networks are known for their usage in challenging tasks with the graph structure. On the other hand, they are vulnerable to adversarial attacks as regular neural network models. In this paper, they have contributed a new approach and labeled it as uncertainty-matching graph neural networks (UM-GNN). This network is a combination of a GNN and FCN with specific regulations. The regulations are basically taking the output from both models after the training process while contributing a new loss, which consists of the GNN model uncertainty value additionally to enhance the model from poisoning attacks.

# 2. The method and my interpretation

## 2.1. The original method

The method behind this approach is constructing a new loss function. The loss function is a combination of Cross-Entropy Loss of the GCN model output, aligning predictions over FCN and GCN, consisting of the uncertainty of the GCN model and KL divergence of the FCN model output. The intuition behind this new loss function is to avoid wrong learning by updating the gradients when given the poisoned dataset.

## 2.2. My interpretation 

We have implemented a GCN model, FCN model, and dataset builder, where the datasets are the same as mentioned in the paper. The critical point is also implemented, which is the loss function. In the source code, it can be visible that uncertainty matching functions are applied by using the Monte Carlo method. The paper did not specifically explain the key points of the UM-GNN. With deep research and reading, we have reached the main points of the essence of it.

# 3. Experiments and results

## 3.1. Experimental setup

The problem setup of our implementation is mostly the same as the given approach in the paper. The implementation of the GCN and FCN models may differ from others due to the creativity among humans. But the datasets are the same: Cora, Citeseer, and Pubmed.

## 3.2. Running the code

The project has a flat-code structure, that is no directory is present except the parent project directory. In the models.py file, we have GCN Fully Connected and VariationalDropout models. VariationalDropout model is taken from [here](https://github.com/elliothe/Variational_dropout/blob/master/variational_dropout/variational_dropout.py) with a small set of modifications. In the loss.py, we have all three loss functions present in the paper. In the solver.py, we have main script. running `python3 solver.py` is enough once you installed all the required dependencies.

## 3.3. Results

All history of my results are publicly available at [wandb](https://wandb.ai/adnanhd/um-gcn), which I have finetuned several hyperparameters to see if Dropout Inference is useful as it is stated in the paper as their novelty. Here showes some accuracy results to be compared.
![W B Chart 7_7_2022, 10 57 05 AM](https://user-images.githubusercontent.com/47499605/177722206-206d4459-7ae5-4e97-ac7e-e0d1823b4668.svg)

When the p value in the variational inference (stated as inference in the legend), is set to 0.2 instead of 0.0 almost 20% accuracy decrease occurs. However, random noise, as it is described as Random Noise Attack in the paper, between the vertices of the CiteSeer dataset, 20% 50% and 100%, does not reduce the accuracy.

# 4. Conclusion

From the results in (3.3), I can say that having a surrogate model predicting from a distribution might prevent the model from overfitting.

# 5. References

Shanthamallu, U. S., Thiagarajan, J. J., & Spanias, A. (2020). [Uncertainty-Matching Graph Neural Networks to Defend Against Poisoning Attacks](https://www.aaai.org/AAAI21Papers/AAAI-4382.ShanthamalluU.pdf). doi:10.48550/ARXIV.2009.14455

# Contact

Adnan Harun DOGAN: adnan.dogan@metu.edu.tr & adnanharundogan@gmail.com
