# @TODO: Paper title

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper “DualGraph: A graph-based method for reasoning about label noise” addresses the problem of being introduced unreliable labels derived from large-scale dataset preventing neural networks from fully exploring the data. Unlike the other approaches, the mentioned method , DualGraph, aims to capture structural relationships among labels at two different levels, namely the instance-level and distrubition-level.

## 1.1. Paper summary

The paper introduces two graph neural networks with a joint loss which is used to weight edge loss to enhance clean examples and weaken negative examples to prevent label noise.  Spesifically, the instance level relation make use of instance similarity to characterize sample category, while the distrubution-level relation express instance similarity distribution from each sample to all other samples.

# 2. The method and my interpretation

## 2.1. The original method

@TODO: Explain the original method.

## 2.2. My interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

The paper introduces the graph neural network capturing structural relationship among labels at two different levels. Moreover, it presents iterative optimization mechanism that contains two phases to train two graph neural networks alternatively. As a result, the obtained results tells us that Dual-Graph can effectively improve the classification accuracy of noise labels as mentioned in the paper.

Our results are close to those reported in the paper. However, we have used different dataset comparing to the mentioned paper’s since mentioned dataset is hard to build because of containing too much samples. We used the same graph network structure with slightly difference changes such as replacing GNN with GCN. Our approach is as follows : two graph neural networks are trained simultaneously, and the feedback path is formed between those to teach each other. All data is forwarded and possibly clean labels are being selected. After that, communication is performed between two Graph Neural Network to determine training data. At the end, back propagation is performed for selected data determined by peer network.

# 5. References

@TODO: Provide your references here.

# Contact

Kadir TANRIVERDI(kadir.tanriverdi@metu.edu.tr)
Osman Nuri IRKICATAL(nuri.irkicatal@metu.edu.tr)
