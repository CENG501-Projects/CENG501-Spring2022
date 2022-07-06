# @TODO: Paper title

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

We are working on the paper named "ON STATISTICAL BIAS IN ACTIVE LEARNING: HOW AND WHEN TO FIX IT". It is published in 2021 by Sebastian Farquhar, Yarin Gal and Tom Rainforth. Paper is mainly focusing on bias in active learning and creating new methods for decresing this bias. 

## 1.1. Paper summary

As we know that train step is the one of the most crucial parts in deep learning. But, in this step we have one problem. If we have a huge dataset for training, we may encounter with problems in training such as time limit etc. So, there is as clever solution for this problems, Active Learning. In this method, we are selecting most informative data instances from dataset and train our model with them. After that, we can get nearly the same accuracy as model which is trained with all of the train data. However, there is also a crucial problem, bias. When we train our model with less data instance, sometimes, our model will be act more biasly. Because, our dataset in Active Learning may not follow the population (All train data) distribution. So, our paper suggest a great improvment in that problem. 

Paper suggest a solution which minimizing the bias in active learning while not changing current Active Learning methods. In this paper, we are using pool based active learning. This means we have tow sets called D_pool and D_train. We are training our model with D_train and after train step we test model with D_pool dataset. Afterwards, according to their scores (Scores we get from q function (BALD etc.)), we select instance with least certainity value and remove move it to D_train and train our model again. 

Other than algorithm, it build two risk estimators called R_Pure (Plain Unbiased Risk Estimator) and R_Lure (Levelled Unbiased Risk Estimator). We can think risk estimators as loss formulas. By using these estimators, we can get unbiased Active Learning algorithm.

This is the algorithm from paper:
 
![1 1](https://user-images.githubusercontent.com/62703461/177638538-59b2c37c-b818-4968-8977-5a6c6bf326ac.png)


# 2. The method and my interpretation

## 2.1. The original method

@TODO: Explain the original method.

## 2.2. My interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

Firstly, we need to import required libraries. Click only "run" button of first cell.
Secondly, we need to open/import MNIST dataset. It is really easy with Google Colab. After that code will prepare dataset to appropriate form for train and test. 
Afterwards, we are creating CNN and also we have many configurations. These are configurations for model: 

Conv 1            : 1-16 channels, 5x5 kernel, 2x2 max pool
Conv 2            : 16-32 channels, 5x5 kernel, 2x2 max pool
Fully connected 1 : 128 hidden units
Fully connected 2 : 10 hidden units
Loss function     : Negative log-likelihood
Activation        : ReLU
Batch Size        : 64


## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

Running (Train, test etc.) is really easy in our implementation. We created our project with Google Colab and it is divided into cells (Imports, Train Part etc.). We need to click a button at the top of each cell to run it. We do not download any file or dataset. We are using MNIST Dataset and we can import it easily by using Google Colab. 

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
