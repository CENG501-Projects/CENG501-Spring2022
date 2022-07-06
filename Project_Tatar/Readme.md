# @TODO: Paper title:Deterministic Mini-batch Sequencing for Training Deep Neural Networks Subhankar Banerjee, Shayok Chakraborty.

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

@TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).
Article,Deterministic Mini-batch Sequencing for Training Deep Neural Networks Subhankar Banerjee, Shayok Chakraborty,published 35th AAAI conference on Atificial Intelligence at 2021. The main goal of this project is to implement proposed alghoritm ,selecting and sequencing mini batches for training neural network. After implementing propesed alghoritm, obtained results will be compared with results obtained from SGD(Stoachastic Gradient Descent),DPP(Determinantal point process) and Submodular methods. 

## 1.1. Paper summary

@TODO: Summarize the paper, the method & its contributions in relation with the existing literature.
Summarize the paper, the method & its contributions in relation with the existing literature.

Generalization performance of the model is firmly related to mini batches which are used for computing gradients and updating parameters of the model during the backpropogation alghoritm(Gradient Descent). This leads to motivate to researcher devoloping intelligent sampling techniques rather than stoachastic ones. In this paper, authors propose an algorithm to generate a deterministic sequence of mini-batches to train a deep neural network. Their alghoritm based on idea which is selecting mini batch such that the data distrubution represented by this selected mini batch and the already selected one, is closest to the distrubution of the unselected training samples. MMD(Maximum Mean Disperancy) metric is used for measuring the closness of these distrubutions.Mini batch selection method can be formulated as an optimization problem with minimizing MMD metrics between the distrubutions. MMD is a statistical metric to compute the difference in marginal probablity between two distrubutions,which is calculated as the difference between the emprical means of the two distrubutions after mapping onto a Reproducing Kernel Hilbert Space. 
Contributions in existing literature can be listed as follows:
* This is the first attempt to use MMD metric to select mini batches for training DNN.
* Mini batch sequencing strategy is determinsitic and it is independent from the network architecture,task type(regression,classification etc..) and data in hand.
* Benchmark experimental studies was implmented to test the generallizability of the model on cahallenging data set and other methods like DPP,Submodular and SGD.
* Contrary to most of the methods, mini batch sequence can be pre computed indipendently for a given task
* It doesnt require extensive hyper-parameter tunning
* Alghoritm is based on solving Lineer Programing Problem.


# 2. The method and my interpretation

## 2.1. The original method

@TODO: Explain the original method.

Explain the original method.

Mini batch selection framework can be model as optimization problem. 
Symbol can be listed as follows:
* Phi fucntion represents a mapping from the input X space to RKHS(Reproduced Kernel Hilbert Space)
* Set of D represents training data set(N is the number of samples in subset of D).
* Subset of P represents already seleceted training samples (np is the number of samples in the subset of P)
* Subset Q represents unselected training samples in subset D(nq is the number of samples in the subset of Q)
* Function \Phi() represents mapping from the input space X to the RKHS(Reproduced Kernel Hilbert Space) H.
* k is the predetermined number of batch-size.(k = 50)
MMD between the selected and unselected set can be formulated as such : 

$$ MMD= ||( \frac{\sum_{i ∈ PuB}\phi(X_i)}{n_P+k} - \frac{\sum_{i ∈ Q\B}\phi(X_i)}{n_Q-k}||$$

Defining a vector $$ m ∈ (0,1)^{n_Q} $$ . m vector is binary vector and can be initilized randomly before implmenting the optimization alghoritm. If mi=1 that means sample xi in the subset of Q should be selected and be transferred from Q to P. Otherwise (mi=0), it shouldnt be selected.Using properties of RKHS such as reproducing property the MMD objcetive can be simplified as such: 

Minimize for m : $$ MMD= m^T\phi_1 m - \phi_2^{T} m + \phi_3^{T}m  $$  

Such that :  $$ m_i∈(0,1), ∀i , \sum_{i=1}^{n_Q} m_i = k   $$  

Matrix \phi_1 ,vectors \phi_2 and \phi_3 are defined as : 

$$ \phi_1(i,j) = \phi(xi,xj); (xi,xj)∈Q ,∀i,j  $$ 

$$ \phi_2(i) = \frac{n_P+k}{N}\sum_{j=1}^{n_Q} \phi(xi,xj);(xi,xj)∈Q,∀i,j  $$ 

$$ \phi_3(i) = \frac{n_Q-k}{N}\sum_{j=1}^{n_P} \phi(xi,xj); xi∈Q,xj∈P , ∀i,j  $$ 

This kernel evalutaions must be in order. Due to binary integer constaint on m, all kernel matrix and vector can be combined in Z matrix such:

$$ Z(i,j) = \phi_1(i,j); i != j, Z(i,j) = \phi_1(i,j)-\phi_2(i) + \phi_3(i) ; i = j  $$ 

Rewriting the objective with considering Z matrix such : 

Minimize for m : $$ MMD= m^T Z m  $$  

Such that :  $$ m_i∈(0,1), ∀i , \sum_{i=1}^{n_Q} m_i = k   $$  

This is an quadratic integer programming(IQP). Authors propesed a method for efficient LP relaxation to this IQP proplem to make ILP(Integer Lineer Programming).

Defining a binary matrix W with size of nQ by nQ and rewrite the optimization problem with introducing this matrix such:

Minimize for m and W : $$ MMD= \sum_{i,j}z_{ij}w_{ij}  $$  

Such that :  $$ w_{ij}=m_i m_j ; m_i,w_{ij}∈(0,1) ,∀i,j ; \sum_{i=1}^{n_Q} m_i = k   $$ 

The main idea behind the Lp relaxation is manipulate the quadratic term wij as lineer term. This is related to sign of zij. For selecting mini batches both mi and mj should be 1.

Rationality behind this idea based on :

* if zij<0: quadratic equation can be written lineer constarint: -mi-mj + 2wij<=0.
 if mi and mj are both zero or one of them zero and other one is one,wij is forced to equaş zero. When both are one, wij can be both zero or one.
 Minimization of the MMD means wij is forced to be 1. 

* if zij=>0: quadratic equation can be written lineer constarint: +mi+mj - 2wij-1<=0.
 if mi and mj are both one ,wij is forced to equaş one. When both are zero or one of them is zero, wij can be zero or one.
 Minimization of the MMD means wij is forced to be 0. 
 
So the ILP problem van be formulated as :

Minimize for m and W : $$ MMD= \sum_{i,j}z_{ij}w_{ij}  $$  

Such that :  $$ -m_i-m_j+ 2w_{ij}=<0 \for z_{ij}<0\ ,∀i,j ; \sum_{i=1}^{n_Q} m_i = k   $$ 

Such that :  $$ +m_i+m_j =< 1 + 2w_{ij} \for z_{ij}>0\ ,∀i,j ; \sum_{i=1}^{n_Q} m_i = k   $$ 

Propes alghortim : 

Require Training Data D with N samples, already selected set of training samples P,set of unselceted training samples Q, mini batch size k,kernel function.
*iterate over Subset of Q:
* 1 - Compute matrix \phi_1 and the vectors \phi_2 and \phi_3
* 2 - Compute Matrix Z
* 3 - Solve The LP problem 
* 4 - Construct the m vector
* 5 - Selcet the mini batch from Q based on the entries m
* 6 - Go step 1 and iterate 
* Stop the İterations

## 2.2. My interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

Explain the parts that were not clearly explained in the original paper and how you interpreted them.

Parts related to how to solve ILP problem, corresponds the 3th step of the alghortim, was not clearly stated. Authors used  off-the shelf ILP solver to solve this problem. I extensively researched off the shelf ILP solver like LPSolve IDE or Python library like Pulp, but I did not utilize this ILP solver into this problem. So I decided to solve in house python code. But, I did not finish this submodule due to fallowing reasons : 

* 1 - Computations of kernel matrix (Phi1) has quadratic complexity and can be very challenging for large data set in terms of computational effort.At each iteration Phi1 matrix is computed.
* 2 - I could not find which off the shelf ILP solver was implmented in the Article. 
* 3 - I could not implement Step 3 in the alghortim due to conditional constraint and dynamic nature of the ILP problem. For sake of completness I would like to present  my uncomplete code in free format at this section such that: 

* def kernel(x,y):
  * sigma = 1 
  * return ((x**2+y**2-2*sigma**2)/4)*math.exp(-1*(x**2+y**2)/2*sigma**2)
  
* phi_2_list = []
* for i in Q:
  * for j in Q: 
    * phi_2 = np.sum(kernel(i,j))*(np+k)/N 
    * phi_2_list.append(phi_2)
    * phi_2_list = np.array(phi_2_list)
  phi_2_list.reshape(nQ)
  
* phi_3_list = []
* for i in Q:
    * for j in P:
        * phi_3 = (nq-k)/N * np.sum(kernel(i,j)) 
        * phi_3_list.append(phi_3)
        * phi_3_list = np.array(phi_3_list)
   * phi_3_list.reshape(nQ)
   
* phi_matrix_list = []
* for i in Q:
  * for j in Q: 
    *phi_matrix =np.array(kernel(i,j)
    * phi_matrix_list.append(phi_matrix)
  *phi_matrix.reshape(nQ,nQ)
  
 * Z_matrix_list = []
 * for i in range((phi_matrix.shape[1])):
    * for j in range((phi_matrix.shape[1])):
      * if i == j:
      * Z = phi_matrix[i,j]-phi_2_list[i]+phi_3_list[i]
    * else:
      * Z = np.array(phi_matrix[i,j]).reshape(1)
      * Z_matrix_list.append(Z)
* Z_matrix_list = np.array(Z_matrix_list)
* Z_matrix_list=Z_matrix_list.reshape(3,3)
* Z_matrix_list = np.array(Z_matrix_list).reshape(1,nQ*nQ)

* for i in range(iteration):
  * m = np.random.randn(nQ)
  * m_i_list = []
  * for i in range(nQ):
    * m_i = m[i]
    * for j in range(nQ):
      * z = Z[i,j]
      * m_j = m[j]
      * w_ij = m[i]*m[j]
      * MMD = z*w_ij
      * if z <0 :
        * ILP solver Part with 2 constarint(one of them is related to LP relaxation, other one is related to pre dtermined batch size(k)) plus binary constarint
      * else:
        * ILP solver part with 2 constarint(one of them is related to LP relaxation, other one is related to pre dtermined batch size(k)) plus binary constarint 
  * m_i_list.append(m_i)

Note: I dont want add extra file for this uncomplete code to mislead anybody who reads to this readme file. This is my first entry in Github so I am sorry for inconvenience.  

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

Describe the setup of the original paper and whether you changed any settings.

Orginal Settings can be listed as follows : 

* 1 - Architecture : Resnet18 pre trained model for Backbone Architecture.
* 2 - Regularization coefficient = 0.0005
* 3 - Learning rate = 0.001 10% decrease in each epoch
* 4 - Optimization Alghoritm : SGDM with Momentum Parameters = 0.9
* 5 - Total Number of Epoch : 35
* 6 - Pre determined Batch size: k = 50 
* 7 - Gaussian Kernel with parameter 1 for MMD calculations

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

Due to reasons listed at Section 2, I could not complete the project.I am able to obtain Results for CIFAR 10 data set with Random sampling.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

Random sampling strategy :Test Accuracy is 83.%

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

I could not obtain the results but article offers a very elegant way to select and sequence mini batches to train the model. Generalization performance of the network computed by the test loss and test error metrics on the CIFAR 10,MNIST and SVNH data sets. Propes alghoritm always give equal or better performance than Random Sampling. The submodular selection methods, performs better on the SVNH data set but it is not consistent across different data set.DPP methods, on the other hand, perform better ın CIFAR 10 data set but for other data set gives slower decrease in test loss and error. 


# 5. References

@TODO: Provide your references here.

Deterministic Mini-batch Sequencing for Training Deep Neural Networks
Subhankar Banerjee, Shayok Chakraborty

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.

Provide your names & email addresses and any other info with which people can contact you.
email: Utatar4@gamil.com. If anybody who interested in this article and it's code implmentation, I will be very appreciated to receive any help and collabaration.
Thank you in advance.
