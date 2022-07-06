# High-Fidelity and Arbitrary Face Editing

This readme file is an outcome of the [CENG501 (Spring 2022)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 2022) Project List](https://github.com/CENG501-Projects/CENG501-Spring2022) for a complete list of all paper reproduction projects.

# 1. Introduction

High-Fidelity and Arbitrary Face Editing is a paper aimed at improving the outputs in face editing techniques. The paper was published at Computer Vision and Pattern Recognition journal in 2021. The paper focuses on the problems of the cycle consistency. Cycle consistency is widely used for face-to-face translation. However, it is observed that generator finds a way to hide information especially rich details such as wrinkles and moles. Our implementation focuses on feeding generator with high-frequency information of the input image so that we could avoid information hiding.

## 1.1. Paper summary

Face editing is a process where an attribute of the face is changed while the whole other parts are remin unchanged. Significant improvement in the Generative Adversarial Networks (GANs) produce satisfactory results for the face editing like StarGAN and RelGAN. However they lack of rich details and some outputs are blurry. In order to prevent this problem HifaFace is produced. HifaFace feeds the generator with high-frequency components of the input images. Input images are decomposed while using Haar wavelet tranformation so that images are transformed into multiple frequency domains. High-frequency domains are fed into the generator while using Wavelet-based-skip-connection. . Furthermore, our method is able to effectively exploit large amounts of unlabeled face images for training, which can further improve the fidelity of synthesized faces in the wild. Powered by the proposed framework,high-fidelity and arbitrarily controllable face editing results outcome.
# 2. The method and my interpretation

## 2.1. The original method

Paper proposes solving the problem of cycle consistency from two perspectives. On the one hand, high-frequency signals are decomposed from the input image and fed into the end of the generator to mitigate the generator's struggles for synthesizing rich details. With this process generator gives up to encode the hidden information. On the other hand, additional discriminator is added to prevent the generator to lose rich details. 

Proposed method consists mainly four parts which are respectively Wavelet-Based Generator G, High-Frequency Discriminator DH, Image-Level Discriminator DI and  Attribute Classifier C. Structure of the method is given in the following figure.
![WhatsApp Image 2022-07-06 at 21 01 33 (2)](https://user-images.githubusercontent.com/60968544/177636010-9e073e85-792e-497c-9034-6fe22cd2b45f.jpeg)

Wavelet transformation is usually applied at the image level, but here we implement it at the feature level. Firstly, wavelet pooling is applied in order 
to extract features in the domains of different frequencies from different layers of the encoder. Haar wavelet tranformation is chosen for this process. Then we ignore the information of LL, and apply wavelet unpooling to LH, HL and HH to reconstruct the information for high-frequency components of the original feature. This transformation is applied both in the generator and discriminator. Major difference between the other face editing methods and HifaFace is this Haar wavelet transformation process.

![WhatsApp Image 2022-07-06 at 21 01 33 (1)](https://user-images.githubusercontent.com/60968544/177638389-57a726ff-f62b-4142-86cb-67ea5de2aef3.jpeg)


## 2.2. My interpretation 
We had some difficulties while implementing the code for this paper. <br/>
Selecting dataset and preprosessing was difficult so rather than using CelebA-HQ we have used CelebA. Chosen dataset has lower quality.  <br/>
There are some misleading dimension sizes written in the actual implementation explanation in the paper. Especially in the convolution chennel size parameters were not compatible. Thus we have changed the channel size.  <br/>
In the classification step, non-linaerization method was not described in the paper. Thus we have chosen Leaky ReLu. We have chosen this activation function because Leaky ReLu is used in the other modules as activation function. Furthermore we know that Leaky ReLu prevent dead ReLu problem and has better convergence.  <br/>
Also in the loss functions definition there is a missing part. Attribute Regression Loss has a hyperparameter α. Its interval is given as α ∈ [0, 2].
We expect to control the attributes with athis scale factor α. However what value is used while training and testing is not given in the paper. So we set α to 1.


# 3. Experiments and results

## 3.1. Experimental setup

The dataset setup was slightly changed. The datasets used in the original paper are High quality but the ones we have used are the low quality versions of the same datasets due to the limited computing power we have with our local computers and the limitations in Google Colab.

## 3.2. Running the code

The code was implemented using Google Colab, so uploading the python file to the GoogleColab and executing code blocks following the structure can make it run. The network architectures and functions overall are implemented according to the paper.

## 3.3. Results

We didn't have the computational power in to produce any results. We have tried using Google Colab in order to produce any result but we couldn't manage without overstepping the daily quota. With this reason we couldn't control the errors and implementation mistakes in our code. 


# 4. Conclusion
HifaFace has better results when it is compared to other GAN implementations. Especially RelGAN and STGAN is analyzed and the main difference between these face editing tools is the generator and discriminator structure. HifaFace has the advantage of the keeping the details of the some attributes in the face thanks to wawelet transformation process. The key idea of this proposed method is to adopt a wavelet-based generator and a high-frequency discriminator. Moreover, a novel attribute regression loss is designed to achieve arbitrary face editing on a single attribute. Extensive experiments demonstrate the superiority of this framework for high-fidelity and arbitrary face editing. 

We couln't generate the results which brings no comparison between our results and the results in the paper. 
 
![WhatsApp Image 2022-07-06 at 21 01 33](https://user-images.githubusercontent.com/60968544/177644510-150d4955-26b4-475d-a94a-2e1edfbe2f62.jpeg)



# 5. References

[Google Colab](https://colab.research.google.com/)  <br/>
[FFHQ](https://paperswithcode.com/dataset/ffhq)  <br/>
[CelebAHQ](https://paperswithcode.com/dataset/celeba-hq)  <br/>
[PyWavelets](https://github.com/PyWavelets/pywt)  <br/>


@inproceedings{gao2021hifaface,  <br/>
  title={High-Fidelity and Arbitrary Face Editing},  <br/>
  author={Gao, Yue and Wei, Fangyun and Bao, Jianmin and Gu,  <br/>
          Shuyang and Chen, Dong and Wen, Fang and Lian, Zhouhui},  <br/>
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},  <br/>
  year={2021}  <br/>
}

# Contact
Belemir Kürün - [kurun.belemir@metu.edu.tr](kurun.belemir@metu.edu.tr)  <br/>
Arda Karaman - [arda.karaman@metu.edu.tr](arda.karaman@metu.edu.tr)  <br/>
