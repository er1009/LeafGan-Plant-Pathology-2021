# LeafGan-Plant-Pathology-2021


Neural Network for
Computer Vision
Final Project

LeafGAN
 23.04.22

Eldad Ron 207021916
Bar Yacobi 315471367
Jacob Monayer 206993198

















Abstract
Class Imbalance Problem is the case in which certain classes are over-represented than other classes. Training a network on an imbalanced dataset will make the network to be biased towards learning more representations of the data-dominated classes, underrepresented classes will be under looked.
In this work we used LeafGan architecture to overcome this unbalanced dataset problem. The dataset we are working with is the Plant Pathology 2021 – FGVC8, that has an unbalanced problem (fig 1). We synthesized minority class disease images using the LeafGan to make the dataset more balanced.
LeafGan is a novel image-to-image translation system with own attention mechanism. LeafGAN generates a wide variety of diseased images via transformation from healthy images. It's built on CycleGAN and a proposed label-free leaf segmentation module (LFLSeg) to guide the network in transforming the relevant regions (leaf areas) while preserving the backgrounds.














Introduction
A generative adversarial network (GAN) is a class of machine learning frameworks in which two neural networks contest with each other in a game (in the form of a zero-sum game, where one agent's gain is another agent's loss). Given a training set, this technique learns to generate new data with the same statistics as the training set.
The core idea of a GAN is based on the "indirect" training through the discriminator, another neural network that can tell how much an input is "realistic", which itself is also being updated dynamically (fig 1). This basically means that the generator is not trained to minimize the distance to a specific image, but rather to “fool” the discriminator. This enables the model to learn in an unsupervised manner. When used for image generation, the generator is typically a deconvolutional neural network, and the discriminator is a convolutional neural network (fig 2).









Advantages: 
Better modeling of data distribution (images sharper and clearer)
In theory, GANs can train any kind of generator network. Other frameworks require generator networks to have some specific form of functionality, such as the output layer being Gaussian.
Disadvantages: 
Hard to train, unstable. Good synchronization is required between the generator and the discriminator, but in actual training it is easy for D to converge and G to diverge. D/G training requires careful design.
Mode Collapse issue - the learning process of GANs may have a missing pattern, the generator begins to degenerate, and the same sample points are always generated, and the learning cannot be continued.

GANs became extremely popular in recent years and a lot of studies are being conducted in this domain (fig 3).






The Proposed Solution
To overcome the imbalance problem in our plant pathology dataset we trained LeafGAN model to synthesize the minority classes of diseased images. The LeafGAN architecture is built on CycleGAN (Image to image translation) and a proposed label-free leaf segmentation module (LFLSeg) to guide the network in transforming the relevant regions (leaf areas) while preserving the backgrounds.
Image to image translation
Image to image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. CycleGAN is capable of learning to translate an image from a source domain X to a target domain Y in the absence of paired examples (fig 4). This is an important solution because our dataset is made of unaligned pairs (Healthy leaf, Diseased leaf).




CycleGAN 
inverse mapping - Because the classic GAN mapping (G: X -> Y, presented in the Introduction) is highly under-constrained, CycleGAN couple it with an inverse mapping        (F: Y -> X) (fig 5), and introduce a cycle consistency loss to enforce F(G(X)) ≈ X (and vice versa) (fig 5).



Consistency loss - the CycleGAN encourages cycle consistency by adding an additional loss to measure the difference between the generated output of the second generator and the original image, and the reverse. This acts as a regularization of the generator models, guiding the image generation process in the new domain toward image translation (fig 6).




identity loss - It is often beneficial to add the identity loss - the identity loss is simple, G(y) ≈ y and F(x) ≈ x. Adding an identity loss generally helps preserve color and tint in translated images, especially for photo to image generations (fig 7).

In total, we have four losses, two adversarial losses, cycle loss and the identity loss, but these losses aren’t equally important. We may want to give a greater importance to the cycle loss than the adversarial loss or vice versa. To solve this, the authors introduce λ. In the final loss equation, the term for cycle loss is multiplied by λ, as to give it more or less importance. The authors set λ equal to 10 for their experiments. 
the authors define this as the full loss function:





LFLSeg 
LFLSeg is simple but effective weakly supervised label-free leaf segmentation module that helps the classification model to learn the dense and interior leaf regions implicitly. From an architecture point of view, the backbone of LFLSeg is a simple CNN, and is designed to discriminate between "full leaf", "partial leaf", and "non-leaf" objects. Specifically, "full leaf" objects are images that contain a single full leaf, while "partial leaf" objects are images that contain part of a "full leaf", and "non-leaf" objects do not contain any part of a leaf.
The segmented leaf region is obtained using a heatmap with respect to the "full leaf" class by applying the Grad-CAM technique. This heatmap is a probability map representing the contribution of each pixel to the final decision of the "full leaf" class, and thus can be used as a binary mask after thresholding with a specific threshold value δ.
The adversarial loss of the LeagGan with the LFLSeg masks is:


Note that ys = Sy * y is the masked version of the image y ∈ Y, where Sy = LFLSeg(y) is the masking which represents the leaf area after feeding image y to the LFLSeg module.

Method
In this work, we train our LeafGAN models to generate new images of apple diseases. We used the plant pathology 2021 dataset resized to 900x900. We trained Resnet152 model to identify the apple disease classes. We noticed that this dataset is unbalanced, and our model is biased toward the majority classes, therefore we are going to synthesize disease images with the help of LeafGAN.


Training LFLSeg module to obtain image masks
For obtaining the image’s masks we trained the LFLSeg module. We created the dataset for training the LFLSeg module, containing 3 classes: 
●	Full leaf: images that contain full apple leaf, this class is made of the plant pathology we downloaded from Kaggle - 18,632 images.
●	Partial leaf: images that contain a part of a full leaf, we created this class by randomly choosing 1/9 of the full leaf class, we cropped each image to 9 patches - 18,632 images.
●	Non leaf: images that do not contain any part of a leaf. We created this class by randomly downloading 100 classes and 150 images for each class from the ImageNet dataset - 15,000 images.
Images from the 3 classes with GradCam Heat Map:










We used the ResNet-101 model weights that the original authors of the LeafGAN offer in their GitHub repo, and fine-tuned it to our plant pathology dataset as the backbone of LFLSeg. We trained our model for 100 epochs with the hyperparameters, and augmentations recommended in the LeafGAN article.
After training the LFLSeg, we experimented with different thresholds, to choose the optimal threshold to obtain the best mask results (fig 8).








Training LeafGAN models to synthesize images
We chose 4 diseases to synthesize images to, based on class distribution. 
In order to synthesize the diseased images we trained 4 LeafGANs (for each disease domain) models:  healthy -> rust
  healthy -> Complex
  healthy -> Frog Eye Leaf Spot
  healthy -> Powdery Mildew
For each model we created a dataset as follows:
  TrainA – 80% Healthy leaf images (Domain A)
  TrainA_masked- Masks for all the images in TrainA
  TestA – 20% Healthy leaf images (Domain A)
  TrainB – 80% Disease leaf images (Domain B)
  TrainB_masked- Masks for all images in TrainB
  TestB – 20% Disease leaf images (Domain B)
We used the LFLSeg module for generating masks for all training images.

Each model was trained for 100 epochs, with the hyperparameters, and augmentations recommended in the LeafGAN article. The training time for each model was approximately 60 hours.
After training we experimented with the model weights for each model, to find out in which epoch the model generates the best image. It's interesting to see that the model is probably converging around epoch 75 (fig 9).
















During training, we examined the images in loss manner (real, fake, reconstructed and identity) every epoch to monitor the model learning process:
Real A                                                      Real B





Fake A                                                   Fake B





Idt A                                                      Idt B





Rec A                                                        Rec B




Synthesize Images
To balance our dataset, we examined each class distribution. Based on that distribution we decided how many images to generate for each class.
Translation methods:
we experimented with 3 image translation methods with the LeafGAN:
●	The trivial translation, the one the models was trained to: Healthy -> Disease
●	Feed forward Healthy image through 2 different generators: 
Healthy -> Disease1 -> Disease2. Final product: Disease1+Disease2.
●	Feed forward Diseased image through another disease generator:
Disease1 -> Disease2. Final product: Disease1+Disease2.
The two last translations are particularly beneficial for our dataset because it is multi-labeled, although we didn't train the LeafGAN models for those translations.  

Adding synthetic images to training set:

We split our original dataset to train and validation (80/20) and then we generated images to be added to the training set as follows:

●	Healthy -> Powdery mildew - 166 images (new label: Powdery mildew)
●	Healthy -> Complex - 198 images (new label: Complex)
●	Healthy -> Rust - 140 images (new label: Rust)
●	Healthy -> Frog Eye Leaf Spot - 196 images (new label: Frog Eye Leaf Spot)
●	Healthy -> Rust -> Frog Eye Leaf Spot - 100 images (new label: Rust + Frog Eye Leaf Spot)
●	Scab -> Frog Eye Leaf Spot - 112 images (new label: Scab + Frog Eye Leaf Spot)

To avoid data leaking and reduce the over-represented classes (Healthy and Scab) we removed the original images that used to generate the synthesized images.

Training set labels distribution before (fig 10) and after (fig 11) adding the synthetic images.

  






Results
translation	Real Target domain image	Good Fake generated image	Bad Fake generated image
Healthy -> rust	



Healthy -> Frog Eye Leaf Spot	



Healthy -> Powdery Mildew	



Healthy -> Complex	



Healthy -> Rust -> Frog Eye Leaf Spot 	



Scab -> Frog Eye Leaf Spot	




Examination of the generated images -
We carefully examined all generated images and compared them to the real images in the dataset, to further understand the LeafGAN generating capabilities.
●	Basic translations the GAN was trained on - when translating healthy images to rust, frog eye leaf spot and complex we got really good and realistic results, although in some cases the LeafGAN was messing with the background. It was more challenging for the LeafGAN to translate to powdery mildew, in a lot of the cases it seems to miss the leaf and implant the disease on the background, therefore the generated images look fake.
●	Feed forward healthy image through 2 generators - although we didn't train the LeafGAN to translate images in that way, surprisingly we got nice results when translating healthy images to rust and then to frog eye leaf spot, ending up with multi diseased image (rust +  frog eye leaf spot). The generated images look real when comparing it to the original images from the dataset.
●	Feed forward disease image through another disease generator - like the translation mention above we didn't train the LeafGAN to translate images in that way, again we got good results when translating scab disease images to frog eye leaf spot, ending up with multi diseased image (scab + frog eye leaf spot). The synthetic images look very realistic compared to the original images from the dataset.
Experimenting with various translations, shows the LeafGAN high performance in image translation even on translation tasks it did not train on, it reveals the big potential of GANs in general.
LeafGAN graph examination
We integrated the original LeafGAN code to the wandb platform for graph visualization, to examine and monitor all the losses through training.
Cycle loss - the LeafGAN cycle loss was reduced during training for both domains (A and B) in all models as expected (fig 12). 









Generator loss vs. discriminator loss - it's interesting to see that we got some negative correlation between the generator and discriminator losses. This is also expected as the generator and discriminator are competing with each other (fig 13).








Identity loss - the LeafGAN identity loss was reduced during training for both domains (A and B) in all models as expected.






To evaluate the LeafGAN performance, we added generated images to the plant pathology dataset to balance the data using the method mentioned in the proposed solution section.
We trained a resnet152 model to predict 6 labels (complex, frog_eye_leaf_spot, powdery mildew, rust, scab, healthy), first on the original dataset without the synthetic images, after that we added the synthetic images and train the model on the new dataset. 
Then we compared the performance (f1-score) of the two models to evaluate the LeafGAN beneficial effect on the imbalance dataset (fig 15).

Validation f1





















Test Classification Results
Model	Label	Precision	Recall	F1-Score





Resnet152 Original dataset	Complex	0.73	0.74	0.74
	Frog Eye Leaf Spot	0.83	0.86	0.85
	Powdery Mildew	0.79	1.0	0.88
	Rust	0.87	0.98	0.92
	Scab	0.89	0.87	0.88
	Healthy	0.98	0.93	0.96
	Average	0.87	0.89	0.88





Resnet152 generated dataset	Complex	0.76	0.71	0.73
	Frog Eye Leaf Spot	0.86	0.89	0.87
	Powdery Mildew	0.87	0.99	0.92
	Rust	0.90	0.98	0.94
	Scab	0.93	0.89	0.91
	Healthy	0.97	0.97	0.97
	Average	0.90	0.90	0.90
*Bold = higher score between the two models
By comparing the two models results, we can see that the model that was trained on the generated images dataset is outperforming the model that was trained on the original dataset, although the performances are quite similar. 
Summary and Conclusion
The LeafGAN is a powerful image translation model that can be used for various tasks. It is able to produce realistic disease images that can be used for data augmentations in agriculture computer vision tasks. 
Our main goal was to overcome the imbalance dataset problem. With the help of the LeafGAN, we did improve our classifier performances on identifying diseased apple leaves but in small percentage (~2%).
It looks like with further work we can improve the performances even more, maybe by changing the heuristic of adding the synthetic images and the amount of the synthetic images being added to the dataset.
Due to hardware limitations, we trained our LeafGAN models for 100 epochs, unlike the recommended 200 epochs by the paper. Maybe training for more epochs can lead us to better results.
Further work can be done to improve images masks that are fed to the LeafGAN network, to separate more accurately the leaves from the background and improve LeafGAN attention mechanism.
In conclusion, it was a very interesting experience working with this advanced CycleGan architecture. During our work we faced some challenges due to the fact that training a GAN is very different from other machine learning projects we worked on before. Eventually, we’re satisfied with the results and learned a lot during the making of this project. 

References
[1] Cap, Q. H., Uga, H., Kagiwada, S., & Iyatomi, H. (2020). Leafgan: An effective data augmentation method for practical plant disease diagnosis. IEEE Transactions on Automation Science and Engineering.‏
[2] Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision (pp.2223-2232).
[3] LeafGAN github repository:  https://github.com/IyatomiLab/LeafGAN


