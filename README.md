# LeafGan-Plant-Pathology-2021

![](RackMultipart20220611-1-s4b6n0_html_85ca6e4cb94458e9.png)

![Shape2](RackMultipart20220611-1-s4b6n0_html_51a8cbd7ca258007.gif) ![Shape1](RackMultipart20220611-1-s4b6n0_html_51a8cbd7ca258007.gif)

fake rust image

fake complex image

![](RackMultipart20220611-1-s4b6n0_html_88117be0916815a0.png) ![](RackMultipart20220611-1-s4b6n0_html_e6e50b216465e7dc.png)

![Shape3](RackMultipart20220611-1-s4b6n0_html_51a8cbd7ca258007.gif)

Real healthy image

![](RackMultipart20220611-1-s4b6n0_html_47b6b22a8f876963.jpg)

![Shape4](RackMultipart20220611-1-s4b6n0_html_e1acb3c25ba11bcf.gif) ![Shape5](RackMultipart20220611-1-s4b6n0_html_ba702043948a2d4f.gif)

fake powdery mildew image

fake frog eye leaf spot image

![](RackMultipart20220611-1-s4b6n0_html_1b9d4d6b197d6f55.png) ![](RackMultipart20220611-1-s4b6n0_html_c2bfbc0d0b4c2c19.png)

**Abstract**

Class Imbalance Problem is the case in which certain classes are over-represented than other classes. Training a network on an imbalanced dataset will make the network to be biased towards learning more representations of the data-dominated classes, underrepresented classes will be under looked.

In this work we used LeafGan architecture to overcome this unbalanced dataset problem. The dataset we are working with is the Plant Pathology 2021 – FGVC8, that has an unbalanced problem (fig 1). We synthesized minority class disease images using the LeafGan to make the dataset more balanced.

LeafGan is a novel image-to-image translation system with own attention mechanism. LeafGAN generates a wide variety of diseased images via transformation from healthy images. It&#39;s built on CycleGAN and a proposed label-free leaf segmentation module (LFLSeg) to guide the network in transforming the relevant regions (leaf areas) while preserving the backgrounds.

![](RackMultipart20220611-1-s4b6n0_html_8422b4361039a502.jpg)

![Shape6](RackMultipart20220611-1-s4b6n0_html_e891786e18deec41.gif)

CycleGan architecture

**Introduction**

A generative adversarial network (GAN) is a class of machine learning frameworks in which two neural networks contest with each other in a game (in the form of a zero-sum game, where one agent&#39;s gain is another agent&#39;s loss). Given a training set, this technique learns to generate new data with the same statistics as the training set.

The core idea of a GAN is based on the &quot;indirect&quot; training through the discriminator, another neural network that can tell how much an input is &quot;realistic&quot;, which itself is also being updated dynamically (fig 1). This basically means that the generator is not trained to minimize the distance to a specific image, but rather to &quot;fool&quot; the discriminator. This enables the model to learn in an unsupervised manner. When used for image generation, the generator is typically a deconvolutional neural network, and the discriminator is a convolutional neural network (fig 2). ![](RackMultipart20220611-1-s4b6n0_html_2d2213628d7d7b52.png)

![](RackMultipart20220611-1-s4b6n0_html_d84e6e8da73454b6.png)

![Shape8](RackMultipart20220611-1-s4b6n0_html_a38eea10a6cd1d2f.gif) ![Shape7](RackMultipart20220611-1-s4b6n0_html_a38eea10a6cd1d2f.gif)

_Fig_ 2

_Fig_ 1

Advantages:

Better modeling of data distribution (images sharper and clearer)

In theory, GANs can train any kind of generator network. Other frameworks require generator networks to have some specific form of functionality, such as the output layer being Gaussian.

Disadvantages:

Hard to train, unstable. Good synchronization is required between the generator and the discriminator, but in actual training it is easy for D to converge and G to diverge. D/G training requires careful design.

Mode Collapse issue -the learning process of GANs may have a missing pattern, the generator begins to degenerate, and the same sample points are always generated, and the learning cannot be continued.

G ![](RackMultipart20220611-1-s4b6n0_html_dcb3d5a4656f36ae.jpg) ANs became extremely popular in recent years and a lot of studies are being conducted in this domain (fig 3).

![Shape9](RackMultipart20220611-1-s4b6n0_html_a38eea10a6cd1d2f.gif)

_Fig_ 3

**The Proposed Solution**

To overcome the imbalance problem in our plant pathology dataset we trained LeafGAN model to synthesize the minority classes of diseased images. The LeafGAN architecture is built on CycleGAN (Image to image translation) and a proposed label-free leaf segmentation module (LFLSeg) to guide the network in transforming the relevant regions (leaf areas) while preserving the backgrounds.

**Image to image translation**

Image to image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. CycleGAN is capable of learning to translate an image from a source domain X to a target domain Y in the absence of paired examples (fig 4). This is an important solution because our dataset is made of unaligned pairs (Healthy leaf, Diseased leaf). ![](RackMultipart20220611-1-s4b6n0_html_4232fe99c9171642.png)

![Shape10](RackMultipart20220611-1-s4b6n0_html_a38eea10a6cd1d2f.gif)

_Fig_ 4

**CycleGAN**

**inverse mapping** - Because the classic GAN mapping (G: X -\&gt; Y, presented in the Introduction) is highly under-constrained, CycleGAN couple it with an inverse mapping (F: Y -\&gt; X) (fig 5), and introduce a cycle consistency loss to enforce F(G(X)) ≈ X (and vice versa) (fig 5). ![](RackMultipart20220611-1-s4b6n0_html_582d2bb44710b6fc.png)

![Shape11](RackMultipart20220611-1-s4b6n0_html_a38eea10a6cd1d2f.gif)

_Fig_ 5

**Consistency loss** - the CycleGAN encourages cycle consistency by adding an additional loss to measure the difference between the generated output of the second generator and the original image, and the reverse. This acts as a regularization of the generator models, guiding the image generation process in the new domain toward image translation (fig 6). ![](RackMultipart20220611-1-s4b6n0_html_69091377fde23ad6.png)

![Shape12](RackMultipart20220611-1-s4b6n0_html_a38eea10a6cd1d2f.gif)

_Fig_ 6

![Shape13](RackMultipart20220611-1-s4b6n0_html_a38eea10a6cd1d2f.gif)

_Fig_ 7

**identity loss** - It is often beneficial to add the identity loss - the identity loss is simple, G(y) ≈ y and F(x) ≈ x. Adding an identity loss generally helps preserve color and tint in translated images, especially for photo to image generations (fig 7). ![](RackMultipart20220611-1-s4b6n0_html_1d1eade5f4e686e8.png)

In total, we have four losses, two adversarial losses, cycle loss and the identity loss, but these losses aren&#39;t equally important. We may want to give a greater importance to the cycle loss than the adversarial loss or vice versa. To solve this, the authors introduce λ. In the final loss equation, the term for cycle loss is multiplied by λ, as to give it more or less importance. The authors set λ equal to 10 for their experiments.

the authors define this as the full loss function:

![](RackMultipart20220611-1-s4b6n0_html_9f351b445edafca3.png)

**LFLSeg**

LFLSeg is simple but effective weakly supervised label-free leaf segmentation module that helps the classification model to learn the dense and interior leaf regions implicitly. From an architecture point of view, the backbone of LFLSeg is a simple CNN, and is designed to discriminate between &quot;full leaf&quot;, &quot;partial leaf&quot;, and &quot;non-leaf&quot; objects. Specifically, &quot;full leaf&quot; objects are images that contain a single full leaf, while &quot;partial leaf&quot; objects are images that contain part of a &quot;full leaf&quot;, and &quot;non-leaf&quot; objects do not contain any part of a leaf.

The segmented leaf region is obtained using a heatmap with respect to the &quot;full leaf&quot; class by applying the Grad-CAM technique. This heatmap is a probability map representing the contribution of each pixel to the final decision of the &quot;full leaf&quot; class, and thus can be used as a binary mask after thresholding with a specific threshold value δ.

The adversarial loss of the LeagGan with the LFLSeg masks is: ![](RackMultipart20220611-1-s4b6n0_html_720ca984519f21b8.png)

Note that ys = Sy \* y is the masked version of the image y ∈ Y, where Sy = LFLSeg(y) is the masking which represents the leaf area after feeding image y to the LFLSeg module.

**Method**

In this work, we train our LeafGAN models to generate new images of apple diseases. We used the plant pathology 2021 dataset resized to 900x900. We trained Resnet152 model to identify the apple disease classes. We noticed that this dataset is unbalanced, and our model is biased toward the majority classes, therefore we are going to synthesize disease images with the help of LeafGAN.

**Training LFLSeg module to obtain image masks**

For obtaining the image&#39;s masks we trained the LFLSeg module. We created the dataset for training the LFLSeg module, containing 3 classes:

- **Full leaf:** images that contain full apple leaf, this class is made of the plant pathology we downloaded from Kaggle - 18,632 images.
- **Partial leaf:** images that contain a part of a full leaf, we created this class by randomly choosing 1/9 of the full leaf class, we cropped each image to 9 patches - 18,632 images.
- **Non leaf:** images that do not contain any part of a leaf. We created this class by randomly downloading 100 classes and 150 images for each class from the ImageNet dataset - 15,000 images.

I ![](RackMultipart20220611-1-s4b6n0_html_77388f7d150fbdbf.png) ![](RackMultipart20220611-1-s4b6n0_html_75161fbaf7338bb.jpg) mages from the 3 classes with GradCam Heat Map:

![](RackMultipart20220611-1-s4b6n0_html_3af6ea20d501e76f.jpg)

![](RackMultipart20220611-1-s4b6n0_html_81a3f14df5a52f27.png) ![](RackMultipart20220611-1-s4b6n0_html_6e22c8cf412f1e4d.png) ![](RackMultipart20220611-1-s4b6n0_html_d5a980e42771a6f0.png)

![Shape16](RackMultipart20220611-1-s4b6n0_html_8eb642b8a87e1d56.gif) ![Shape15](RackMultipart20220611-1-s4b6n0_html_a49e6158eb89924a.gif) ![Shape14](RackMultipart20220611-1-s4b6n0_html_a49e6158eb89924a.gif)

_ **Non leaf** _

_ **Partial leaf** _

_ **Full leaf** _

We used the ResNet-101 model weights that the original authors of the LeafGAN offer in their GitHub repo, and fine-tuned it to our plant pathology dataset as the backbone of LFLSeg. We trained our model for 100 epochs with the hyperparameters, and augmentations recommended in the LeafGAN article.

After training the LFLSeg, we experimented with different thresholds, to choose the optimal threshold to obtain the best mask results (fig 8).

![](RackMultipart20220611-1-s4b6n0_html_8c93a7e3d49e211c.png)

![](RackMultipart20220611-1-s4b6n0_html_80b2d7be7a00a22c.jpg)

![Shape17](RackMultipart20220611-1-s4b6n0_html_a38eea10a6cd1d2f.gif)

_Fig_ 8 -

**Training LeafGAN models to synthesize images**

We chose 4 diseases to synthesize images to, based on class distribution.

In order to synthesize the diseased images we trained 4 LeafGANs (for each disease domain) models: healthy -\&gt; rust

healthy -\&gt; Complex

healthy -\&gt; Frog Eye Leaf Spot

healthy -\&gt; Powdery Mildew

For each model we created a dataset as follows:

TrainA – 80% Healthy leaf images (Domain A)

TrainA\_masked- Masks for all the images in TrainA

TestA – 20% Healthy leaf images (Domain A)

TrainB – 80% Disease leaf images (Domain B)

TrainB\_masked- Masks for all images in TrainB

TestB – 20% Disease leaf images (Domain B)

We used the LFLSeg module for generating masks for all training images.

Each model was trained for 100 epochs, with the hyperparameters, and augmentations recommended in the LeafGAN article. The training time for each model was approximately 60 hours.

After training we experimented with the model weights for each model, to find out in which epoch the model generates the best image. It&#39;s interesting to see that the model is probably converging around epoch 75 (fig 9).

![](RackMultipart20220611-1-s4b6n0_html_482b10e6e17881b9.jpg)

![Shape18](RackMultipart20220611-1-s4b6n0_html_a38eea10a6cd1d2f.gif)

_Fig_ 9

During training, we examined the images in loss manner (real, fake, reconstructed and identity) every epoch to monitor the model learning process:

**Real A Real B**![](RackMultipart20220611-1-s4b6n0_html_5621629802640558.png) ![](RackMultipart20220611-1-s4b6n0_html_8d06f32ff259d967.png)

**Fake A Fake B**![](RackMultipart20220611-1-s4b6n0_html_bf5d52b153ec17d7.png) ![](RackMultipart20220611-1-s4b6n0_html_e56b0a4455391909.png)

**Idt A Idt B**![](RackMultipart20220611-1-s4b6n0_html_f5a9c4ab21f3bfd.png) ![](RackMultipart20220611-1-s4b6n0_html_4d8d302f1513772d.png)

**Rec A Rec B**![](RackMultipart20220611-1-s4b6n0_html_ab01ee0af61a5098.png) ![](RackMultipart20220611-1-s4b6n0_html_b2cbadd32b3fdc99.png)

**Synthesize Images**

To balance our dataset, we examined each class distribution. Based on that distribution we decided how many images to generate for each class.

**Translation methods:**

we experimented with 3 image translation methods with the LeafGAN:

- The trivial translation, the one the models was trained to: Healthy -\&gt; Disease
- Feed forward Healthy image through 2 different generators:

Healthy -\&gt; Disease1 -\&gt; Disease2. Final product: Disease1+Disease2.

- Feed forward Diseased image through another disease generator:

Disease1 -\&gt; Disease2. Final product: Disease1+Disease2.

The two last translations are particularly beneficial for our dataset because it is multi-labeled, although we didn&#39;t train the LeafGAN models for those translations.

**Adding synthetic images to training set:**

We split our original dataset to train and validation (80/20) and then we generated images to be added to the training set as follows:

- Healthy -\&gt; Powdery mildew - 166 images (new label: Powdery mildew)
- Healthy -\&gt; Complex - 198 images (new label: Complex)
- Healthy -\&gt; Rust - 140 images (new label: Rust)
- Healthy -\&gt; Frog Eye Leaf Spot - 196 images (new label: Frog Eye Leaf Spot)
- Healthy -\&gt; Rust -\&gt; Frog Eye Leaf Spot - 100 images (new label: Rust + Frog Eye Leaf Spot)
- Scab -\&gt; Frog Eye Leaf Spot - 112 images (new label: Scab + Frog Eye Leaf Spot)

To avoid data leaking and reduce the over-represented classes (Healthy and Scab) we removed the original images that used to generate the synthesized images.

Training set labels distribution before (fig 10) and after (fig 11) adding the synthetic images.

![](RackMultipart20220611-1-s4b6n0_html_6ea5ae3dbed59765.png)

![](RackMultipart20220611-1-s4b6n0_html_f1f31daac4b50548.png)

![Shape20](RackMultipart20220611-1-s4b6n0_html_bae58f493f91f913.gif) ![Shape19](RackMultipart20220611-1-s4b6n0_html_d25e4d07e93843ca.gif)

_Fig_ 11

_Fig_ 10

**Results**

| **translation** | **Real Target domain image** | **Good Fake generated image** | **Bad Fake generated image** |
| --- | --- | --- | --- |
| Healthy -\&gt; rust | ![](RackMultipart20220611-1-s4b6n0_html_bf87f189aebe5621.jpg)
 | ![](RackMultipart20220611-1-s4b6n0_html_24025ee9aceec179.png)
 | ![](RackMultipart20220611-1-s4b6n0_html_db64fa7bc8691279.png)
 |
| Healthy -\&gt; Frog Eye Leaf Spot | ![](RackMultipart20220611-1-s4b6n0_html_9a9e9dac584cbc20.jpg)
 | ![](RackMultipart20220611-1-s4b6n0_html_8aab6f14775de01a.png)
 | ![](RackMultipart20220611-1-s4b6n0_html_ff2552e53d4f1a22.png)
 |
| Healthy -\&gt; Powdery Mildew | ![](RackMultipart20220611-1-s4b6n0_html_73b1914bc287d262.jpg)
 | ![](RackMultipart20220611-1-s4b6n0_html_c3e0a4c0bdb1bb9.png)
 | ![](RackMultipart20220611-1-s4b6n0_html_1206920e5911881e.png)
 |
| Healthy -\&gt; Complex | ![](RackMultipart20220611-1-s4b6n0_html_7e94a1969157a1a6.jpg)
 | ![](RackMultipart20220611-1-s4b6n0_html_1cffbf92f29a87b5.png)
 | ![](RackMultipart20220611-1-s4b6n0_html_f5d4051d12ca0592.png)
 |
| Healthy -\&gt; Rust -\&gt; Frog Eye Leaf Spot | ![](RackMultipart20220611-1-s4b6n0_html_6559e71cdd81c7de.jpg)
 | ![](RackMultipart20220611-1-s4b6n0_html_62ead8364339be0f.png)
 | ![](RackMultipart20220611-1-s4b6n0_html_5c5395881d03c703.png)
 |
| Scab -\&gt; Frog Eye Leaf Spot | ![](RackMultipart20220611-1-s4b6n0_html_cb97ec6e72153ccc.jpg)
 | ![](RackMultipart20220611-1-s4b6n0_html_eec8834a77571e06.png)
 | ![](RackMultipart20220611-1-s4b6n0_html_7df0b1550b90806f.png)
 |

**Examination of the generated images -**

We carefully examined all generated images and compared them to the real images in the dataset, to further understand the LeafGAN generating capabilities.

- Basic translations the GAN was trained on - when translating healthy images to rust, frog eye leaf spot and complex we got really good and realistic results, although in some cases the LeafGAN was messing with the background. It was more challenging for the LeafGAN to translate to powdery mildew, in a lot of the cases it seems to miss the leaf and implant the disease on the background, therefore the generated images look fake.
- Feed forward healthy image through 2 generators - although we didn&#39;t train the LeafGAN to translate images in that way, surprisingly we got nice results when translating healthy images to rust and then to frog eye leaf spot, ending up with multi diseased image (rust + frog eye leaf spot). The generated images look real when comparing it to the original images from the dataset.
- Feed forward disease image through another disease generator - like the translation mention above we didn&#39;t train the LeafGAN to translate images in that way, again we got good results when translating scab disease images to frog eye leaf spot, ending up with multi diseased image (scab + frog eye leaf spot). The synthetic images look very realistic compared to the original images from the dataset.

Experimenting with various translations, shows the LeafGAN high performance in image translation even on translation tasks it did not train on, it reveals the big potential of GANs in general.

**LeafGAN graph examination**

We integrated the original LeafGAN code to the wandb platform for graph visualization, to examine and monitor all the losses through training.

![](RackMultipart20220611-1-s4b6n0_html_6e1c860d9b14dca8.png)Cycle loss - the LeafGAN cycle loss was reduced during training for both domains (A and B) in all models as expected (fig 12).

![Shape21](RackMultipart20220611-1-s4b6n0_html_bae58f493f91f913.gif)

_Fig_ 12

![](RackMultipart20220611-1-s4b6n0_html_8aa2688c6edbcf96.png)Generator loss vs. discriminator loss - it&#39;s interesting to see that we got some negative correlation between the generator and discriminator losses. This is also expected as the generator and discriminator are competing with each other (fig 13).

![Shape22](RackMultipart20220611-1-s4b6n0_html_bae58f493f91f913.gif)

_Fig_ 13

![](RackMultipart20220611-1-s4b6n0_html_6a0984ac823095a8.png)Identity loss - the LeafGAN identity loss was reduced during training for both domains (A and B) in all models as expected.

![Shape23](RackMultipart20220611-1-s4b6n0_html_bae58f493f91f913.gif)

_Fig_ 14

To evaluate the LeafGAN performance, we added generated images to the plant pathology dataset to balance the data using the method mentioned in the proposed solution section.

We trained a resnet152 model to predict 6 labels (complex, frog\_eye\_leaf\_spot, powdery mildew, rust, scab, healthy), first on the original dataset without the synthetic images, after that we added the synthetic images and train the model on the new dataset.

Then we compared the performance (f1-score) of the two models to evaluate the LeafGAN beneficial effect on the imbalance dataset (fig 15).

**Validation f1**![](RackMultipart20220611-1-s4b6n0_html_c768d32fc4cb1391.png)

![Shape24](RackMultipart20220611-1-s4b6n0_html_bae58f493f91f913.gif)

_Fig_ 15

**Test Classification Results**

| Model | Label | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
|


Resnet152 Original dataset | Complex | 0.73 | **0.74** | **0.74** |
| Frog Eye Leaf Spot | 0.83 | 0.86 | 0.85 |
| Powdery Mildew | 0.79 | 1.0 | 0.88 |
| Rust | 0.87 | 0.98 | 0.92 |
| Scab | 0.89 | 0.87 | 0.88 |
| Healthy | **0.98** | 0.93 | 0.96 |
| Average | 0.87 | 0.89 | 0.88 |
|


Resnet152 generated dataset | Complex | **0.76** | 0.71 | 0.73 |
| Frog Eye Leaf Spot | **0.86** | **0.89** | **0.87** |
| Powdery Mildew | **0.87** | **0.99** | **0.92** |
| Rust | **0.90** | 0.98 | **0.94** |
| Scab | **0.93** | **0.89** | **0.91** |
| Healthy | 0.97 | **0.97** | **0.97** |
| Average | **0.90** | **0.90** | **0.90** |

**\*Bold = higher score between the two models**

By comparing the two models results, we can see that the model that was trained on the generated images dataset is outperforming the model that was trained on the original dataset, although the performances are quite similar.

**Summary and Conclusion**

The LeafGAN is a powerful image translation model that can be used for various tasks. It is able to produce realistic disease images that can be used for data augmentations in agriculture computer vision tasks.

Our main goal was to overcome the imbalance dataset problem. With the help of the LeafGAN, we did improve our classifier performances on identifying diseased apple leaves but in small percentage (~2%).

It looks like with further work we can improve the performances even more, maybe by changing the heuristic of adding the synthetic images and the amount of the synthetic images being added to the dataset.

Due to hardware limitations, we trained our LeafGAN models for 100 epochs, unlike the recommended 200 epochs by the paper. Maybe training for more epochs can lead us to better results.

Further work can be done to improve images masks that are fed to the LeafGAN network, to separate more accurately the leaves from the background and improve LeafGAN attention mechanism.

In conclusion, it was a very interesting experience working with this advanced CycleGan architecture. During our work we faced some challenges due to the fact that training a GAN is very different from other machine learning projects we worked on before. Eventually, we&#39;re satisfied with the results and learned a lot during the making of this project.

**References**

[1] Cap, Q. H., Uga, H., Kagiwada, S., &amp; Iyatomi, H. (2020). Leafgan: An effective data augmentation method for practical plant disease diagnosis. _IEEE Transactions on Automation Science and Engineering_.‏

[2] Zhu, J. Y., Park, T., Isola, P., &amp; Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In _Proceedings of the IEEE international conference on computer vision_ (pp.2223-2232).

[3] LeafGAN github repository: [https://github.com/IyatomiLab/LeafGAN](https://github.com/IyatomiLab/LeafGAN)
