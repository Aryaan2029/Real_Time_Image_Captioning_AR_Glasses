# Automated Image Captioning 

Our Team - Alison Ma, Aryan Pariani, Aryaan Mehra, Max Rivera

## Introduction/Background:

Our project aims to identify objects with images and derive useful meanings from them by combining two major fields: computer vision and natural language processing. According to the CDC, 12 million people already suffer from visual impairments in the US alone, a number that is only expected to double by 2050 [2]. Current methods, such as guide dogs and corrective lenses, may improve quality of life, but increased accessibility of mobile devices and smart glasses provide an opportunity to employ more sophisticated methods, such as machine learning, to help both the visually impaired and those with impaired/damaged processing centers.

## Problem Definition:

Given an image, our project aims to identify and categorize objects with a CNN model for object detection and utilize RNN with Attention Mechanism and LSTM to produce a caption. By doing this, we not only address the hindrances that visual impairment constructs, but we also gain an understanding of implementing a pipeline for objection detection and captioning. Our study will limit itself to the scope of the English language.


## Methodology:

* We have used a version of the Fast-RCNN to create bounding boxes on our dataset as we believe it will provide us with more accurate captions later when we create our RNN model. 

* We chose Mask-RCNN both because of the speed and the accuracy of the model. Mask-RCNN is backed by a Resnet 101 architecture. Normally, when we train a very deep network, it tends to perform worse on the training set as the layers get deeper. The presence of skip connections in a ResNet with identity blocks, enables the activations of subsequent layers to equal prior layers without having the additional layers hurt the performance of the model. Skip connections also help to tackle the problem of vanishing gradients in the model, feeding the output of one layer as the input to the next layers, instead of only the next one. 

* This turns out to improve the performance of the model as well at times as the presence of additional hidden units in a layer might improve feature selection. 

* An example of the ResNet structure is the following:

![](Images/Screen%20Shot%202020-10-31%20at%2010.51.32%20PM.png) 

and an excerpt from a plot of the generated graph of the network neural network architecture in Mask-RCNN, showing the skip connection from the "res4g_out" activation layer to the "add_147" Add layer, shown below: 

![](Images/Screenshot%202020-11-06%20at%201.42.50%20AM.png) 


* Using the object classes and the position of their bounding boxes in each image, we trained a Recurrent Neural Network with an Attention Mechanism and LSTM units to generate sequences or sentences of captions for activities being performed or events and objects depicted in the images. 

* The Flickr Dataset has a large amount of image data with bounding box position, height and width labels as well as relevant captions, and is therefore very commonly used for image caption generation purposes. We will also be making use of this extensive dataset for image caption generation.

* As we work on our project, we will also try to incorporate visualizations of convolutional filters and features being detected with extensive evaluation of our results. 

## Data Collection:

* We first tried to use the MSCOCO dataset which is a consolidated dataset containing 328,000 images with over 2.5 million labelled instances across 91 common object classes. 

* When trying to work with a dataset this massive, we ran into several issues with downloading, storing and applying a CNN to it. 

* As a result, we decided it would be best to use a smaller dataset which also had thousands of images and labelled instances but would be easier to work with. 

* After researching the various different datasets out there we decided to use the Flickr30k dataset from Kaggle which, apparently over time, has become the standard dataset to use for image captioning purposes.

* It contains 31,873 images with 276,000 labels on these images and Mask-RCNN with 81 overall classes. As shown below.

![](Images/Screen%20Shot%202020-10-31%20at%206.58.07%20AM.png)

* This turned out to work well for us and we moved ahead with the process of applying Fast-RCNN on the dataset to generate bounding boxes for the images since our thesis is that the presence of bounding boxes will improve accuracy in the final product. 
* In order to gauge the efficacy of the dataset we started by applying a simple pre-trained YOLO model on the dataset to see if it worked and to see the results. ![](Images/Screen%20Shot%202020-10-31%20at%206.32.53%20AM.png)

* This seemed to work well and we had no issues generating bounding boxes, hence, we moved into applying Fast-RCNN. The above image shows the YOLO object detection capability.
* On researching Fast-RCNN, we realised that several versions of it have now been released including Fast-RCNN, Faster-RCNN and Mask-RCNN, the most recent and accurate version of the model. 
* We realised that making such a model from scratch would be extremely difficult and time consuming so we decided to use transfer learning. 
* We were able to find a model which had trained Mask-RCNN on the MS COCO dataset and download those pre-trained weights. 
* Next, we applied those weights to train all 31,873 images in the Flickr dataset to generate bounding boxes for all the objects in each image.
* The following are some examples:![](Images/Screen%20Shot%202020-10-31%20at%206.58.34%20AM.png)

* This is a relatively high accuracy image where even items such as bags, handbags and traffic lights are being detected. 

![](Images/Screen%20Shot%202020-10-31%20at%206.58.49%20AM.png)

* This image is a relatively low accuracy prediction with items like car and clock being recognized incorrectly. 

* After obtaining bounding box information on all 31,873 images, our data collection process ended. Next we will be focussing on our RNN model for captioning.![](Screen%20Shot%202020-12-03%20at%2010.40.05%20PM.png)

* Another important aspect that we explored in the 30K images were the frequency of the different words we were tokenizing. As you can see after cleaning the tokens for punctuation, white spaces, etc, certain words such as ‘a’, ‘in’ and ‘dog’ occur very frequently whereas more complicated words occur only once which makes it difficult for the model to learn those and map them to specific images. 

## RNN Model Architecture:

We have demonstrated the use of object spatial relationship modeling for image captioning, specifically within the Transformer encoder-decoder architecture. This is achieved by incorporating the object relation module within the Transformer encoder.

* We introduce the Object Relation Transformer, an encoder-decoder architecture designed specifically for image captioning, that incorporates information about the spatial relationships between input detected objects through geometric attention. 
* We quantitatively demonstrate the usefulness of geometric attention through both baseline comparison and an ablation study on the Flickr dataset. 
* Lastly, we qualitatively show that geometric attention can result in improved captions that demonstrate enhanced spatial awareness.

![](Images%20captions/model.png)

Below, we’ve described our own model based on an Object Relation Transformer. We first utilize the Mask-RCNN image detector in order to generate the bounding boxes and encode object features for our images. We do this by using an Embedding Layer in our RNN. Next, we use an LSTM Layer in order to better convey spatial relationships between different objects. We then add our image features and caption features into a 1-D array in order to generate valid outputs to match images to captions. The use of a Dropout layer is to prevent overfitting.

![](Images%20captions/Screen%20Shot%202020-11-26%20at%209.44.13%20PM.png)

The recurrent neural network uses the long short-term memory blocks to take a particular word or phoneme, and evaluate it in the context of others in a string, where memory can be useful in sorting and categorizing these types of inputs. In general, LSTM is an accepted and common concept in pioneering recurrent neural networks.

![](Images%20captions/LSTM.png)

## Results:

* By using the Mask-RCNN model described above, we were able to generate bounding box information for all 31,873 images in the Flickr dataset. 

* We were also able to store this information as a dictionary using json and store all the information inside text files. 

* Each image has the three keys in a dictionary; 

* “Class ID’s” which signifies which class the object in the image belongs to, “ROI’s” specify the location in the image as a list as [center y coordinate, center x coordinate, height, width] - [y1,x1, y2, x2] and “Scores” which contains information about the accuracy of each class identified in a given image. 

![](Images/Screen%20Shot%202020-10-31%20at%207.04.02%20AM.png)

* We were successful in the milestone which we had set for ourselves which was to finish all the data analysis and collection.

* We can now move into making our RNN image captioning model to generate captions for our images.

* As it can be observed below, the Mask-RCNN object detection model generated bounding boxes with crucial spatial information about the position of a bride and the relative position of the umbrella over her head. This is one very good example we were able to spot where the spatial relationship information provided by bounding box coordinates and dimensions was able to help the attention-based Recurrent Neural Network Model generate precise captions that convey these spatial relationships, here the case being that the generated caption points out that the bride is being escorted “under an umbrella”. This is one aspect where we hoped our model would outperform image captioning models that are commonly implemented without using an external object detection model or bounding boxes to help captioning, in terms of spatial relationships between objects in images, that would help the visually challenged understand their environment much better and more precisely. 

![](Images%20captions/Screenshot%202020-11-27%20at%2012.37.07%20AM.png)
![](Images%20captions/Screenshot%202020-11-27%20at%2012.37.40%20AM.png)

* Some more examples from our test set of successful encoding of bounding box spatial information to precise captions that convey the same information, are shown below:

![](Images%20captions/Screenshot%202020-11-27%20at%2012.24.36%20AM.png)
![](Images%20captions/Screenshot%202020-11-27%20at%2012.23.36%20AM.png)
![](Images%20captions/Screenshot%202020-11-27%20at%2012.20.58%20AM.png)
![](Images%20captions/Screenshot%202020-11-27%20at%2012.17.15%20AM.png)

* As we can see below, after training our model for 200 epochs in total, the overall accuracy for our training set was 90.57% whereas our f1 metric was also 0.91. It is important to note though, that these results were on our training set but our test accuracy plateaued at ~30%. 

![](Images%20captions/Screenshot%202020-11-30%20at%2010.16.34%20PM.png)

## Discussion:

* From the results of the object detection performed by the Mask-RCNN model it is clear that many images have extremely accurate detection of objects. This will be extremely helpful when generating accurate captions for our dataset. 

* Similarly, there are several images which were relatively less accurate and described some classes correctly and others incorrectly in an image. This may cause our captioning system to have some images being highly accurate and some images being very inaccurate. It will be interesting to see the discrepancy between accurate and inaccurate images and understand the extent to which inaccuracy in the object detection system will play a role in providing inaccurate captions in the captioning system.

* On looking at our images we can see that the bounding boxes have helped some images obtain very accurate results in their classification of spatial relationships. Although the total test accuracy did not exceed 30%, the few images that did have accurate bounding box representations, as shown above, had very accurate captions with spatial relationships.

* Below we’ve also attached the BLEU score distributions of some of our best and worst image captions. The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric for evaluating a generated sentence to a reference sentence. A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0.

# Worst Captions:
![](Images%20captions/Screen%20Shot%202020-11-26%20at%209.44.50%20PM.png)

# Best Captions:
![](Images%20captions/Screen%20Shot%202020-11-26%20at%209.45.00%20PM.png)

* Another important observation is that even though the loss on the training set kept decreasing as the model trained all the existing images, it was unable to improve the testing accuracy beyond the 6th-7th epoch. This may be because the dataset was too small and we need more pictures to have accurate results. We tried to solve this overfitting problem by adding Dropout layers but it did not have a significant effect on reducing bias.

![](Images%20captions/Screen%20Shot%202020-11-26%20at%209.44.31%20PM.png)

## Conclusion: 

* We confidently believe that we have achieved our main objective - taking on the popular image captioning problem with a slightly more unique approach that involves feeding spatial information conveyed by bounding boxes generated by an object detection model, to the attention-based RNN for generating captions. But, there are some major improvements that we can make in the future.

* How can we improve upon tackling overfitting in the future? Although we made optimizations to improve our model’s generalization ability, like dropout layers, etc, we believe the main problem is not enough pictures in the dataset. Even with 30,000 images in our dataset, we need a much larger dataset, so that our image captioning model can generalize better for images of most common daily events in life. Clearly, to do well on a dataset of common daily events in life, we need a lot more images that contain frequently occurring objects in them, depicting millions of different events. We can also tackle overfitting by adding lasso (L1) and ridge (L2) regression parameters to the trainable layers in our attention-based Recurrent Neural Network model. 

* How can we improve upon tackling the vanishing gradient problem for learning longer and more complex captions? We can try to add more LSTM units in our Recurrent Neural Network model that can help to diminish the vanishing gradient problem even more and help to learn more complex and longer sentences/captions. 

## References:

[1]  	Zhongliang Yang,  Yu-Jin Zhang,  Sadaqat ur Rehman,  and Yongfeng
Huang.  2017.  Image captioning with object detection and localization. 
arXiv:abs/1706.02430, Retrieved from https://arxiv.org/abs/1706.02430

[2] 	CDC. Center for Disease Control and Prevention. Retrieved from	     
https://www.cdc.gov/visionhealth/basics/ced/fastfacts.htm#:~:text=Approximately
%2012%20million%20people%2040,due%20to%20uncorrected%20refractive%20error

[3]	B. Makav and V. Kılı̧c. 2019.  A new image captioning approach for visually im-paired people. 
In 2019 11th International Conference on Electrical and Elec-tronics Engineering (ELECO), 
November 28-30, 2019, Bursa, Turkey. IEEE, Piscataway, NJ, 945–949. 
https://doi.org/10.23919/ELECO47770.2019.8990630.

[4]	Matterport. “Matterport/Mask_RCNN.” GitHub, https://github.com/matterport/Mask_RCNN

[5]	Hsankesara. “Flickr Image Dataset.” Kaggle, 12 June 2018, 
www.kaggle.com/hsankesara/flickr-image-dataset

[6]	Simao Herdade, Armin Kappeler, Kofi Boakye, Joao Soares. Image Captioning: Transforming Objects into Words. https://papers.nips.cc/paper/2019/file/680390c55bbd9ce416d1d69a9ab4760d-Paper.pdf
