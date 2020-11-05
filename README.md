# Automated Image Captioning 

Our Team - Alison Ma, Aryan Pariani, Aryaan Mehra, Max Rivera

## Introduction/Background:

Our project aims to identify objects with images and derive useful meanings from them by combining two major fields: computer vision and natural language processing. According to the CDC, 12 million people already suffer from visual impairments in the US alone, a number that is only expected to double by 2050 [2]. Current methods, such as guide dogs and corrective lenses, may improve quality of life, but increased accessibility of mobile devices and smart glasses provide an opportunity to employ more sophisticated methods, such as machine learning, to help both the visually impaired and those with impaired/damaged processing centers.

## Problem Definition:

Given an image, our project aims to identify and categorize objects with a CNN model for object detection and utilize RNN with Attention Mechanism and LSTM to produce a caption. By doing this, we not only address the hindrances that visual impairment constructs, but we also gain an understanding of implementing a pipeline for objection detection and captioning. Our study will limit itself to the scope of the English language.


## Methodology:

1. We will build We will be using a version of the Fast-RCNN to create bounding boxes on our dataset as we believe it will provide us with more accurate captions later when we create our RNN model. 

2. We chose Mask-RCNN both because of the speed and the accuracy of the model. Mask-RCNN is backed by a Resnet 101 architecture. Normally, when we train a very deep network, it tends to perform worse on the training set as the layers get deeper. The presence of skip connections in a ResNet with identity blocks, enables the activations of subsequent layers to equal prior layers without having the additional layers hurt the performance of the model. 

3. This turns out to improve the performance of the model as well at times as the presence of additional hidden units in a layer might improve feature selection. 

4. An example of the ResNet structure is the following:

![](Images/Screen%20Shot%202020-10-31%20at%2010.51.32%20PM.png)

5. Using the object classes and the position of their bounding boxes in each image, we will train a Recurrent Neural Network with an Attention Mechanism and LSTM units to generate sequences or sentences of captions for activities being performed or events and objects depicted in the images. 

6. The MSCOCO Dataset has a large amount of image data with bounding box position, height and width labels as well as relevant captions, and is therefore very commonly used for image caption generation purposes. We will also be making use of this extensive dataset for image caption generation.

7. As we work on our project, we will also try to incorporate visualizations of convolutional filters and features being detected with extensive evaluation of our results. 

## Data Collection:

1. We first tried to use the MSCOCO dataset which is a consolidated dataset containing 328,000 images with over 2.5 million labelled instances across 91 common object classes. 

2. When trying to work with a dataset this massive, we ran into several issues with downloading, storing and applying a CNN to it. 

3. As a result, we decided it would be best to use a smaller dataset which also had thousands of images and labelled instances but would be easier to work with. 

4. After researching the various different datasets out there we decided to use the Flickr30k dataset from Kaggle which, apparently over time, has become the standard dataset to use for image captioning purposes.

5. It contains 31,873 images with 276,000 labels on these images and 27 overall classes. As shown below.

![](Images/Screen%20Shot%202020-10-31%20at%206.58.07%20AM.png)

6. This turned out to work well for us and we moved ahead with the process of applying Fast-RCNN on the dataset to generate bounding boxes for the images since our thesis is that the presence of bounding boxes will improve accuracy in the final product. 
7. In order to gauge the efficacy of the dataset we started by applying a simple pre-trained YOLO model on the dataset to see if it worked and to see the results. 

![](Images/Screen%20Shot%202020-10-31%20at%206.32.53%20AM.png)

8. This seemed to work well and we had no issues generating bounding boxes, hence, we moved into applying Fast-RCNN. The above image shows the YOLO object detection capability.
9. On researching Fast-RCNN, we realised that several versions of it have now been released including Fast-RCNN, Faster-RCNN and Mask-RCNN, the most recent and accurate version of the model. 
10. We realised that making such a model from scratch would be extremely difficult and time consuming so we decided to use transfer learning. 
11. We were able to find a model which had trained Mask-RCNN on the MS COCO dataset and download those pre-trained weights. 
12. Next, we applied those weights to train all 31,873 images in the Flickr dataset to generate bounding boxes for all the objects in each image.
13. The following are some examples:

![](Images/Screen%20Shot%202020-10-31%20at%206.58.34%20AM.png)

14. This is a relatively high accuracy image where even items such as bags, handbags and traffic lights are being detected. 

![](Images/Screen%20Shot%202020-10-31%20at%206.58.49%20AM.png)

15. This image is a relatively low accuracy prediction with items like car and clock being recognized incorrectly. 

16. After obtaining bounding box information on all 31,873 images, our data collection process ended. Next we will be focussing on our RNN model for captioning.

## Results:

1. By using the Mask-RCNN model described above, we were able to generate bounding box information for all 31,873 images in the Flickr dataset. 

2. We were also able to store this information as a dictionary using json and store all the information inside text files. 

3. Each image has the three keys in a dictionary; 

4. “Class ID’s” which signifies which class the object in the image belongs to, “ROI’s” specify the location in the image as a list as [center y coordinate, center x coordinate, height, width] - [y1,x1, y2, x2] and “Scores” which contains information about the accuracy of each class identified in a given image. 

![](Images/Screen%20Shot%202020-10-31%20at%207.04.02%20AM.png)

5. We were successful in the milestone which we had set for ourselves which was to finish all the data analysis and collection.

6. We can now move into making our RNN image captioning model to generate captions for our images.


## Discussion:

1. From the results of the object detection performed by the Mask-RCNN model it is clear that many images have extremely accurate detection of objects. This will be extremely helpful when generating accurate captions for our dataset. 

2. Similarly, there are several images which were relatively less accurate and described some classes correctly and others incorrectly in an image. This may cause our captioning system to have some images being highly accurate and some images being very inaccurate. It will be interesting to see the discrepancy between accurate and inaccurate images and understand the extent to which inaccuracy in the object detection system will play a role in providing inaccurate captions in the captioning system.

3. We will use evaluation metrics to measure the performance of our models beyond just accuracy, such as F-1 Score, area under curve, specificity and sensitivity for both the training and test dataset. We shall discuss how to tackle model-related problems identified by evaluating these metrics, especially with respect to regularization, overfitting, the vanishing gradient problem, and suitable gradient descent optimizers, to name a few. 

4. We will evaluate the performance of the model by analyzing generated visualizations, filters of feature maps, and convolutional filters to assess the performance of filters in our Convolutional Neural Network and the LSTM units in our Recurrent Neural Network to identify problems and discuss possible solutions in terms of tuning of hyperparameters. 

5. Potential quantitative evaluation of our approach include CIDEr and BLEU. From the MSCOCO dataset, there are reference captions which can be compared with our generated captions with CIDEr which measures cosine similarity or BLEU which measures sensitivity [3].


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

