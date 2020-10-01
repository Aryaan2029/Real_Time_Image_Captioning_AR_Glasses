# Automated Image Captioning 

Our Team - Alison Ma, Aryan Pariani, Aryaan Mehra, Max Rivera, Jun Chen

## Background Information:

In our project, we’re looking to not only understand the distinct features of multiple objects, rather the collective relationships of these objects to derive useful meanings from images. To do so, we will be leveraging two major fields of artificial intelligence: Computer Vision and Natural Language Processing. Our model consists of two sub-models: an object detection and localization model, which will extract the information of objects and their spatial relationship in images respectively using bounding boxes; and a deep recurrent neural network (RNN) based on long short-term memory (LSTM) units with attention mechanism for sentence generation. 

We believe such models have several applications to assist the visually impaired in understanding the environment around them. Over 12 million people suffer from visual impairments in the US alone [cdc.gov]. This does not account for millions of additional people who cannot process what they see due to impaired functions in other brain regions. Current tools used to improve quality of life include guide dogs, corrective lenses (look at others). Increased accessibility and development of mobile devices and smart glasses provides an opportunity to employ more sophisticated ways of helping not only the visually impaired, but also those with impaired/damaged processing centers to navigate their surroundings. 

## Problem Definition:

The number of people who require visual aid will only continue to rise and if unaddressed, will further exacerbate personal, societal, and economic consequences. To combat this universal problem, the need for intelligent image captioning research and automatic tools are necessary not only to provide newer qualities of aid such as interpretation, but also for these tools to become more readily available and affordable. With computer vision techniques and natural language processing, an image captioning process can be done to assist the visually impaired so that they may further understand and interact with their surrounding environment. Given an image, our project aims to identify and categorize objects within images with a CNN model for object detection and utilize RNN with Attention Mechanism and LSTM to caption the image. By doing this, we not only address the hindrances that visual impairment constructs, but we also gain an understanding of how the pipeline of objection detection and object captioning works. Our study will limit itself to the scope of the English language. 

## Methodology:

1. We will build and train a custom CNN model or fine-tune a pretrained CNN model with transfer learning for Object Detection (most likely a Fast R-CNN Model) to first identify, categorize and encapsulate with bounding boxes the classes of common objects found in the images. 

2. Using the object classes and the position of their bounding boxes in each image, we will train a Recurrent Neural Network with an Attention Mechanism and LSTM units to generate sequences or sentences of captions for activities being performed or events and objects depicted in the images. 

3. The MSCOCO Dataset has a large amount of image data with bounding box position, height and width labels as well as relevant captions, and is therefore very commonly used for image caption generation purposes. We will also be making use of this extensive dataset for image caption generation.

4. As we work on our project, we will also try to incorporate visualizations of convolutional filters and features being detected with extensive evaluation of our results. 

## Potential Results:

1. Our Object Detection CNN model is successful, for the most part, in identifying different classes of objects in images, and precisely encapsulating these objects in bounding boxes. 

2. Our LSTM Recurrent Neural Network can generate simple and coherent sequences or sentence captions that make basic sense of events depicted in images, taking into special account the names and positions of objects detected and bounding boxes fit by the Object Detection CNN model. 

3. A related research has been done with VGG16 a deep learning architecture for image classification for extraction of visual attributes which was fed into Stanford CoreNLP Model for generating the captions  [3] . This related work can be used as comparison for our performance metrics.

4. Potential quantitative evaluation of our approach include CIDEr and BLEU. From the MSCOCO dataset, there are referenced captions which can be compared with our generated captions with CIDEr which measures cosine similarity or BLEU which measures sensitivity [3].

## Discussion:

1. We will use evaluation metrics to measure the objective, measurable performance of our models beyond just accuracy, such as F-1 Score, area under curve, specificity and sensitivity for both the training and test dataset. We shall discuss how to tackle model-related problems identified by evaluating these metrics, especially with respect to regularization, overfitting, the vanishing gradient problem, and suitable gradient descent optimizers, to name a few. 

2. We will evaluate the performance of the model by analyzing generated visualizations, filters of feature maps, and convolutional filters to assess the performance of filters in our Convolutional Neural Network and the LSTM units in our Recurrent Neural Network to identify problems and discuss possible solutions in terms of tuning of hyperparameters. 

## References:

[1]  	Zhongliang Yang,  Yu-Jin Zhang,  Sadaqat ur Rehman,  and Yongfeng
Huang.  2017.  Image captioning with object detection and localization. 
arXiv:abs/1706.02430, Retrieved from https://arxiv.org/abs/1706.02430

[2] 	CDC. Center for Disease Control and Prevention. Retrieved from	     
https://www.cdc.gov/visionhealth/basics/ced/fastfacts.htm#:~:text=Approximately
%2012%20million%20people%2040,due%20to%20uncorrected%20refractive%20error

[3]	B. Makav and V. Kılı̧c. 2019.  A new image captioning approach for visually 
im-paired people. In 2019 11th International Conference on Electrical and Elec-tronics
Engineering (ELECO), November 28-30, 2019, Bursa, Turkey. IEEE, Piscataway, NJ, 945–949.
https://doi.org/10.23919/ELECO47770.2019.8990630.


## Dataset:

[Description]
