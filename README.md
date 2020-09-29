# Image Captioning 

Our Team - Alison Ma, Aryan Pariani, Aryaan Mehra, Max Rivera, Jun Chen

## Background Information:

In our project, we’re looking to not only understand the distinct features of multiple objects, rather the collective relationships of these objects to derive useful meanings from images. To do so, we will be leveraging two major fields of artificial intelligence: Computer Vision and Natural Language Processing. Our model consists of two sub-models: an object detection and localization model, which will extract the information of objects and their spatial relationship in images respectively using bounding boxes; and a deep recurrent neural network (RNN) based on long short-term memory (LSTM) units with attention mechanism for sentence generation. 

We believe such models have several applications to assist the visually impaired in understanding the environment around them. Over 12 million people suffer from visual impairments in the US alone [cdc.gov]. This does not account for millions of additional people who cannot process what they see due to impaired functions in other brain regions. Current tools used to improve quality of life include guide dogs, corrective lenses (look at others). Increased accessibility and development of mobile devices and smart glasses provides an opportunity to employ more sophisticated ways of helping not only the visually impaired, but also those with impaired/damaged processing centers to navigate their surroundings. 


## Methodology:

We will build and train a custom CNN model or fine-tune a pretrained CNN model with transfer learning for Object Detection (most likely a Fast R-CNN Model) to first identify, categorize and encapsulate with bounding boxes the classes of common objects found in the images. 
Using the object classes and the position of their bounding boxes in each image, we will train a Recurrent Neural Network with Attention Mechanism and LSTM units to generate sequences or sentences of captions for activities being performed or events and objects depicted in the images. 
The MSCOCO Dataset has a large amount of image data with bounding box position, height and width labels as well as relevant captions, and is therefore very commonly used for image caption generation purposes. We will also be making use of this extensive dataset for image caption generation.
As we work on our project, we will also try to incorporate visualizations of convolutional filters, etc. 

## Dataset:

[Description]

## References:

Image Captioning with Object Detection and Localization: https://arxiv.org/abs/1706.02430

Statistics about visual impairment: https://www.cdc.gov/visionhealth/basics/ced/fastfacts.htm#:~:text=Approximately%2012%20million%20
people%2040,due%20to%20uncorrected%20refractive%20error

Paper used image captioning, mentioned background for visually impaired people: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8990630&tag=1
