# Attention U-Net for Anatomical Segmentation and Enlargement Detection

## Overview 
<br> The CXR Attention U-Net is a deep learning model designed to create masked segmentations of chest X-ray (CXR) images. It integrates attention mechanisms into the traditional U-Net architecture to enhance the model's focus on the relevant anatomical regions, thereby improving the accuracy of the segmentation. The model is based on the Attention U-Net architetcure. These gates allow the network to concentrate on pertinent regions in the input images, effectively suppressing irrelevant features and highlighting prominent features. In a medical imaging task such as this, where the regions we want to segment are often small and vary in size and shape, these features are very benefical. Beyond segmentation, we also compute a clinical metric called the cardiothoracic ratio, CTR, which is a ratio of the heart’s width and chest width to detect conditions like cardiomegaly. 

<br> [Presentation Slides](https://docs.google.com/presentation/d/1lrFUmw1toBuzCr4wEwi2cEgJF9aRI2ZorvTLPGRc6zQ/edit?usp=sharing)

## Masked Segmentation 
Our model takes an input of a chest X-ray and outputs a segmented mask, where there are three possible labels for any given pixel: lung, heart, and background. 

ADD BARE X-RAY AND SEGMENTED X-RAY HERE

## Cardiothoracic Ratio (CTR)
The CTR is computed by measuring the width of the chest and the width of the heart. A 'normal CTR,' which would suggest a patient is healthy, is between .42 and .5. Anything greater than .5 suggests a diagnosis of cardiomegaly. 

## Results 
Based on the model evalutation, here are our results:
### Hyperparameters
After some hyperparameter tuning, we determined that the best hyperparameters are: `batch_size` = 8 and `learning_rate` = .0001.
### Binary Cross Entropy Loss
ADD GRAPH HERE

Here is the progression of the training and validation losses throughout the training process. Initially, all loss curves drop sharply which indicates that the model was able to learn the basic structure of segmentation fairly quickly.

Losses continue to decline more gradually, suggesting the model is refining it’s understanding of the boundaries we want. The gap between training and validation loss is small which suggests the model is maintaining good generalization. 

Towards the end of the training, both losses plateau at a low value. The test loss is stable and close to validation loss, which indicates the model has not overfit and will perform well on unseen data. The values on the slide represent the final values observed. The steady decline in all losses confirms that our model has learned effective representations over time.

