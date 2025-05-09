# Attention U-Net for Anatomical Segmentation and Enlargement Detection

## Overview 
The CXR Attention U-Net is a deep learning model designed to create masked segmentations of chest X-ray (CXR) images. It integrates attention mechanisms into the traditional U-Net architecture to enhance the model's focus on the relevant anatomical regions, thereby improving the accuracy of the segmentation. The model is based on the Attention U-Net architecture. These gates allow the network to concentrate on pertinent regions in the input images, effectively suppressing irrelevant features and highlighting prominent features. In a medical imaging task such as this, where the regions we want to segment are often small and vary in size and shape, these features are very benefical. Beyond segmentation, we also compute a clinical metric called the cardiothoracic ratio, CTR, which is a ratio of the heart’s width and chest width to detect conditions like cardiomegaly. 

Our main objective was to accurately segment and isolate the heart and lungs from chest X-ray images. We used the MIMIC-CXR dataset as our input and trained an Attention U-Net model to generate precise organ-specific masks. We begin with a raw chest X-ray. This image is passed into our heart and lung segmentation network—represented here in the center. The model processes the image through multiple convolutional and attention layers designed to extract relevant anatomical structures. The output is a set of binary segmentation masks: one for the heart and another for both lungs. These masks clearly outline the boundaries of each organ, isolating them from surrounding tissues. This kind of spatial localization is critical for downstream medical analysis. As part of post-processing, we overlay the predicted masks onto the original X-ray. This allows us to extract measurable features. Here we demonstrate how you could calculate the CTR. This experiment confirms that our model can not only generate clean and anatomically correct masks, but also supports clinical measurements that could aid in diagnosis.


<img width="582" alt="Screenshot 2025-05-08 at 5 32 22 PM" src="https://github.com/user-attachments/assets/895be860-cc7b-42a4-8052-212b1baeab6c" />

<br> [Presentation Slides](https://docs.google.com/presentation/d/1lrFUmw1toBuzCr4wEwi2cEgJF9aRI2ZorvTLPGRc6zQ/edit?usp=sharing)

## Dataset  

### Chest X-ray Dataset (MIMIC-CXR)
For training and validation purposes, we used the [MIMIC_CXR](https://physionet.org/content/mimic-cxr/2.1.0/) dataset. This is a publicly available set containing over 370,000 chest x-ray scans corresponding to over 250,000 radiologist studies. We used a random subset of around 500 due to timing constraints, with a train/validation/test split of 80/15/5 which was randomly selected within the subset. In preprocessing, images were all converted to greyscale and resized to a fixed resolution of 512x512 pixels. Normalization and tensor conversion were also applied at this step. One thing we would like to note, is that the dataset includes chest X-rays exclusively from Beth Israel Deaconess Medical Center in Boston, MA. With more time and resources, in order to provide better generalization it would be beneficial to include more diverse scans collected from different hospitals in different parts of the world. 

### Segmentation Mask Labels
For the ground truth segmentation labels, we used the [CheXmask database](https://github.com/ngaggion/CheXmask-Database), which provides separate binary masks for the heart and lungs corresponding to each chest X-ray image.
<img width="957" alt="Screenshot 2025-05-09 at 4 16 32 PM" src="https://github.com/user-attachments/assets/25993268-7577-4b38-b0f8-ef5931c2e36e" />


Our model takes an input of a chest X-ray and outputs a segmented mask, where there are three possible labels for any given pixel: lung, heart, and background. On the left is an example of one of our inputs before the model's prediction, on the right is the model's output.

<img width="312" alt="Screenshot 2025-05-08 at 4 24 11 PM" src="https://github.com/user-attachments/assets/0373bac4-186b-40a6-9e0c-e131189e66f7" /><img width="312" alt="Screenshot 2025-05-08 at 4 23 45 PM" src="https://github.com/user-attachments/assets/7348f2f5-9862-49b0-aad0-5c853543def6" />

## Cardiothoracic Ratio (CTR)
The CTR is computed by measuring the diameter of the chest and the diameter of the heart. A 'normal CTR,' which would suggest a patient is healthy, is between .42 and .5. Anything greater than .5 suggests a diagnosis of cardiomegaly. CTRs for both the ground truth and predicted masks were calculated, then compared with mean and absolute error to give us an idea of how well our segmentation translated to the true diagnosis. In our model, the diameter of the heart and chest are derived following the segmentation of the X-ray. All non-zero pixels in the heart mask are identified, their x-coordinates extracted, then the minimum coordinate is subtracted from the maximum coordinate to compute the heart diameter. The same process computes the chest diameter, except the lung mask is utilized. 

<img width="312" alt="Screenshot 2025-05-08 at 5 31 35 PM" src="https://github.com/user-attachments/assets/52da07a8-913b-4770-bb55-650812d9cdae" />
  
## Results 
Based on the model evalutation, here are our results:

### Hyperparameters
#### Dataset / DataLoader
input_size: (512, 512)
batch_size: 8

#### Training schedule
Epochs: 30  
Training/validation/test split: 80 % / 15 % / 5 % 

#### Model architecture
n_channels: 1 (grayscale input)  
n_classes: 3 (background, lung, heart)  
Encoder feature sizes: [128, 256, 512, 1024]  
Bottleneck channels: 1024→2048  
Attention intermediate channel: half of each decoder feature size (F_int = f // 2)  
#### Optimization
Optimizer: Adam  
Learning rate (lr): 1 × 10⁻⁴  
Loss function: CrossEntropyLoss  

#### W&B logging

log_freq: 100 steps for gradient/parameter logging  
Image‐logging: 4 random val samples per epoch  
Logged metrics: train_loss, val_loss, test_loss, IoU for lung & heart  

### Model Training
<img width="520" alt="Screenshot 2025-05-08 at 6 06 01 PM" src="https://github.com/user-attachments/assets/d9ec809d-4fff-442a-a314-8328d288a1ab" />

Here is the progression of the training and validation losses throughout the training process. Initially, all loss curves drop sharply which indicates that the model was able to learn the basic structure of segmentation fairly quickly. Losses continue to decline more gradually, suggesting the model is refining its understanding of the boundaries we want. The test loss is stable and close to validation loss, which indicates the model has not overfit and will perform well on unseen data. The values on the slide represent the final values observed. The steady decline in all losses confirms that our model has learned effective representations over time. In order to make the model more complex, we updated the feature map size to double at every block, increased the bottleneck channels, and had the decoder upsample back through the wider feature maps. 

### CTR and Classification 
Out of the 474 X-rays we analyzed, 233 were known to be positive for cardiomegaly while 241 were negative for the condition. The box plot below shows our model's predicted CTR distribution, with orange representing the cases with cardiomegaly. After removing IQR outliers and computing both a Welch's t-test and a Mann-Whitney U Test, we were able to reject the null hypothesis of the two conditions belonging to the same distribution. Our results for the two tests are as follows: for the T-Test we got a test statistic of -11.543 with a p-value of 6.569e-27 and for the Mann-Whitney U Test we computed the test statistic to be 10405.5 and a p-value of 1.758e-25. Using the standard 95% confidence interval, we can safely reject the null hypothesis which assumed the two populations share an identical distribution. 

<img width="371" alt="Screenshot 2025-05-09 at 10 41 49 AM" src="https://github.com/user-attachments/assets/49de0a97-8845-4598-b295-f51707f2a7e9" />

### Project Structure
  

```
├── CheXmask-Database/        # External repo of pixel-level heart & lung masks
│   └── …
├── MIMIC-preprocessing/      # Scripts to normalize & resize raw MIMIC-CXR images
│
├── processed_masks/          # Output masks generated by preprocessing-ctr.sh
│   ├── heart/                # Binary heart masks
│   └── lungs/                # Binary lung masks
│
├── Experiment.ipynb          # Statistical analysis: CTR distributions, ROC/AUC, t-tests
│
├── preprocessing-ctr.sh      # Batch-submission script (NYU HPC) to build masks from landmarks
├── preprocessing-ctr.py      # Build segmentation masks from landmarks for ground-truth
├── training-ctr.sh           # Batch-submission script (NYU HPC) to train Attention U-Net
├── training-ctr.py           # Main training & CTR extraction code (Attention U-Net)
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview & structure
└── LICENSE                   # License information
```


Our ROC curve yields an AUC of .787, which indicates a good separation. We pick the Youden-J optimum at CTR= .573, shown on the curve as the red dot, which balances sensitivity and specificity. At that cutoff, the model scores a precision of .803 and a recall of .825 for classification of cardiomegaly. Under our project's circumstances, optimizing recall reflected the priority of catching as many true cardiomegaly cases as possible. 

<img width="281" alt="Screenshot 2025-05-09 at 10 50 36 AM" src="https://github.com/user-attachments/assets/69b7dc62-e6e9-4978-9407-caab75e265e0" />
