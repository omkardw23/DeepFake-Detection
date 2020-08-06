# DeepFake Hunter : 

![](https://github.com/omkardw23/DeepFake-Detection/blob/master/Images%20for%20readme/fake_image_title.jpg)

# (I) Abstract : 

Deepfake techniques, which present realistic AI-generated videos of people doing and saying fictional things,  have the potential to have a significant impact on how people determine the legitimacy of information presented online. These content generation and modification technologies may affect the quality of public discourse and the safeguarding of  human rights—especially given that deepfakes may be used maliciously as a source of misinformation, manipulation, harassment, and  persuasion.  Identifying such manipulated media is of paramount importance in today's cyberspace.  

# (II) Dataset Used : 

The data is comprised of .mp4 files, split into compressed sets of ~10GB apiece. A metadata.json accompanies each set of .mp4 files, and contains filename, label (REAL/FAKE), original and split columns, listed below under Columns. We will be predicting whether or not a particular video is a deepfake. A deepfake could be either a face or voice swap (or both). In the training data, this is denoted by the string "REAL" or "FAKE" in the label column. In your submission, you will predict the probability that the video is a fake.

Kaggle's dataset link : 

**https://www.kaggle.com/c/deepfake-detection-challenge/data**

Dataset summary : 
* Number of training video samples : **401**
* Number of testing video samples : **400**

Total size ~ 470 GB

![](https://github.com/omkardw23/DeepFake-Detection/blob/master/Images%20for%20readme/Deepfake%20Detection%20Challenge%20-%20Kaggle.mp4)

(I) General-Image-Processing module : 
------------------------------------------
This notebook was created to implement the code trying to extract the face/s using Viola Jones algorithm, that is by using Haar cascades designed in OpenCV. The haarcascade can be downloaded from :
https://github.com/opencv/opencv/tree/master/data/haarcascades 
or 
are available in Kaggle's data repository as well. Using the pre-defined openCV functions we were able to detect the faces successfully. 
Moreover, the code also contains necessary function defined from graying the cropped face image/s, and histogram equalizing it(in order to spread the intensity near uniformly across the gray value range). Moreover, to analyze the frequencies present, a 2D DFT was also computed. 

(II) Model 1 :
----------------
----------------
MTCNN :
--------
MTCNN (Multi-task Cascaded Convolutional Neural Networks) is an algorithm consisting of 3 stages, which detects the bounding 
boxes of faces in an image along with their 5 Point Face Landmarks. Each stage gradually improves the detection results by passing it’s inputs through a CNN, which returns candidate bounding boxes with their scores, followed by non max suppression.

In stage 1 the input image is scaled down multiple times to build an image pyramid and each scaled version of the image is passed through it’s CNN. In stage 2 and 3 we extract image patches for each bounding box and resize them (24x24 in stage 2 and 48x48 in stage 3) and forward them through the CNN of that stage. Besides bounding boxes and scores, stage 3 additionally computes 5 face landmarks points for each bounding box. 


Mesonet : 
--------------------------
MesoNet is based on mesoscopic level of analysis. It is a Convolutional Neural Network architecture, used to detect forged images (DeepFakes and Face-to-Face). There are two variants- Meso-4 and MesoInception-4. MesoInception-4 is obtained by replacing the first two convolutional layers of the Meso-4 architecture by the famous Inception module proposed by Szegedy et al. in Going deeper with convolutions. Cvpr, 2015.
( https://www.researchgate.net/publication/327435226_MesoNet_a_Compact_Facial_Video_Forgery_Detection_Network )

## Pipeline :
----------------
The process is to input the video train/test files directly and extracting multiple frames from the video files. After each frame is extracted we use face detection libraries like face_recognition and MTCNN to extract the regions in the frame which contain faces. We then generate batches of these frames for each video file. These face snaps or bounding boxes containing only detected faces are then passed through the MesoNet architecture. The final prediction depends on how many positive outputs we get from each batch of extracted faces (after passing every selected frame of that video through the MesoNet architecture). From one video we have to process at least 20 or more frames to get reliable predictions.

Steps to detect a DeepFake in test dataset: 

1. Train the MesoNet model by https://github.com/DariusAf using the train dataset available on the kaggle DeepFakes Detection Challenge (https://www.kaggle.com/c/deepfake-detection-challenge/data) or load the pre-trained MesoNet model by the authors of MesoNet. The train data video/ frames may also be preprocessed using image processing techniques for feature extraction or faster training (Using transforms or equalization or other IP techniques)

2. After training the model collect the test dataset from the same source in a seperate folder (name it TEST_VIDEOS).

3. Next, each video file is processed to extract frames containing faces from the video. For this we use OpenCV to loop through all the frames of the video and then pass these frames through the MTCNN detector() function which returns us the annotations or values of the detected faces (these contain the face landmark values, the bounding box coordinates of the faces, keypoints and their confidence values).

4. Using the bounding box values for all the frames, we then separately export these frames as images using the Python Image Library (PIL). These images will be the input to our model in the form of batches which will be generated using the ImageDataGenerator() function (a keras.preprocessing module).

5. In this step, we process the images from the previous step and extract features from the image using standard image processing techniques like Histogram Equalization, etc. This step is optional and its inclusion will depend on the train dataset images. If the train data was preprocessed, then these images must be too using the same techniques.

6. Now we predict the DeepFake probability of these frames using MesoNet. Each frame is predicted and the result for each frame of the video is stored in a vector. This vector represents the combined probabilities of all the frames of the video. Using this vector we predict if the video is a DeepFake or not. If a number of consecutive frames have lower output values (float), then the video is stated as a DeepFake.  

7. Each video file in the 'TEST_FOLDER' is predicted using the above steps.







