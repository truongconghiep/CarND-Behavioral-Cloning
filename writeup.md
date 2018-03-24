# **Behavioral Cloning** 

#### Hiep Truong Cong

In this project of Udacity Self-Driving Car Nanodegree I use a deep CNN to train a model to drive a car in a simulator provided by Udacity. First the car was driven by a human a round a track to collect data images und driving information for the training. Then a CNN model was trained with the collected data to drive itself around the track.

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)
[image1]: ./output_image/data image.png
[image2]: ./output_image/cropped_img.png
[video1]: ./pvideo.mp4

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md for the project report.

## Data collection

First, I drive the four rounds around track 1 to collect data images. In each frame, the simulator generates 3 images from 3 different camera, which are mounted in then center, on the left and on the right of the car 

![alt text][image1]

After finish the first drive on track 1 I recognised, that the track has more left curves than right curves, that might make the data set unbalanced, then I decided to drive 2 rounds more on the reverse direction to collect more data

In the next step, I crop the collected images to reduce data size. The images contain a lot information, for example: road, tree, sky. Some information is not necessary for training, therefore we can cut them out of images. I decide to remove 50 pixel on top of images, wherec contains sky and 20 pixels at the bottom, where contains the hood of the car. The rest of images contain useful information, like road and landlines.

![alt text][image2]

Finally I generated 25083 training images

## First achitecture (an appropriate model architecture)

To get started I decided to use the simple model, provided in lectures of Udacity. The model includes two convolution layers and three full-connected layers.

<pre><code>
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 86, 316, 6)        456       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 43, 158, 6)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 39, 154, 16)       2416      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 19, 77, 16)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 23408)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 120)               2809080   
_________________________________________________________________
dense_2 (Dense)              (None, 84)                10164     
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 85        
=================================================================
Total params: 2,822,201
Trainable params: 2,822,201
Non-trainable params: 0
</code></pre>

After the first train, the car drives very unconfident and often get out of the road, therefore I have tried to tune the number of epochs. After some tries, the car still got stuck, then I tried to augment the training data to build a bigger data set by flipping the images. With the augmented data set I trained the model some more times, but the result is still not as expected. I realised, may be the simple model is not powerful enough, so I went further to apply the model, which is introduced by NVIDIA.

### Final Architecture and Training Strategy

I choose the model below as my final model. The model has five convolution layers and four fully-connected layers. 

<pre><code>
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_2 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_2 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_5 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_6 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
</code></pre>


#### Training 

To train the model I used Adam optimizer with its default learning rate. 

In the input layer the input image is normalized. The input image size is 90x320x3. By using Adam optimizer, the training data set is splited in training set (about 80%) and validation set (about 20%).
I tried to train the model with different number of epochs. My experiments shows the model was well trained with 5 epochs. With a larger number of epoch the car does not keep on the road any more. 


At the end of training process, I got a pretty good test accuracy and validation accuracy

<pre><code>
19808/20066 [============================>.] - ETA: 3s - loss: 0.0163
19840/20066 [============================>.] - ETA: 2s - loss: 0.0163
19872/20066 [============================>.] - ETA: 2s - loss: 0.0163
19904/20066 [============================>.] - ETA: 1s - loss: 0.0163
19936/20066 [============================>.] - ETA: 1s - loss: 0.0163
19968/20066 [============================>.] - ETA: 1s - loss: 0.0163
20000/20066 [============================>.] - ETA: 0s - loss: 0.0163
20032/20066 [============================>.] - ETA: 0s - loss: 0.0163
20064/20066 [============================>.] - ETA: 0s - loss: 0.0163
20066/20066 [==============================] - 257s 13ms/step - loss: 0.0163 - val_loss: 0.0375
</code></pre>

At the end of the process, I run the trained model in the simulator and finally the vehicle is able to drive autonomously around the track without leaving the road.

## Result and discussion


