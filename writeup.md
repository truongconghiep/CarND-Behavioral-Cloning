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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md for the project report

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



## Data collection

First, I drive the some rounds around track 1 to collect data images. In each frame, the simulator generates 3 images from 3 different camera, which are mounted in then center, on the left and on the right of the car 

![alt text][image1]

After finish the first drive on track 1 I recognised, that the track has more left curves than right curves, that might make the data set unbalanced, then I decided to drive 2 rounds more on the reverse direction to collect more data

In the next step, I crop the collected images to reduce data size. The images contain a lot information, for example: road, tree, sky. Some information is not necessary for training, therefore we can cut them out of images. I decide to remove 50 pixel on top of images, wherec contains sky and 20 pixels at the bottom, where contains the hood of the car. The rest of images contain useful information, like road and landlines.

![alt text][image2]

## First achitecture (an appropriate model architecture)
#### 1. An appropriate model architecture

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

#### 2. Attempts to reduce overfitting in the model

After the first train, the car drives very unconfident and often get out of the road, therefore I have tried to tune the number

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy


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


#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
