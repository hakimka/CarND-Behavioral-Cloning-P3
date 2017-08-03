#**Behavioral Cloning** 


---

**Behavioral Cloning Project**

This project includes a simulator to collect driving behavior of a user. Using Karas I built a convolution network that leaned steering angles on a simulation track. The model got trained and validated on images and the angles collected during simulation. The highlights and details of the project are presented in this document. 

[//]: # (Image References)

[image1]: ./images/DNNArch.jpg "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model 
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed


My model consists of a 
- Preprocessing layer where the pixels values on 3 channel are normalized with mean in 0. (line 69)
- Preprocessing also crops the top part and bottom part of the input image (line 70)
- Convolution layer #1: kernel 5x5 with 24 filters, activation 'relu', followed by max pooling with kernel 2x2. (lines 72-74)
- Convolution layer #2:kernel 5x5 with 36 filters, activation 'relu', followed by max pooling with kernel 2x2.  (lines 75-77)
- Convolution layer #3:kernel 5x5 with 48 filters, activation 'relu', followed by max pooling with kernel 2x2. (lines 78-80)
- Convolution layer #3:kernel 3x3 with 64 filters, activation 'relu' (lines 81-82)
- dropout 10% (line 83)
- Fully connected layer #1 (flatten first) of 1164 nodes, followed by 'relu' activation. (lines 84-87)
- Fully connected layer #2 of 100 nodes, followed by 'relu' activation. (lines 88-90)
- Fully connected layer #3 of 50 nodes, followed by 'relu' activation. (lines 91-93)
- Fully connected layer #4 of 10 nodes, followed by ' relu' activation. (lines 94-96)
- Fully connect final layer of 1 node (line 98) 
- 
####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 83). 

The model was trained and validated on different sections of data set to ensure that the model was not overfitting (code line 48-52). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 101).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also used flipping images (mirroring) to double the size of the data set.  

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to try something small experiment and see if the car drives on the tracks. 

My first step was to use a convolution neural network model similar to the AlexNet. 
Before I chose the presented model, I tried several models. Reading the discussion on the forum section for the Behavior Cloning  thread, I noticed some people suggested simple 2-3 conv layer network following by fully connected layers. I started a simple 3 conv layer network, that did not seem to get me anywhere. I increased a number of layers and made but keept my filter low (under 10 on each layer). After reviewing lectures from udacity, I decided to use a network architecture used by NVidia team for self driving car. That got me futher, now my car was able to drive some distance on the track, but not all the way through. After I increased the kernel size on the first layers to 5x5, I got the desired result. 
 

None of the models (the successful or failed) exhibited over fitting. I never had to deal with growing accuracy on the training set and degrading on the validation..

The biggest "aha" moment was to add the dropout of 10% from the convolutional layers to fully connected. Then the things started to work. 


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 73-98) consisted of a convolution neural network with the layers described in the previous section. 

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

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
