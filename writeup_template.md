# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/train_hist.png "Visualization-train"
[image2]: ./examples/valid_hist.png "Visualization-valid"
[image3]: ./examples/test_hist.png "Visualization-test"
[image4]: ./examples/visualisation_sample_images.png "Visualization-Sample-images"
[image4]: ./examples/grayscale.jpg "Grayscaling"
[image5]: ./examples/random_noise.jpg "Random Noise"
[image6]: ./test_images/13.jpg "Traffic Sign 1"
[image7]: ./test_images/17.jpg "Traffic Sign 2"
[image8]: ./test_images/22.jpg "Traffic Sign 3"
[image9]: ./test_images/23.jpg "Traffic Sign 4"
[image10]: .test_images/25.jpg "Traffic Sign 5"
[image11]: ./examples/test_image_1_conv_1.PNG "Convolvution Layer1 Feature map for test_image 1"
[image12]: ./examples/test_image_1_conv_2.PNG "Convolvution Layer2 Feature map for test_image 1"
[image13]: ./examples/test_image_2_conv_1.PNG "Convolvution Layer1 Feature map for test_image 2"
[image14]: ./examples/test_image_2_conv_2.PNG "Convolvution Layer2 Feature map for test_image 2"

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.  As per below charts, the amount of examples per class in every dataset show similar shapes and regardless from not having an uniform distributions, the amount of examples per class in the training set is proportional to the same class in validation and test datasets per class.

Training                   |  Validation               |  Testing
:-------------------------:|:-------------------------:|:-------------------------:
![][image1]                |  ![][image2]              |  ![][image3]

The images come in RGB format and are already contained in numpy arrays and the labels come as a list of integers.

*Image data shape = (32, 32, 3)
*Number of classes = 43

Every image has a corresponding label and these labels correspond to a category of traffic signal. These categories can be seen in the file signnames.csv.

![][image4]  

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, Normalization of the data is done to have zero mean and equal variance using pixel depth.Here we have defined a pixel depth of 128.We normalized the train,valid and test dataset using (pixel - 128)/ 128 which is a quick way to approximately normalize the data.In Normalization the values of the intensity image no longer go from 0 to 255, but they range now from -1 to 1 in floating point format. After normalization we shuffle the train,valid and test dataset in order to get the accuracy.

```
pixel_depth = 128
norm =  ((test.astype(float))- pixel_depth) /pixel_depth
```

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten               | outputs 400    								|
| Fully connected		| Input = 400 and Output = 120					|
| RELU					|												|
| Fully connected		| Input = 120 and Output = 84 					|
| RELU					|												|
| Fully connected		| Input = 84 and Output = 43 					|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, Below parameter values.
* Optimizer: AdamOptimizer
* batch size : 128
* epochs : 50
* Learning Rate : 0.0009
* Dropout : 0.7 
* Regularization factor = 5e-4 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.950 (95.0%)
* test set accuracy of 0.926(92.6%)


Used the same [LeNet architecture](http://yann.lecun.com/exdb/lenet/) but normalization method is different.It has been done using formula ((test.astype(float))- 128) /128.
Also, I have tweaked the learning rate(to 0.0009) and epoch (to 50).
After this change, I was able to improve the accuracy of the model as mentioned above.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

Below are the qualities of test images which might be difficult to classify:
The images might be difficult to classify because they are in different size, with water mark and clear shape with no shadow.
* Test 2 & 4 have watermarks.
* Test 1 is tilted slightly which may cause difficulty for the model.
* Test 3 has leaves in some part of the shape and also shadow falling on it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work      		| Road Work    									| 
| Bump Road    			| Priority Road    								|
| No Entry				| No Entry  									|
| Yield 	      		| Yield  						 				|
| Slippery Road			| Slippery Road  								|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
The 5 probabilities are the following :

Values per image = [  9.99999881e-01   1.44251814e-07   4.88518648e-09   2.90861918e-11  2.84980678e-14]
                   [  9.99971271e-01   1.59520787e-05   1.21704834e-05   2.26005412e-07  1.66006856e-07]
                   [  9.99999881e-01   3.94416695e-08   3.04867633e-08   1.05452322e-08  4.73372008e-09]
                   [  9.97203112e-01   2.72253528e-03   7.43896235e-05   9.44109146e-09  1.19879473e-09]
                   [  1.00000000e+00   1.21382693e-09   2.14998141e-11   4.40841897e-16  1.24699529e-16]

Index per image =  [25 20 26 18 36]
                   [25 22 29 30 23]
                   [17 26 34 20 23]
                   [13  2 38 14 34]
                   [23 11 41 20 28]
                   
-----------------------------
image 1 	truth: Road work
Top five Predictions: 
['Road work' 'Dangerous curve to the right' 'Traffic signals'
 'General caution' 'Go straight or right']

-----------------------------
image 2 	truth: Bumpy road
Top five Predictions: 
['Priority Road' 'Bumpy road' 'Bicycles crossing' 'Beware of ice/snow'
 'Slippery road']

-----------------------------
image 3 	truth: No entry
Top five Predictions: 
['No entry' 'Traffic signals' 'Turn left ahead'
 'Dangerous curve to the right' 'Slippery road']

-----------------------------
image 4 	truth: Yield
Top five Predictions: 
['Yield' 'Speed limit (50km/h)' 'Keep right' 'Stop' 'Turn left ahead']

-----------------------------
image 5 	truth: Slippery road
Top five Predictions: 
['Slippery road' 'Right-of-way at the next intersection'
 'End of no passing' 'Dangerous curve to the right' 'Children crossing']

For image 2,the prediction was wrong.The Actual value is found in 2nd place.
 
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Read an test_input image and reshaped into the default shape 1,32,32,3.Then using the "graph.get_tensor_by_name" function the visualization of the featuremaps of each convolvution layers has been made.
ex: Here there are two convolvution layers say conv1 and conv2,with depth depths 6 and 16 respectively.

The above are the test images taken
![alt text][image9] ![alt text][image7]

Here the outputs of the feature maps of test image 1 for both convolvution layers
![alt text][image11] ![alt text][image12] 

and the outputs of the feature maps of test image 2 for both convolvution layers
![alt text][image13] ![alt text][image14] 

