#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/Histograma1.png "Histograma_1"
[image2]: ./examples/Histograma2.png "Histograma_2"
[image3]: ./examples/img1.jpeg "Traffic Sign 1"
[image4]: ./examples/img2.jpeg "Traffic Sign 2"
[image5]: ./examples/img3.jpeg "Traffic Sign 3"
[image6]: ./examples/img4.jpeg "Traffic Sign 4"
[image7]: ./examples/img5.jpeg "Traffic Sign 5"
[image8]: ./examples/img6.jpeg "Traffic Sign 6"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used numpy to calculate summary statistics of the traffic
signs data set, including de Validation set that was not included in the jupyter notebook, the results where:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43
* Number of validating examples= 4410

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

As exploratory visualization of the data set I used a histogram showing how the data it is uneven distributed among classes, indicating that probably it will be necessary to perform data augmentation at the under-represented classes.

![alt text][image1]

After the data augmentation, process, I repeat the histogram, to validate the result.

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step I decided to perform data-augmentation, selecting the classes with less that 500 examples, I have performed  several experiments with different thresholds, but the results where equivalent and with 500, the training is quite rapid. For each image in the selected classes, two rotated images 10º left and 10º right are added.

The new number of training examples = 46919 from the original 34799.

Nexts steps are the ones recommended in the class-videos and forums and slack. Until I applied all of them I was unable to reach the 95% level.

* Grayscaling
* Histogram normalization
* Data Normalization

Forums where of great help for reshaping the image after the grayscaling, due to the characteristics of CV2.

All the steps are applied to the aumengted training, test and validation set.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Validation data was provided in this version of the project, so it was not necessary to split the training data into a training set and validation set and data augmentation is needed before preprocessing the images. 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.



My final model is LeNet architecture modified with drop-out in the fully-connected layers, I have been researching the Inception architecture, but I was advised to use Keras, so I decided to conserve plain vanilla LeNet architecture with the drop-out characteristic, especially once it reached 95% with the preprocessing of the images.



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I perform several experiments changing epochs, dropout probabilities, learning rate, and batch size. Dropout is the most sensitive variable that I have found to try to improve in some percentage point the result, but most of the experiments, worsened the results sometimes in catastrofic ways, and I was unable to obtain a significant improvement. The keys to obtain satisfactory results are data augmentation and preprocessing.




####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.



My final model results were:
* training set accuracy of 0.949
* validation set accuracy of 0.930
* test set accuracy of 0.947


 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]



* Images 1,2 and 3 are clear pictures and I were expecting a correct classfication.
* Images 4 and 6 are in a different perspective so are more difficult examples.
* Image 5 background may add noise and perhaps it is difficult to separate the traffic signal.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).



Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 11,Right-of-way at the next intersection      		| 11,Right-of-way at the next intersection 									| 
| 1,Speed limit (30km/h)   			| 1,Speed limit (30km/h)					|
| 14,Stop			| 12,Priority road			|
| 18,General caution   		| 9,No passing			|
| 12,Priority road	| 12,Priority road    							|
| 28,Children crossing | 12,Priority road  

The model was able to correctly guess 3 of the 6 traffic signs, which gives an accuracy of 50%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Traffic sign         	|     Predictions        					| Probabilities
|:---------------------:|:---------------------------------------------:| 
| 11,Right-of-way at the next intersection     	| 11, 30, 27, 28, 24  									| 9.74252045e-01,   2.57477257e-02,   2.16697359e-07,
          2.65665374e-08,   2.34724373e-09
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
