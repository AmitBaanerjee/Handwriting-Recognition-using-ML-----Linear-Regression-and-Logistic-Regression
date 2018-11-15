# Handwriting-matching using Logistic Regression and Linear Regression


Handwritting Recognition using Logistic Regression
One of the most challenging problems in Computer Science is recognition and identifying text written by human writers. This is mainly due to the abundance variation features used to represent each alphabet, symbol, and number written by humans. In order to help recognize various handwritings and classifying them with increased accuracy, we have used Machine Learning algorithms such as Linear Regression, Logistic Regression, and Artificial Neural Networks. In this project, our goal is to use the concepts of various Machine Learning algorithms and Gradient Descent to implement Handwriting Recognition on the CEDAR Research Lab’s ‘AND’ Dataset.

Overview
Handwriting Recognition can be defined as identifying writer for each handwritten document, sentence, word or even a single letter. The traditional approach for Handwriting Recognition will be to appoint a Human-writing Examiner. One who can recognize the writer based on some features such as curvature of a different letter, the spacing between the letter, etc. But this method is proven to be highly inefficient, as it provides low accuracy and also requires a lot of human labor. Hence, in order to implement a more efficient process and increase the accuracy of recognition, we need to implement a Handwriting Recognition process using Machine Learning. A Machine Learning based approach can be used to better understand the patterns between two identical letters or words provided in the dataset and then classify each element based on the patterns. One such real-world example of Handwriting Recognition using Machine Learning is the method of Name and Address Recognition used by United States Postal Service, to recognize and extract sender and receiver’s name and address, provided on the postal envelope or parcel. In this project, we use the concepts of Handwriting Recognition, wherein we follow approaches of implementing it using Linear Regression, Logistic Regression, and Artificial Neural Networks to identify authors of the handwritten text using some predefined features.

The CEDAR Letter Dataset
The Center of Excellence for Document Analysis and Recognition (CEDAR) letter dataset, is a huge dataset contains a variety of handwriting example, submitted by different kinds of 48 writers. In this project, we are going to use a dataset that represents various features of the 49 word ‘AND’ that is in the form of images that are extracted from each of the s ubmitted sample 50 manuscripts using a handwriting-mapping function. Hence, once extracted, we can work on two types of datasets, based on the feature extraction process;

Human Observed Dataset
The Human Observed Dataset mostly consists of cursive samples of the original dataset. In this dataset, the features for each ‘AND’ image have been manually extracted by a number of 56 human document examiner. Each image is represented in the form of 9 features in the dataset. Hence, a sample row of the dataset can be represented as follows;

GSC Dataset using Feature Engineering
The Gradient Structural Concavity algorithm is a type of Feature Engineering algorithm that generates a vector of size 2 from each of the handwritten ‘AND’ images. These values are between 0 and 1. Hence, a sample row of the dataset can be represented as follows;

Linear Regression Model
Linear Regression is one of the most used algorithms for prediction analysis. Linear regression can be described as a machine learning algorithm that is used to predict the value of the dependent variable Y, given enough amount of information about the independent variable X. More generally it uses the equation of the line to map values such that

                              Y =W * X
Here Y is the dependent variable, X is the independent variable and W corresponds to the weights 73 of the independent variables.

Logistic Regression Model
Logistic Regression is a classification algorithm that can be used to assign labels to data belonging discrete categories. Unlike linear regression which is a type of linear problem as try to divide a collection of continuous values using equations, logistic regression can be used to classify elements of a dataset using the logistic sigmoid function [3]. For example, if linear regression helps to predict the score of a baseball match, logistic regression can be used to help us predict whether our team will win the match or not.

                              Y = σ (W * X)
Here, Y is the dependent variable, X is the independent variable and W corresponds to the weights of the independent variables. The Sigmoid activation (σ) works like an activation function. The function maps the input data and provides the output value between 0 and 1. It can be represented using the following equation;

                              σ = 1 / 1 + e−Z
Here, Z is the input provided to the function.

Neural Network Model
In the past few years, there has emerged a new field of study in Machine Learning that is being used widely throughout the IT Industry and it is popularly known as Artificial Neural Networks. Artificial Neural Networks, also abbreviated as ANN, are build based on the concept of Human Brain. Like the human brain, the ANN is comprised of n number of neurons that work together to model complex pattern & prediction problems. For any ANN architecture, there is one input layer, one or more than one hidden layer & one output layer. The multiple hidden layers are used to determine distinctive patterns in the data taken as input & also to increase the accuracy of the model used. Weights ‘W’ is a concept that is used to determine the importance of each neuron in the network. Hence more the weight associated with a neuron, more important it is for generating the required output. In order to increase the efficiency of the neural network, we need to adjust weights assigned to every edge connecting two neurons in the network. Now let's try to understand some key points of a simple neural network using the figure in the article written by Jahnavi Mahanta;

In the figure above, the X1, X2 & X3 together comprise the input layer. Similarly, H1 & H2 are the neurons in the hidden layer & finally, O3 is the output layer. Moreover, W1 to W8 are all the weights assigned to each edge of the Neural Network such that

                            Output O = 1 / (1 + ℯ-F)
                          Here F = W1*X1 + W2*X2 + W3*X3
Algorithms & Equations used
The CEDAR Letter dataset consists of a Human Observed dataset of 9 features and a GSC dataset of 512 features that describe each handwritten ‘AND’ word. But the problem is that even though these features are very helpful, they are not in a formal order and are basically too many to be mapped using a Machine Learning model. Hence we first need to generate a dataset of valid inputs using two techniques, namely concatenation and subtraction. Then we need to reduce them to a potential number that can be used as an input to our model. This can be done using clustering & basis functions. Hence, let us first understand these concepts;

Dataset Generation using Concatenation
Concatenation in the terms of this dataset can be described as joining two sets of images based on their features. This is done in order for a Machine Learning algorithm to recognize the patterns between positive and negative result based on features. Hence, for the Human Observed Dataset, as one sample is represented using 9 features, after concatenation we will compare each sample using 18 features. This can be represented as follows:

Dataset Generation using Subtraction
Subtraction in terms of this dataset can be described as reducing features of one sample image to the order of values of features of another sample. This is done in order for a Machine Learning algorithm to recognize the difference or similarity between two handwriting samples. The algorithm can thus learn from these differences and classify new samples based on that. Hence, for the Human Observed Dataset, as we try to find the absolute difference between the two samples, we can represent it using 9 features as follows:

K-Means Clustering Algorithm
K-means clustering is one of the most used clustering algorithms. It is a type of unsupervised learning algorithm that is used to group unlabeled data into definite categories. It works iteratively to add each data point into a specific category of data defined by the user. K-means clustering works on the fundamentals of mean or centroid that we have to define and iterate upon in the data. These centroids are random data points in the dataset placed at equal distances from each other. K-means algorithm uses these data points as a reference medium to cluster another similar type of data around them. In this specific problem, we are using the data which is in the form of search queries and grouping them into clusters using K-means clustering. Hence a total of 69 thousand queries can be clustered together into 10 different query groups.

Radial Basis Functions
Radial Basis Function is a function that can be used to describe an element of the particular basis for a function space. Generally, Basis functions are used to represent the relationship between multiple non-linear inputs and target values in a dataset. In this project, we are going to use Gaussian Radial Basis Functions. It calculates the radial basis functions using the following formula:

                  Φj(x) = exp (−0.5 * (x – μj)T * ∑j-1 * (x – μj))
Where μj is the centroid obtained for the Basis function and ∑j-1 is the covariance matrix.

Gradient Descent
Gradient Descent is an optimization algorithm that uses gradients of the cost function to minimize the loss value. We use gradient descent to find the local minimum value of Weights W of the input values in this project. One of the easiest ways of learning Gradient Decent is to take an example of the two mountains and a valley. Here we as a traveler have the goal of reaching the local minimum or lowest points of the valley. We can achieve these goals by taking steps along the side of the mountain to reach the goal. If we take too small steps, then it will take a lot of time to reach the goal and on the other hand, if we take to large steps then we may skip the goal altogether. Considering the example above we can now map the gradient descent algorithm into the following graph, where the two ends of the graph are the mountains and the value X is the lowest point of the valley. We can understand this better by looking at the diagram drawn by Welch Labs;

Image

One of the most used variations of gradient descent algorithm is the Stochastic Gradient Descent Algorithm; also abbreviated as SGD. The term stochastic means that we compute a part of the problem representing the whole. Hence unlike the Batch Gradient Decent or Gradient Decent algorithm, in SGD we compute the gradient based on a single training sample as the stochastic approximation of the whole true gradient of the problem.

Implementation using Linear Regression
Now that we understand the concept of Concatenation, Subtraction, K-means Clustering, Radial Basis Functions & Gradient Descent, we can look into the three major approaches of Machine Learning to implement handwriting recognition on the CEDAR Letter Dataset. We have used the Error Function of Linear Regression Algorithm to calculate the Erms values as follows:

                            ED(W) = ∑n=1…N {tn|wT * Φ(xn)}2
Here ED is the change in error in the output.

For Human Observed Dataset
The Human Observed Dataset can represent a sample using 9 features and hence we can use the dataset generated by the concatenation method (18 features) or the dataset generated using the subtraction method (9 features). In both the approaches, we try to reduce the error by varying the Hyper Parameters. Here we are evaluating the performance of the Linear Regression Model based on the following parameter;

Root Mean Square error (RMS): This parameter can be defined as the square root of the differences between the actual output and the expected output, which is wholly divided by the number of outputs. The values of the RMS after we varied the Hyper-Parameter values are as shown in the graph below:
For GSC Dataset
The GSC Dataset uses Feature Engineering to extract 512 valid features from the Handwriting samples. As we cannot input all the features at once, we need to use radial basis functions to reduce the nonlinearity of the dataset. Now we can use the dataset generated by the concatenation method (1024 features) or the dataset generated using the subtraction method (512 features) with reduced dimensionality as an input to the Machine Learning algorithm. In both the approaches, we try to reduce the error by varying the Hyper Parameters. Here we are evaluating the performance of the Linear Regression Model based on the following parameter;

Root Mean Square error (RMS): This parameter can be defined as the square root of the differences between the actual output and the expected output, which is wholly divided by the number of outputs. The values of the RMS after we varied the Hyper-Parameter values are as shown in the graph below:
Implementation using Logistic Regression
In order to classify the problem into 1 or 0, we have implemented Logistic Regression on the CEDAR Dataset using both the Concatenation and Subtraction methods. In order to calculate the error between the output of the model and the desired output, we have used following Error Function in Logistic Regression:

                  J(θ) = ∑i=1…m [yi * log(hθ(xi))+(1-yi) * log(1-hθ(xi))]
Here J(θ) is the Error Function, yi is the output variable and xi is the input variable for the Error Function. Here we are evaluating the performance of the Logistic Regression Model based on the following parameter;

Accuracy: This parameter can be defined as the degree to which the result generated by the model resembles the correct and predefined output The accuracy values after we varied the Hyper-Parameter values are as shown in the graph below:
Implementation using Artificial Neural Networks
In order to increase efficiency and increase accuracy, we have implemented an Artificial Neural Network using Keras and Tensorflow Libraries. Here, we have used ‘rmsprop’ as the optimizer for Human Observed Dataset and ‘sgd’ optimizer for GSC Dataset. Also, the activation function used is ‘relu’ activation function. Here we are evaluating the performance of the Logistic Regression Model based on the following parameter;

Accuracy: This parameter can be defined as the degree to which the result generated by the model resembles the correct and predefined output Hence, the accuracy values after we varied the Hyper-Parameter values are as shown in the graph below:
Conclusion
This project aims at using machine learning to implement Handwriting Recognition problem using Machine Learning algorithms. By performing this project, we were able to understand how Linear Regression differs from Logistic Regression. We were also able to learn how to effectively fit on a nonlinear dataset i.e. dataset containing more than two features on multiple Machine Learning algorithms. By performing this project, we gained the knowledge of preprocessing datasets using various techniques such as Concatenation and Subtraction. From the observation of various Erms & Accuracy measures, we can deduce the following conclusions:

For the Human-Observed Dataset, as we are selecting an equal number of same and different pairs of writers for training, the model produces a result of 0.50 Erms values and accuracy of around 50% for both the Linear Regression and Logistic Regression model.
Similarly, even though the GSC Dataset has more samples for the models to train on, we do get to see a minor increase in the accuracy, but as the dataset is equally distributed, the result is still around 0.50 Erms values and accuracy is around 50% for both the Linear Regression and Logistic Regression model.
For Neural Network, we have observed that initially we can obtain accuracies similar to that of Linear and Logistic Regression models, but once we try to tune the Hyper Parameters such as increasing the number of input samples and adding more hidden layer and nodes, we see a drastic increase in accuracy for the model.
