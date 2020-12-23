# CNN-Image-Classification

In this project, a study of optimization algorithms is conducted for the classification of images. Convolutional neural networks have been used to learn the features 
present in each set of images. 
A study of the features learnt by the network is also done in this project.
The project is implemented using the Keras library.

The conclusions obtained after the completion of the project are:

As we can see in the case of ADAM optimizer the training data has been overfit because of which we get 99% accuracy in the training data set. But in the validation data set, we get the final accuracy to be around approximately 83%. Since the data is being overfit, we can conclude that the model beings to remember the input image patterns instead of learning from the input images. Hence, because of this we get a constant decrease in training loss and an increase in the validation loss while executing the “model.fit” function.

![alt text](https://github.com/rohitmurali8/CNN-Image-Classification/blob/master/adam.PNG)

In case of the RMSProp optimizer, there is a steady increase in the training accuracy but the model does overfit the data since the training accuracy is not very high. The validation accuracy increases in the start and saturates after a certain time. While observing the training and validation loss plots, we can see that both of the losses decrease steadily with some increasing spikes in the validation loss.

![alt text](https://github.com/rohitmurali8/CNN-Image-Classification/blob/master/rmsprop.PNG)


While using the ADADelta optimizer we can see that the training and validation accuracies and losses are almost the same. Although this is a good sign for predictions on unseen data, the model as whole would not be very accurate in classifying images as the accuracy obtained during the training is just approximately 80%.

![alt text](https://github.com/rohitmurali8/CNN-Image-Classification/blob/master/adadelta.PNG)

To reduce the overfitting problem we can use one of the following methods:

1 – Regularization

2 – Decreasing the learning rate during training

3 – Dropout

4 – Data Augmentation

5 – Batch normalization

For the case of predicting labels for images, we test it with images from the validation data set. The target labels are either 0 or 1, representing a cat or a dog 
respectively. Since, our neural network model has only 1 output node, the way it predicts images is for a 0 label image, the output prediction is a very small number 
and the predicted output for an image with label 1 is very close to 1.

For the feature map outputs, we see that the convolution layers deep inside the CNN detect patterns of edges in each of the images. This is because as we go deep 
into the CNN the image becomes smaller and smaller and only the most dominant and unique features remain. This helps the filter applied to the images to detect the 
features which will differentiate between a cat and a dog as well as to identify features that represent images of the same class.

![alt text](https://github.com/rohitmurali8/CNN-Image-Classification/blob/master/convolution1.PNG)

![alt text](https://github.com/rohitmurali8/CNN-Image-Classification/blob/master/convolution2.PNG)

![alt text](https://github.com/rohitmurali8/CNN-Image-Classification/blob/master/convolution3.PNG)

