# Phone-price-classification

***

## Goal of the project

Picking the price of a mobile phone is not an easy task, especially if you are neither the producer of the phone nor the first hand vendor. Therefore, there is a need to distinguish a way of finding the appropriate price for each mobile phone device. In general each mobile phone can be assigned to one of these four price ranges: low, medium, high, and very high. Moreover, each mobile phone comes with a set of features and specifications such as the RAM memory, CPU, front and back camera quality, and so on. All these features play a role in determining the mobile phone’s price range. In this project, machine learning techniques will be applied to try and make predictions about the price range of a phone. Various classification techniques, artificial neural networks and hybrid models will be trained and tested using a dataset which is provided by a shop owner and contains the information about the above mentioned features and their corresponding price ranges. All the models will be trained with different normalized versions of the dataset and with different parameters or structures (in case of MLP, Ensembles) to see in which conditions they perform better. The aim of the project is to find:
Under which normalization and feature engineering each model works the best
Which model has the best accuracy at predicting price ranges
The models with better accuracy can then be utilized for the prediction of the price range of real life mobile phones by simply providing their features. 

## Methodology 

In this project, various supervised learning techniques will be used on a dataset of mobile phone instances with the aim of achieving high accuracy prediction of the price range for a mobile phone. The project mainly consists in three steps:
Data manipulation which includes feature engineering and application of various normalization techniques.
Training each model with specific parameters or structures (eg. trying MLP with different number of nodes in the hidden layer)
Testing the model and calculating accuracy metrics to see which of the models perform better, and under which conditions does each model perform the best.
Taking the best performing models and applying them to a real life app.
Testing and training data are taken from the same dataset using python libraries to do the random selection. For every model and for the data manipulation steps python will be used. Mainly the sklearn, pandas, and tensorflow libraries. Other libraries might be employed as well for auxiliary tasks.

## Results

For the general results and conclusions of the project refer to this document: https://docs.google.com/document/d/1WMHTixsGzPBncYrOVy9_4B2drAg4eB2AfYZwkwCodnI/edit?usp=sharing

## GUI
The user interface of this program was developed using python and especially the tkinter library. More information on the library is found at: https://docs.python.org/3/library/tkinter.html . The goal of this interface is to allow users with no experience in AI to choose a machine learning model and perform predictions without any knowledge about the internal implementation. The user is provided the predicted price range and the accuracy of prediction each time.
<p align="center">
  <img width="955" alt="image" src="https://github.com/guy1998/Phone-price-classification/assets/104024859/a141392b-a583-4449-96ca-5320d43d3b19"> <br>
  <p align="left">
    As it can be seen here the main page of the program allows the user to pick the machine learning model they prefer. It is a pretty straightforward menu containing only buttons with the name of the model on them. After a model is selected the program navigates to the prediction page that can be seen below.
  </p>
  <img width="960" alt="image" src="https://github.com/guy1998/Phone-price-classification/assets/104024859/ad00251f-cf49-44fd-99b7-4391647ce94a"> <br>
  <p align="left">
  On the prediction page on the left of the screen user can apply all the parameters of this new instance of a mobile phone such as battery power, RAM, clock speed, camera, and so on. The user is provided with the information on the expected prediction that is written in the text box on the right. Once the user fills all the fields he can press predict and that will automatically start the model's work to provide a prediction. When the execution finishes an alert box containing the prediction and the accuracy of this prediction is shown to the user as can be seen in the picture.
  </p>
</p>
