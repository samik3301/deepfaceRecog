# Deep Face based Advanced Facial Recognition

### For testing the repository and local build

Recommended to test this in a python virtual environment.

*Run the following command.*

`git clone [repo-name].git`

`pip install -r requirements.txt`

### How to run the project?

To test this project, unique person image data can be placed inside `deepfacerecog/db/`. Upto 5 different images of the same person can be added to the database.

`target.img` is the testing data, which will be tested against the known labeled data in the database. Change this to test against different data from different people.

Navigate into the project directory after cloning the project and then run, `deepface.ipynb`. 

### Methodology and Working Explained: 

[deepface](https://github.com/serengil/deepface)

This repository is very well maintained and can be used for our use case. Other facial recognition repositories and APIs are outdated and inefficient, performance wise.

This is an implementation of deepface face recognition module, which hosts variety of models and is build upon a collection of different face recognition models. According to our use case, we can go through the list of models. The default configuration uses VGG-Face model.

` models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]` 

The time taken for the model to find the most similar match from the database can be observed and the best model can be fixed through that metric. 

**Note: Discrepancy from the official documentation on repository, the `Deepface.find()` method returns a list, NOT a dataframe object.**

### Testing pending and in developement-
-Multi class detection and recognition using this module is pending currently.
-The accuracy and other metrics after testing with noisy data and albumentations of the data is remaining.
-Checking with live video feed and face tracking functionality using OpenCV, image frame saving on command with user input label inside correct directory.

**This was tested on Macbook M1 Air CPU**
Python Version 3.10.6
