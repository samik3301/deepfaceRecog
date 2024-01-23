# Deep Face based Advanced Facial Recognition

### For testing the repository and local build

Recommended to test this in a python virtual environment.

*Run the following command.*

`git clone https://github.com/samik3301/deepfaceRecog`

`pip install -r requirements.txt`

### How to run the project?

To test this project, unique person image data can be placed inside `deepfacerecog/db/`. Upto 5 different images of the same person can be added to the database.

`target.img` is the testing data, which will be tested against the known labeled data in the database. Change this to test against different data from different people.

Navigate into the project directory after cloning the project and then run, `deepface.ipynb`. 

### Methodology and Working Explained: 

The deepface repository: [deepface Repository](https://github.com/serengil/deepface)

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

### Progress : 

-Checking with live video feed and face tracking functionality using OpenCV, image frame saving on command with user input label inside correct directory. 

Using the `DeepFace.Stream()` method, which basically makes internal call to other numerous functions to first execute face detection and then uses Face Recognition model to extract the Facial Embeddings. These are then checked with the Facial embeddings within the Database provided and if a match if found then it shows the Label along with the Detected Face.

Stream function will access your webcam and apply both face recognition and facial attribute analysis. The function starts to analyze a frame if it can focus a face sequentially 5 frames. Then, it shows results 5 seconds. 

*Some Important arguments that are to be configured according to our use case:*

```enable_facial_analysis (boolean): Set this to False to just run face recognition```

```source: Set this to 0 for access web cam. Otherwise, pass exact video path.```

```time_threshold (int): how many second analyzed image will be displayed```

```frame_threshold (int): how many frames required to focus on face```

Have kept `time_threshold=2` to reduce the latency as the detector model used is accurate in nature and `frame_threshold=1` to again reduce latency. The higher the frame_threshold, the more the latency on the live video stream analysis. 

Pros : 
1. This works well with single shot learning and Few shot learning. 
2. Highly accurate in terms of face detection. The `detector_backend` argument can be changed to other existing methodologies to compare and customize.
3. Different Face Recognition models which are pretrained on existing datasets can be used, the argument `model_name` provides us with lot of options for selecting a pretrained model. Although, we have to decide which model is suitable for our use case as there is a Speed vs Accuracy tradeoff to a certain extent. 

Cons: 
1. The multi- face detection and multi - face recognition, that is face recognition capability of more than one person in a frame is not possible with this package.
2. The real time video processing feed works with low FPS, as it has to first focus for a certain number of frames to detect the face properly and then uses it as input to the Facial Recognition models which converts it into Facial embeddings to be checked against the Database facial embeddings using similarity metrics. This workflow increases the latency in real time processing.

### Testing pending and in developement-

-Multi class detection and recognition using this module is pending currently.

-The accuracy and other metrics after testing with noisy data and albumentations of the data is remaining.

**This was tested on Macbook M1 Air CPU**

Python Version 3.10.6
